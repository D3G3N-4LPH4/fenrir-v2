#!/usr/bin/env python3
"""
FENRIR - Structured Output Schemas and Sanitizer Fallback

Provides JSON Schema definitions for all LLM calls and a two-stage
parse pipeline:

    1. Prefer message["parsed"] from structured output (json_schema mode)
    2. Extract JSON from message["content"] (strip markdown fences)
    3. Call a cheap sanitizer model to coerce malformed output into schema

Replaces the fragile response.find("{") pattern throughout decision_engine.py.
Mirrors Nocturne's _sanitize_output() / response_format pattern.
"""

import json
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ─────────────────────────────────────────────────────────────────────────────
#  JSON Schemas
# ─────────────────────────────────────────────────────────────────────────────

ENTRY_ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["STRONG_BUY", "BUY", "SKIP", "AVOID"]},
        "confidence": {"type": "number"},
        "risk_score": {"type": "number"},
        "reasoning": {"type": "string"},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "green_flags": {"type": "array", "items": {"type": "string"}},
        "suggested_buy_amount_sol": {"type": ["number", "null"]},
        "suggested_stop_loss_pct": {"type": ["number", "null"]},
        "suggested_take_profit_pct": {"type": ["number", "null"]},
        "social_score": {"type": ["number", "null"]},
        "liquidity_score": {"type": ["number", "null"]},
        "holder_score": {"type": ["number", "null"]},
        "timing_score": {"type": ["number", "null"]},
    },
    "required": [
        "decision", "confidence", "risk_score", "reasoning",
        "red_flags", "green_flags",
    ],
    "additionalProperties": False,
}

EXIT_ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["HOLD", "TAKE_PROFIT", "EXIT", "OVERRIDE_HOLD"]},
        "reasoning": {"type": "string"},
        "urgency": {"type": "number"},
        # Nocturne pattern: AI encodes hold conditions + optional cooldown_until timestamp.
        # Example: "Hold while PnL > 80%. cooldown_until: 2025-10-19T15:55Z"
        "exit_plan": {"type": "string"},
    },
    "required": ["action", "reasoning", "urgency", "exit_plan"],
    "additionalProperties": False,
}

BATCHED_EXIT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "exit_decisions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "token_address": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["HOLD", "TAKE_PROFIT", "EXIT", "OVERRIDE_HOLD"],
                    },
                    "reasoning": {"type": "string"},
                    "urgency": {"type": "number"},
                    # Stores AI's own continuation contract: what it will check next cycle.
                    # May include "cooldown_until: <ISO>" to suppress re-evaluation.
                    "exit_plan": {"type": "string"},
                },
                "required": [
                    "token_address", "action", "reasoning", "urgency", "exit_plan"
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reasoning", "exit_decisions"],
    "additionalProperties": False,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_response_format(schema_name: str, schema: dict) -> dict:
    """Build an OpenRouter response_format dict for strict structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def extract_json(raw: str) -> Any:
    """
    Best-effort JSON extraction from an LLM content string.

    Priority:
    1. Direct json.loads() on stripped text
    2. Strip markdown fences (```json ... ```) then parse
    3. Find first { ... last } and parse the substring
    """
    if not raw:
        return None

    clean = raw.strip()

    # Strip markdown fences
    for fence in ("```json", "```"):
        if fence in clean:
            clean = clean.replace(fence, "").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Last resort: find outermost braces
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Sanitizer fallback
# ─────────────────────────────────────────────────────────────────────────────

async def sanitize_with_llm(
    raw_content: str,
    schema: dict,
    schema_name: str,
    session: aiohttp.ClientSession,
    api_key: str,
    sanitize_model: str = "openai/gpt-4o-mini",
    openrouter_url: str = OPENROUTER_URL,
) -> Any | None:
    """
    Coerce malformed LLM output into the required schema using a cheap
    secondary model call. Mirrors Nocturne's _sanitize_output().

    Called only when the primary parse fails — typically costs <$0.001.

    Returns:
        Parsed dict on success, None on failure.
    """
    try:
        payload = {
            "model": sanitize_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON normalizer. "
                        "Return ONLY a JSON object matching the provided JSON Schema. "
                        "Fix prose, markdown wrapping, or missing fields. "
                        "Do not add fields not in the schema. "
                        "Do not include any preamble or backticks."
                    ),
                },
                {"role": "user", "content": raw_content},
            ],
            "response_format": build_response_format(schema_name, schema),
            "temperature": 0,
            "max_tokens": 1000,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with session.post(openrouter_url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                logger.warning(
                    "Sanitizer call failed: status=%s schema=%s", resp.status, schema_name
                )
                return None
            data = await resp.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            # Prefer parsed field (returned by structured output capable providers)
            parsed = msg.get("parsed")
            if isinstance(parsed, dict):
                return parsed
            # Fall back to content extraction
            return extract_json(msg.get("content", ""))
    except Exception as exc:
        logger.warning("Sanitizer exception for schema '%s': %s", schema_name, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Two-stage parse pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def parse_or_sanitize(
    message: dict,
    schema: dict,
    schema_name: str,
    session: aiohttp.ClientSession,
    api_key: str,
    sanitize_model: str = "openai/gpt-4o-mini",
) -> Any | None:
    """
    Two-stage parse pipeline for LLM response messages.

    Stage 1: Check message["parsed"] — available when provider supports structured output.
    Stage 2: extract_json(message["content"]) — strips markdown, finds {} bounds.
    Stage 3: sanitize_with_llm() — cheap secondary model call as last resort.

    Args:
        message:        Single message dict from response["choices"][0]["message"].
        schema:         JSON Schema dict (e.g. ENTRY_ANALYSIS_SCHEMA).
        schema_name:    Human-readable schema name for the sanitizer prompt.
        session:        aiohttp.ClientSession to use for the sanitizer call.
        api_key:        OpenRouter API key.
        sanitize_model: Cheap model for the sanitizer fallback.

    Returns:
        Parsed dict, or None if all three stages fail.
    """
    # Stage 1: structured output parsed field (zero-cost, already parsed by provider)
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return parsed

    # Stage 2: extract JSON from content string
    content = message.get("content") or ""
    result = extract_json(content)
    if isinstance(result, dict):
        return result

    # Stage 3: sanitizer fallback
    logger.warning(
        "Primary JSON parse failed; calling sanitizer for schema '%s'. "
        "Content preview: %.120s",
        schema_name,
        content,
    )
    return await sanitize_with_llm(
        raw_content=content,
        schema=schema,
        schema_name=schema_name,
        session=session,
        api_key=api_key,
        sanitize_model=sanitize_model,
    )
