#!/usr/bin/env python3
"""
FENRIR - Provider-Resilient API Caller

Wraps OpenRouter POST calls with automatic capability degradation,
mirroring Nocturne's allow_structured / allow_tools retry pattern.

Degradation rules:
  - Provider returns 422 + xAI "deserialize" error  → disable tools for session
  - Provider returns 400/422 + "response_format"    → disable structured output for session
  - Flags persist per-instance so subsequent calls skip known-failing configs

Usage:
    caller = ProviderResilientCaller(api_key=..., session=...)

    response = await caller.post(
        payload={"model": "...", "messages": [...]},
        response_format=build_response_format("entry_analysis", ENTRY_ANALYSIS_SCHEMA),
        tools=MY_TOOLS,
    )
    message = response["choices"][0]["message"]
"""

import json
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ProviderResilientCaller:
    """
    Session-level OpenRouter caller that tracks provider capability flags.

    Once a provider rejects structured outputs or tools, those flags are
    disabled for the lifetime of this caller so subsequent calls don't
    waste a round-trip retrying known-failing configurations.

    Thread safety: not thread-safe; designed for single-asyncio-task use.
    """

    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession,
        url: str = OPENROUTER_URL,
        http_referer: str | None = None,
        app_title: str | None = None,
    ):
        self.api_key = api_key
        self.session = session
        self.url = url
        self.http_referer = http_referer
        self.app_title = app_title

        # Degradation state — survives across calls in this session
        self.allow_structured: bool = True
        self.allow_tools: bool = True

    def _headers(self) -> dict:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            h["HTTP-Referer"] = self.http_referer
        if self.app_title:
            h["X-Title"] = self.app_title
        return h

    async def post(
        self,
        payload: dict,
        response_format: dict | None = None,
        tools: list | None = None,
        tool_choice: str = "auto",
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """
        POST to OpenRouter with automatic structured-output / tool degradation.

        Args:
            payload:         Base payload dict: {"model": ..., "messages": [...], ...}
                             Do NOT include response_format or tools here; pass them
                             as separate arguments so this method can strip them on retry.
            response_format: Structured output schema from build_response_format().
                             Applied only when self.allow_structured is True.
            tools:           Tool definition list. Applied only when self.allow_tools is True.
            tool_choice:     "auto" | "none" | specific tool name.
            max_retries:     Max attempts before raising the last error.

        Returns:
            Raw JSON response dict from OpenRouter.

        Raises:
            aiohttp.ClientResponseError: After all retries are exhausted.
        """
        headers = self._headers()

        for attempt in range(max_retries):
            # Build the final request for this attempt
            request: dict = dict(payload)

            if response_format and self.allow_structured:
                request["response_format"] = response_format
            if tools and self.allow_tools:
                request["tools"] = tools
                request["tool_choice"] = tool_choice

            async with self.session.post(self.url, headers=headers, json=request) as resp:
                if resp.status == 200:
                    return await resp.json()

                error_text = await resp.text()
                try:
                    err_json = json.loads(error_text)
                except json.JSONDecodeError:
                    err_json = {}

                degraded = False

                # ── xAI tool schema rejection ──────────────────────────────
                raw_meta = (
                    (err_json.get("error") or {})
                    .get("metadata", {}) or {}
                ).get("raw", "")
                provider = (
                    (err_json.get("error") or {})
                    .get("metadata", {}) or {}
                ).get("provider_name", "")

                if (
                    resp.status == 422
                    and provider.lower().startswith("xai")
                    and "deserialize" in raw_meta.lower()
                    and self.allow_tools
                ):
                    logger.warning(
                        "xAI rejected tool schema (attempt %s/%s); "
                        "disabling tools for this session.",
                        attempt + 1,
                        max_retries,
                    )
                    self.allow_tools = False
                    degraded = True

                # ── Structured output rejection ────────────────────────────
                if resp.status in (400, 422) and self.allow_structured and not degraded:
                    err_str = json.dumps(err_json)
                    if any(
                        kw in err_str.lower()
                        for kw in ("response_format", "json_schema", "structured")
                    ):
                        logger.warning(
                            "Provider rejected structured output (status=%s, attempt %s/%s); "
                            "disabling for this session.",
                            resp.status,
                            attempt + 1,
                            max_retries,
                        )
                        self.allow_structured = False
                        degraded = True

                if degraded:
                    continue  # retry with degraded flags

                # Unrecoverable error — raise immediately
                resp.raise_for_status()

        raise RuntimeError(
            "ProviderResilientCaller: max_retries=%d exhausted" % max_retries
        )
