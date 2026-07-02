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

import asyncio
import json
import logging
import time
from typing import Any, cast

import aiohttp

from fenrir.core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Error classification (ported from browser-use/terminal harness, §3) ──────
# Ordering is the trick: TERMINAL/OVERFLOW are checked before TRANSIENT so an
# error that matches both (e.g. a 400 carrying "overloaded") is never retried.
# Without that ordering a permanently-failing request would burn the whole
# transient budget hammering something that can never succeed.
TERMINAL_MARKERS = (
    "incorrect api key",
    "invalid api key",
    "401",
    "403",
    "content was flagged",
    "content_policy",
    "invalid_request_error",
    "context_length_exceeded",
    "schema",
    "unsupported",
)
TRANSIENT_MARKERS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection closed",
    "connection aborted",
    "overloaded",
    "rate limit",
    "too many requests",
    "eof",
    "502",
    "503",
    "504",
    "gateway",
    "stream disconnected",
    "server error",
    "temporarily unavailable",
)
# Overflow is terminal-with-a-special-name. browser-use/terminal routes it to
# compaction; FENRIR's prompt is bounded by construction (≤ai_memory_size
# decisions + one token), so there is no compaction path — we treat overflow as
# non-retryable and surface it loudly rather than silently looping.
OVERFLOW_MARKERS = (
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
    "token limit",
    "input too long",
)

# Status codes that are unambiguous on their own (body string is the fallback).
_TERMINAL_STATUS = frozenset({400, 401, 403, 404, 422})
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Backoff: 200ms × 2^min(attempt-1, 5) → 0.2, 0.4, 0.8 … capped ~6.4s.
_BACKOFF_BASE_S = 0.200
_BACKOFF_CAP_EXP = 5


def classify_error(status: int | None, body: str) -> str:
    """
    Classify a failed provider call as 'terminal' | 'transient' | 'overflow'.

    Overflow and terminal are checked before transient so an error that also
    contains a transient-looking substring is never plain-retried. Status codes
    take precedence when unambiguous; the body string is the fallback and the
    only signal for network-level exceptions where ``status`` is None.
    Unknown errors default to terminal — don't burn retries on mysteries.
    """
    b = body.lower()

    if any(m in b for m in OVERFLOW_MARKERS):
        return "overflow"

    if status in _TERMINAL_STATUS:
        return "terminal"
    if status in _TRANSIENT_STATUS:
        return "transient"

    if any(m in b for m in TERMINAL_MARKERS):
        return "terminal"
    if any(m in b for m in TRANSIENT_MARKERS):
        return "transient"

    return "terminal"


def backoff_delay(attempt: int) -> float:
    """Exponential backoff (seconds) for the Nth transient retry (1-indexed)."""
    exp = min(max(attempt - 1, 0), _BACKOFF_CAP_EXP)
    return _BACKOFF_BASE_S * (1 << exp)


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
        breaker: CircuitBreaker | None = None,
    ):
        self.api_key = api_key
        self.session = session
        self.url = url
        self.http_referer = http_referer
        self.app_title = app_title
        self._breaker = breaker

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
        transient_retries: int = 2,
        deadline_s: float | None = None,
    ) -> dict[str, Any]:
        """
        POST to OpenRouter with capability degradation AND transient retry.

        Two independent retry mechanisms layered here:

          1. **Capability degradation** (``max_retries``): a provider that
             rejects structured output or a tool schema gets retried
             *immediately* with that capability stripped, for the lifetime of
             the session. No backoff — the next call is expected to succeed.
          2. **Transient backoff** (``transient_retries``): a 429 / 5xx /
             network-level failure (timeout, connection reset, "overloaded")
             gets retried with exponential backoff. Terminal errors (auth,
             schema, content policy) and overflow are *never* retried — they're
             classified out by :func:`classify_error` before the retry path.

        This sits *beneath* the CircuitBreaker: the breaker sees exactly one
        ``record_failure`` per logical call (when we ultimately give up), not
        one per attempt, so a single flaky call can't trip the breaker.

        Args:
            payload:           Base payload: {"model": ..., "messages": [...], ...}.
                               Do NOT include response_format or tools here; pass
                               them separately so this method can strip them on retry.
            response_format:   Structured output schema from build_response_format().
                               Applied only when self.allow_structured is True.
            tools:             Tool definition list. Applied only when allow_tools.
            tool_choice:       "auto" | "none" | specific tool name.
            max_retries:       Max capability-degradation attempts.
            transient_retries: Max backoff retries for transient errors.
            deadline_s:        Optional wall-clock budget. A backoff sleep is
                               skipped (and the error raised) when it would push
                               total elapsed time past this deadline. Pass the
                               caller's timeout so latency-critical paths (entry
                               sniping) don't retry past their cancellation point.

        Returns:
            Raw JSON response dict from OpenRouter.

        Raises:
            aiohttp.ClientResponseError / network error: after retries exhausted.
        """
        if self._breaker:
            self._breaker.check()

        headers = self._headers()
        start = time.monotonic()
        degraded_attempts = 0
        transient_attempts = 0
        # Hard safety cap so no classification bug can spin forever.
        hard_cap = max_retries + transient_retries + 2

        for _ in range(hard_cap):
            # Build the final request for this attempt
            request: dict = dict(payload)
            if response_format and self.allow_structured:
                request["response_format"] = response_format
            if tools and self.allow_tools:
                request["tools"] = tools
                request["tool_choice"] = tool_choice

            # ── Issue the request ──────────────────────────────────────────
            try:
                async with self.session.post(self.url, headers=headers, json=request) as resp:
                    if resp.status == 200:
                        if self._breaker:
                            self._breaker.record_success()
                        return cast(dict[str, Any], await resp.json())
                    status = resp.status
                    error_text = await resp.text()
                    req_info = resp.request_info
                    history = resp.history
            except (TimeoutError, aiohttp.ClientError) as exc:
                # Network-level failure: no HTTP response (timeout, reset, …).
                # We raise ClientResponseError ourselves only AFTER the loop, so
                # the only ClientErrors caught here are genuine transport faults.
                err_str = f"{type(exc).__name__}: {exc}"
                if transient_attempts < transient_retries and self._sleep_fits(
                    start, transient_attempts + 1, deadline_s
                ):
                    transient_attempts += 1
                    await self._backoff_sleep(
                        transient_attempts, transient_retries, f"network error ({err_str})"
                    )
                    continue
                if self._breaker:
                    self._breaker.record_failure(err_str[:120])
                raise

            # ── Non-200: try capability degradation first ──────────────────
            try:
                err_json = json.loads(error_text)
            except json.JSONDecodeError:
                err_json = {}

            if degraded_attempts < max_retries and self._maybe_degrade(status, err_json):
                degraded_attempts += 1
                continue  # retry immediately with degraded flags (no backoff)

            # ── Not degradable → classify and decide ───────────────────────
            kind = classify_error(status, error_text)
            if (
                kind == "transient"
                and transient_attempts < transient_retries
                and self._sleep_fits(start, transient_attempts + 1, deadline_s)
            ):
                transient_attempts += 1
                await self._backoff_sleep(transient_attempts, transient_retries, f"HTTP {status}")
                continue

            # Terminal, overflow, or transient budget exhausted → give up.
            if self._breaker:
                self._breaker.record_failure(f"HTTP {status} ({kind})")
            if kind == "overflow":
                logger.error(
                    "Provider context overflow (HTTP %s) — not retryable without "
                    "compaction. Reduce ai_memory_size or prompt size.",
                    status,
                )
            raise aiohttp.ClientResponseError(
                req_info, history, status=status, message=error_text[:200]
            )

        # Exhausted the hard safety cap (should be unreachable in practice).
        if self._breaker:
            self._breaker.record_failure("retry_loop_exhausted")
        raise RuntimeError(
            "ProviderResilientCaller: retry loop exhausted "
            f"(degraded={degraded_attempts}, transient={transient_attempts})"
        )

    # ── Retry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sleep_fits(start: float, next_attempt: int, deadline_s: float | None) -> bool:
        """True if the next backoff sleep fits within the wall-clock deadline."""
        if deadline_s is None:
            return True
        elapsed = time.monotonic() - start
        return elapsed + backoff_delay(next_attempt) < deadline_s

    @staticmethod
    async def _backoff_sleep(attempt: int, budget: int, reason: str) -> None:
        """Log and sleep for the Nth transient retry's backoff interval."""
        delay = backoff_delay(attempt)
        logger.warning(
            "Provider transient failure (%s); retry %s/%s after %.0fms",
            reason,
            attempt,
            budget,
            delay * 1000,
        )
        await asyncio.sleep(delay)

    def _maybe_degrade(self, status: int, err_json: dict) -> bool:
        """
        Disable a provider capability that was just rejected, if applicable.

        Mutates self.allow_tools / self.allow_structured for the session and
        returns True when a capability was disabled (caller should retry).
        """
        meta = (err_json.get("error") or {}).get("metadata", {}) or {}
        raw_meta = meta.get("raw", "") or ""
        provider = meta.get("provider_name", "") or ""

        # ── xAI tool schema rejection ──────────────────────────────────────
        if (
            status == 422
            and provider.lower().startswith("xai")
            and "deserialize" in raw_meta.lower()
            and self.allow_tools
        ):
            logger.warning("xAI rejected tool schema; disabling tools for this session.")
            self.allow_tools = False
            return True

        # ── Structured output rejection ────────────────────────────────────
        if status in (400, 422) and self.allow_structured:
            err_str = json.dumps(err_json).lower()
            if any(kw in err_str for kw in ("response_format", "json_schema", "structured")):
                logger.warning(
                    "Provider rejected structured output (status=%s); "
                    "disabling for this session.",
                    status,
                )
                self.allow_structured = False
                return True

        return False
