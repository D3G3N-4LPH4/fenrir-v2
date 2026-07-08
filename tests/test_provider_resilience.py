#!/usr/bin/env python3
"""
Tests for fenrir/ai/provider_resilience.py

Covers the §3 harness port (browser-use/terminal → FENRIR):
  - classify_error: terminal / transient / overflow ordering
  - backoff_delay: exponential schedule, capped
  - post(): transient 429/5xx + network errors retried with backoff
  - post(): terminal errors (401/400-schema) NOT retried
  - post(): overflow NOT retried (no compaction path in FENRIR)
  - post(): capability degradation (structured-output rejection) still works
  - post(): deadline_s guard skips a retry that wouldn't fit the budget
  - CircuitBreaker sees exactly ONE failure per logical call, not per attempt
"""

from types import SimpleNamespace
from typing import cast

import aiohttp
import pytest

from fenrir.ai.provider_resilience import (
    ModelUnavailableError,
    ProviderResilientCaller,
    backoff_delay,
    classify_error,
    is_model_unavailable,
)
from fenrir.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# ═══════════════════════════════════════════════════════════════════
#  Fakes — minimal aiohttp session/response doubles
# ═══════════════════════════════════════════════════════════════════


class FakeResp:
    def __init__(self, status, body="", json_data=None):
        self.status = status
        self._body = body
        self._json = (
            json_data if json_data is not None else {"choices": [{"message": {"content": "{}"}}]}
        )
        self.request_info = SimpleNamespace(real_url="http://test", method="POST")
        self.history = ()

    async def text(self):
        return self._body

    async def json(self):
        return self._json


class FakeCM:
    """Async context manager standing in for session.post(...)."""

    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self._resp

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def post(self, url, headers=None, json=None):  # noqa: A002 - mirror aiohttp
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            return FakeCM(exc=outcome)
        return FakeCM(resp=outcome)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Make backoff instantaneous so tests don't actually wait."""

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("fenrir.ai.provider_resilience.asyncio.sleep", fake_sleep)


def make_caller(outcomes, breaker=None):
    session = FakeSession(outcomes)
    caller = ProviderResilientCaller(  # noqa: S106
        api_key="k", session=cast(aiohttp.ClientSession, session), breaker=breaker
    )
    return caller, session


BASE_PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


# ═══════════════════════════════════════════════════════════════════
#  classify_error
# ═══════════════════════════════════════════════════════════════════


def test_classify_terminal_status():
    assert classify_error(401, "") == "terminal"
    assert classify_error(403, "") == "terminal"
    assert classify_error(400, "bad request") == "terminal"


def test_classify_transient_status():
    assert classify_error(429, "") == "transient"
    assert classify_error(503, "") == "transient"
    assert classify_error(502, "") == "transient"


def test_classify_overflow_beats_status():
    # A 400 whose body mentions a context overflow → overflow, not terminal.
    assert classify_error(400, "maximum context length is 8192 tokens") == "overflow"


def test_classify_terminal_beats_transient_substring():
    # Ordering trick: a terminal marker present alongside a transient word
    # must classify terminal so the retry path never touches it.
    assert classify_error(None, "invalid_request_error: overloaded") == "terminal"


def test_classify_network_string_fallback():
    assert classify_error(None, "Connection reset by peer") == "transient"
    assert classify_error(None, "request timed out") == "transient"


def test_classify_unknown_defaults_terminal():
    assert classify_error(None, "something weird") == "terminal"


# ═══════════════════════════════════════════════════════════════════
#  backoff_delay
# ═══════════════════════════════════════════════════════════════════


def test_backoff_schedule():
    assert backoff_delay(1) == pytest.approx(0.200)
    assert backoff_delay(2) == pytest.approx(0.400)
    assert backoff_delay(3) == pytest.approx(0.800)


def test_backoff_capped():
    # Caps at 2^5 × base = 6.4s regardless of how high the attempt goes.
    assert backoff_delay(50) == pytest.approx(0.200 * 32)


# ═══════════════════════════════════════════════════════════════════
#  post() — transient retry
# ═══════════════════════════════════════════════════════════════════


async def test_transient_503_then_success():
    ok = FakeResp(200, json_data={"choices": [{"message": {"content": "ok"}}]})
    caller, session = make_caller([FakeResp(503, "overloaded"), ok])

    result = await caller.post(payload=BASE_PAYLOAD)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert session.calls == 2  # one failure + one retry that succeeded


async def test_network_error_then_success():
    ok = FakeResp(200, json_data={"choices": [{"message": {"content": "ok"}}]})
    caller, session = make_caller([aiohttp.ClientConnectionError("connection reset"), ok])

    result = await caller.post(payload=BASE_PAYLOAD)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert session.calls == 2


async def test_transient_budget_exhausted_raises():
    caller, session = make_caller(
        [FakeResp(503, "overloaded"), FakeResp(503, "overloaded"), FakeResp(503, "overloaded")]
    )

    with pytest.raises(aiohttp.ClientResponseError):
        await caller.post(payload=BASE_PAYLOAD, transient_retries=2)

    assert session.calls == 3  # initial + 2 retries, then give up


# ═══════════════════════════════════════════════════════════════════
#  post() — terminal / overflow are NOT retried
# ═══════════════════════════════════════════════════════════════════


async def test_terminal_401_not_retried():
    caller, session = make_caller([FakeResp(401, "incorrect api key")])

    with pytest.raises(aiohttp.ClientResponseError):
        await caller.post(payload=BASE_PAYLOAD)

    assert session.calls == 1  # no retry on terminal


async def test_overflow_not_retried():
    caller, session = make_caller([FakeResp(400, "maximum context length exceeded")])

    with pytest.raises(aiohttp.ClientResponseError):
        await caller.post(payload=BASE_PAYLOAD)

    assert session.calls == 1


# ═══════════════════════════════════════════════════════════════════
#  post() — capability degradation still works
# ═══════════════════════════════════════════════════════════════════


async def test_structured_output_degradation():
    err_body = '{"error": {"message": "response_format not supported"}}'
    ok = FakeResp(200, json_data={"choices": [{"message": {"content": "ok"}}]})
    caller, session = make_caller([FakeResp(400, err_body), ok])

    rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    result = await caller.post(payload=BASE_PAYLOAD, response_format=rf)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert caller.allow_structured is False  # disabled for the session
    assert session.calls == 2  # immediate retry, no backoff consumed


# ═══════════════════════════════════════════════════════════════════
#  post() — deadline guard
# ═══════════════════════════════════════════════════════════════════


async def test_deadline_skips_retry():
    # deadline_s=0 → no backoff sleep can ever fit → first transient error
    # is raised immediately without retrying.
    caller, session = make_caller([FakeResp(503, "overloaded")])

    with pytest.raises(aiohttp.ClientResponseError):
        await caller.post(payload=BASE_PAYLOAD, transient_retries=5, deadline_s=0.0)

    assert session.calls == 1


# ═══════════════════════════════════════════════════════════════════
#  CircuitBreaker interaction — one failure per call, not per attempt
# ═══════════════════════════════════════════════════════════════════


async def test_breaker_records_single_failure_per_call():
    breaker = CircuitBreaker(
        "TEST", CircuitBreakerConfig(failure_threshold=3, recovery_timeout_s=60.0)
    )
    # 3 transient failures within ONE post() call (initial + 2 retries).
    caller, session = make_caller([FakeResp(503, "overloaded")] * 3, breaker=breaker)

    with pytest.raises(aiohttp.ClientResponseError):
        await caller.post(payload=BASE_PAYLOAD, transient_retries=2)

    # Despite 3 HTTP attempts, the breaker should see exactly one failure,
    # so a single flaky call cannot trip a 3-failure threshold.
    assert breaker.is_closed
    assert breaker.get_stats()["total_failures"] == 1


async def test_breaker_records_success():
    breaker = CircuitBreaker("TEST", CircuitBreakerConfig())
    ok = FakeResp(200, json_data={"choices": [{"message": {"content": "ok"}}]})
    caller, session = make_caller([FakeResp(503, "overloaded"), ok], breaker=breaker)

    await caller.post(payload=BASE_PAYLOAD)

    assert breaker.get_stats()["total_successes"] == 1
    assert breaker.get_stats()["total_failures"] == 0


# ═══════════════════════════════════════════════════════════════════
#  Credit-aware model fallback
# ═══════════════════════════════════════════════════════════════════

CREDIT_BODY = '{"error":{"message":"Your credit balance is too low","code":400}}'
NO_PROVIDER_BODY = '{"error":{"message":"No allowed providers are available","code":404}}'


class RecordingSession(FakeSession):
    """FakeSession that records the model on each request payload."""

    def __init__(self, outcomes):
        super().__init__(outcomes)
        self.models_seen: list = []

    def post(self, url, headers=None, json=None):  # noqa: A002 - mirror aiohttp
        self.models_seen.append((json or {}).get("model"))
        return super().post(url, headers=headers, json=json)


def make_fallback_caller(outcomes, models, reprobe_interval_s=300.0):
    session = RecordingSession(outcomes)
    caller = ProviderResilientCaller(  # noqa: S106
        api_key="k",
        session=cast(aiohttp.ClientSession, session),
        models=models,
        reprobe_interval_s=reprobe_interval_s,
    )
    return caller, session


class TestIsModelUnavailable:
    def test_402_is_unavailable(self):
        assert is_model_unavailable(402, "") is True

    def test_credit_balance_body(self):
        assert is_model_unavailable(400, "your Credit Balance is too low") is True

    def test_no_providers_body(self):
        assert is_model_unavailable(404, "No allowed providers are available") is True

    def test_auth_error_is_not_unavailable(self):
        assert is_model_unavailable(401, "invalid api key") is False

    def test_schema_error_is_not_unavailable(self):
        assert is_model_unavailable(400, "invalid response_format schema") is False


class TestModelFallback:
    @pytest.mark.asyncio
    async def test_falls_over_on_credit_error(self):
        caller, session = make_fallback_caller(
            [FakeResp(400, CREDIT_BODY), FakeResp(200)],
            models=["primary", "fallback"],
        )
        out = await caller.post(dict(BASE_PAYLOAD))
        assert out["choices"][0]["message"]["content"] == "{}"
        assert session.models_seen == ["primary", "fallback"]
        assert caller._preferred_idx == 1  # stuck to the winner

    @pytest.mark.asyncio
    async def test_falls_over_on_no_provider(self):
        caller, session = make_fallback_caller(
            [FakeResp(404, NO_PROVIDER_BODY), FakeResp(200)],
            models=["primary", "fallback"],
        )
        await caller.post(dict(BASE_PAYLOAD))
        assert session.models_seen == ["primary", "fallback"]

    @pytest.mark.asyncio
    async def test_sticky_skips_dead_primary_next_call(self):
        caller, session = make_fallback_caller(
            [FakeResp(400, CREDIT_BODY), FakeResp(200), FakeResp(200)],
            models=["primary", "fallback"],
        )
        await caller.post(dict(BASE_PAYLOAD))  # primary dies -> fallback
        await caller.post(dict(BASE_PAYLOAD))  # starts at fallback, no primary hit
        assert session.models_seen == ["primary", "fallback", "fallback"]

    @pytest.mark.asyncio
    async def test_all_unavailable_raises(self):
        caller, _ = make_fallback_caller(
            [FakeResp(400, CREDIT_BODY), FakeResp(402, "payment required")],
            models=["a", "b"],
        )
        with pytest.raises(ModelUnavailableError):
            await caller.post(dict(BASE_PAYLOAD))

    @pytest.mark.asyncio
    async def test_non_credit_terminal_does_not_fall_over(self):
        # A bad key fails on every model -> abort, don't waste a fallback call.
        caller, session = make_fallback_caller(
            [FakeResp(401, "invalid api key")],
            models=["primary", "fallback"],
        )
        with pytest.raises(aiohttp.ClientResponseError):
            await caller.post(dict(BASE_PAYLOAD))
        assert session.models_seen == ["primary"]

    @pytest.mark.asyncio
    async def test_reprobe_resets_to_primary(self):
        caller, session = make_fallback_caller(
            [FakeResp(400, CREDIT_BODY), FakeResp(200), FakeResp(200)],
            models=["primary", "fallback"],
            reprobe_interval_s=0.0,  # re-probe on every call
        )
        await caller.post(dict(BASE_PAYLOAD))  # primary dies -> fallback
        assert caller._preferred_idx == 1
        await caller.post(dict(BASE_PAYLOAD))  # reprobe -> primary works again
        assert session.models_seen == ["primary", "fallback", "primary"]
        assert caller._preferred_idx == 0

    @pytest.mark.asyncio
    async def test_no_models_uses_payload_model(self):
        caller, session = make_fallback_caller([FakeResp(200)], models=[])
        await caller.post(dict(BASE_PAYLOAD))
        assert session.models_seen == ["m"]  # backwards-compatible
