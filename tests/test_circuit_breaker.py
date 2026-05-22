#!/usr/bin/env python3
"""
Tests for fenrir/core/circuit_breaker.py

State machine coverage:
  CLOSED  → OPEN after failure_threshold consecutive failures
  OPEN    → raises CircuitOpen immediately
  OPEN    → HALF_OPEN after recovery_timeout_s (time-mocked)
  HALF_OPEN → CLOSED after success_threshold successes
  HALF_OPEN → OPEN again on any failure
  guard() context manager records success/failure and re-raises
  ServiceBreakers registry exposes all four services
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from fenrir.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpen,
    CircuitState,
    ServiceBreakers,
)


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def fast_cfg(
    failure_threshold: int = 3,
    recovery_timeout_s: float = 60.0,
    success_threshold: int = 2,
    half_open_max_calls: int = 1,
) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout_s=recovery_timeout_s,
        success_threshold=success_threshold,
        half_open_max_calls=half_open_max_calls,
    )


def make_cb(**kwargs) -> CircuitBreaker:
    return CircuitBreaker("TEST", fast_cfg(**kwargs))


def trip(cb: CircuitBreaker, n: int, reason: str = "err") -> None:
    for _ in range(n):
        cb.record_failure(reason)


# ═══════════════════════════════════════════════════════════════════
#  CLOSED state
# ═══════════════════════════════════════════════════════════════════

class TestClosed:
    def test_starts_closed(self):
        cb = make_cb()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed

    def test_check_passes_when_closed(self):
        cb = make_cb()
        cb.check()  # no exception

    def test_success_keeps_closed(self):
        cb = make_cb()
        for _ in range(10):
            cb.record_success()
        assert cb.is_closed

    def test_failures_below_threshold_stay_closed(self):
        cb = make_cb(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_closed

    def test_success_resets_failure_counter(self):
        cb = make_cb(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # reset
        cb.record_failure()
        cb.record_failure()
        assert cb.is_closed  # only 2 failures since last success

    def test_threshold_failures_open(self):
        cb = make_cb(failure_threshold=3)
        trip(cb, 3)
        assert cb.state == CircuitState.OPEN

    def test_threshold_is_exact(self):
        cb = make_cb(failure_threshold=5)
        trip(cb, 4)
        assert cb.is_closed
        cb.record_failure()
        assert cb.is_open


# ═══════════════════════════════════════════════════════════════════
#  OPEN state
# ═══════════════════════════════════════════════════════════════════

class TestOpen:
    def test_check_raises_when_open(self):
        cb = make_cb(failure_threshold=1, recovery_timeout_s=999)
        trip(cb, 1)
        with pytest.raises(CircuitOpen) as exc_info:
            cb.check()
        assert exc_info.value.service == "TEST"
        assert exc_info.value.retry_after_s > 0

    def test_is_open_flag(self):
        cb = make_cb(failure_threshold=1)
        trip(cb, 1)
        assert cb.is_open
        assert not cb.is_closed

    def test_check_rejected_increments_counter(self):
        cb = make_cb(failure_threshold=1, recovery_timeout_s=999)
        trip(cb, 1)
        for _ in range(3):
            try:
                cb.check()
            except CircuitOpen:
                pass
        assert cb.get_stats()["total_rejected"] == 3

    def test_transitions_to_half_open_after_timeout(self):
        cb = make_cb(failure_threshold=1, recovery_timeout_s=1.0)
        trip(cb, 1)
        assert cb.is_open

        # Fake the monotonic clock forward past the recovery window
        future = time.monotonic() + 2.0
        with patch("fenrir.core.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = future
            cb._opened_at = future - 2.0  # opened 2s ago
            cb.check()  # should transition to HALF_OPEN, not raise

        assert cb.state == CircuitState.HALF_OPEN

    def test_stays_open_before_timeout(self):
        cb = make_cb(failure_threshold=1, recovery_timeout_s=60.0)
        trip(cb, 1)
        with pytest.raises(CircuitOpen):
            cb.check()
        assert cb.is_open


# ═══════════════════════════════════════════════════════════════════
#  HALF_OPEN state
# ═══════════════════════════════════════════════════════════════════

class TestHalfOpen:
    def _make_half_open(self, **kwargs) -> CircuitBreaker:
        cb = make_cb(failure_threshold=1, recovery_timeout_s=0.0, **kwargs)
        trip(cb, 1)
        # Force elapsed past recovery window
        cb._opened_at = time.monotonic() - 1.0
        cb.check()  # transitions to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        return cb

    def test_success_threshold_closes(self):
        cb = self._make_half_open(success_threshold=2)
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # not yet
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_single_failure_reopens(self):
        cb = self._make_half_open(success_threshold=3)
        cb.record_success()
        cb.record_failure("still broken")
        assert cb.state == CircuitState.OPEN

    def test_reopen_restarts_timer(self):
        cb = self._make_half_open()
        first_opened = cb._opened_at
        cb.record_failure("probe failed")
        # A fresh opened_at should be set
        assert cb._opened_at is not None
        assert cb._opened_at >= first_opened  # type: ignore[operator]

    def test_capacity_limit(self):
        cb = self._make_half_open(half_open_max_calls=1)
        cb._half_open_calls = 1  # simulate one probe in flight
        with pytest.raises(CircuitOpen) as exc_info:
            cb.check()
        assert "HALF_OPEN at capacity" in str(exc_info.value)

    def test_success_then_closed_resets_counters(self):
        cb = self._make_half_open(success_threshold=1)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb._consecutive_failures == 0
        assert cb._consecutive_successes == 0


# ═══════════════════════════════════════════════════════════════════
#  guard() context manager
# ═══════════════════════════════════════════════════════════════════

class TestGuard:
    @pytest.mark.asyncio
    async def test_success_records_success(self):
        cb = make_cb()
        async with cb.guard():
            pass
        assert cb.get_stats()["total_successes"] == 1
        assert cb.get_stats()["total_failures"] == 0

    @pytest.mark.asyncio
    async def test_exception_records_failure_and_reraises(self):
        cb = make_cb(failure_threshold=99)
        with pytest.raises(RuntimeError, match="boom"):
            async with cb.guard():
                raise RuntimeError("boom")
        assert cb.get_stats()["total_failures"] == 1

    @pytest.mark.asyncio
    async def test_open_breaker_raises_circuit_open(self):
        cb = make_cb(failure_threshold=1, recovery_timeout_s=999)
        trip(cb, 1)
        with pytest.raises(CircuitOpen):
            async with cb.guard():
                pass  # should never reach here

    @pytest.mark.asyncio
    async def test_guard_trips_breaker_after_threshold(self):
        cb = make_cb(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ValueError):
                async with cb.guard():
                    raise ValueError("oops")
        assert cb.is_open

    @pytest.mark.asyncio
    async def test_guard_closes_after_recovery(self):
        cb = make_cb(
            failure_threshold=1,
            recovery_timeout_s=0.0,
            success_threshold=1,
        )
        # Trip it
        async with cb._lock:
            cb.record_failure("initial")
        assert cb.is_open

        # Force elapsed past window
        cb._opened_at = time.monotonic() - 1.0

        # First guard call: transitions to HALF_OPEN and succeeds → CLOSED
        async with cb.guard():
            pass
        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_circuit_open_not_recorded_as_failure(self):
        """CircuitOpen rejection should not count as a new failure."""
        cb = make_cb(failure_threshold=1, recovery_timeout_s=999)
        trip(cb, 1)
        initial_failures = cb.get_stats()["total_failures"]
        try:
            async with cb.guard():
                pass
        except CircuitOpen:
            pass
        assert cb.get_stats()["total_failures"] == initial_failures


# ═══════════════════════════════════════════════════════════════════
#  Stats
# ═══════════════════════════════════════════════════════════════════

class TestStats:
    def test_stats_shape(self):
        cb = make_cb()
        stats = cb.get_stats()
        assert stats["service"] == "TEST"
        assert stats["state"] == "closed"
        assert "consecutive_failures" in stats
        assert "total_failures" in stats
        assert "total_successes" in stats
        assert "total_rejected" in stats
        assert "config" in stats

    def test_retry_after_s_none_when_closed(self):
        cb = make_cb()
        assert cb.get_stats()["retry_after_s"] is None

    def test_retry_after_s_positive_when_open(self):
        cb = make_cb(failure_threshold=1, recovery_timeout_s=60.0)
        trip(cb, 1)
        stats = cb.get_stats()
        assert stats["retry_after_s"] is not None
        assert stats["retry_after_s"] > 0

    def test_last_failure_reason_captured(self):
        cb = make_cb(failure_threshold=1)
        cb.record_failure("connection refused")
        assert cb.get_stats()["last_failure_reason"] == "connection refused"


# ═══════════════════════════════════════════════════════════════════
#  Manual reset
# ═══════════════════════════════════════════════════════════════════

class TestReset:
    def test_reset_closes_open_breaker(self):
        cb = make_cb(failure_threshold=1)
        trip(cb, 1)
        assert cb.is_open
        cb.reset()
        assert cb.is_closed

    def test_reset_clears_failure_count(self):
        cb = make_cb(failure_threshold=5)
        trip(cb, 4)
        cb.reset()
        trip(cb, 4)
        assert cb.is_closed  # failure count was reset


# ═══════════════════════════════════════════════════════════════════
#  on_state_change callback
# ═══════════════════════════════════════════════════════════════════

class TestCallback:
    def test_callback_receives_transition_args(self):
        calls = []

        async def cb_fn(service, old, new):
            calls.append((service, old, new))

        cb = CircuitBreaker("SVC", fast_cfg(failure_threshold=1), on_state_change=cb_fn)

        async def run():
            trip(cb, 1)
            await asyncio.sleep(0)  # allow create_task to execute

        asyncio.run(run())
        assert len(calls) == 1
        assert calls[0] == ("SVC", CircuitState.CLOSED, CircuitState.OPEN)


# ═══════════════════════════════════════════════════════════════════
#  ServiceBreakers registry
# ═══════════════════════════════════════════════════════════════════

class TestServiceBreakers:
    def test_all_four_services_exist(self):
        sb = ServiceBreakers()
        assert sb.openrouter.service == "OPENROUTER"
        assert sb.solana_rpc.service == "SOLANA_RPC"
        assert sb.jupiter.service == "JUPITER"
        assert sb.telegram.service == "TELEGRAM"

    def test_all_start_closed(self):
        sb = ServiceBreakers()
        for b in [sb.openrouter, sb.solana_rpc, sb.jupiter, sb.telegram]:
            assert b.is_closed

    def test_get_all_stats_shape(self):
        sb = ServiceBreakers()
        stats = sb.get_all_stats()
        assert stats["healthy"] is True
        assert set(stats["services"]) == {"OPENROUTER", "SOLANA_RPC", "JUPITER", "TELEGRAM"}

    def test_healthy_false_when_any_open(self):
        sb = ServiceBreakers()
        trip(sb.openrouter, sb.openrouter.config.failure_threshold)
        stats = sb.get_all_stats()
        assert stats["healthy"] is False
        assert stats["services"]["OPENROUTER"]["state"] == "open"
        assert stats["services"]["SOLANA_RPC"]["state"] == "closed"

    def test_reset_all_closes_everything(self):
        sb = ServiceBreakers()
        trip(sb.openrouter, sb.openrouter.config.failure_threshold)
        trip(sb.jupiter, sb.jupiter.config.failure_threshold)
        sb.reset_all()
        assert sb.get_all_stats()["healthy"] is True

    def test_services_are_independent(self):
        sb = ServiceBreakers()
        trip(sb.openrouter, sb.openrouter.config.failure_threshold)
        # Other services unaffected
        sb.solana_rpc.check()
        sb.jupiter.check()
        sb.telegram.check()

    def test_different_thresholds(self):
        sb = ServiceBreakers()
        # Telegram has failure_threshold=5, Openrouter has 3
        assert sb.telegram.config.failure_threshold > sb.openrouter.config.failure_threshold

    def test_shared_callback(self):
        calls = []

        async def cb_fn(service, old, new):
            calls.append(service)

        sb = ServiceBreakers(on_state_change=cb_fn)

        async def run():
            trip(sb.openrouter, sb.openrouter.config.failure_threshold)
            trip(sb.jupiter, sb.jupiter.config.failure_threshold)
            await asyncio.sleep(0)

        asyncio.run(run())
        assert "OPENROUTER" in calls
        assert "JUPITER" in calls
