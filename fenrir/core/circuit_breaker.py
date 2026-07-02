#!/usr/bin/env python3
"""
FENRIR - Per-Service Circuit Breaker

Adapted from deploybot's single-job kill switch into a full three-state
machine: CLOSED → OPEN → HALF_OPEN → CLOSED.

One breaker per external dependency:
  - OPENROUTER  (AI / LLM)
  - SOLANA_RPC  (on-chain reads/writes)
  - JUPITER     (price feeds / swaps)
  - TELEGRAM    (alerts / commands)

A service that starts returning repeated errors trips its own breaker
without affecting any other service. After a cooldown the breaker
enters HALF_OPEN, lets one probe through, and re-closes on success.

Usage:
    from fenrir.core.circuit_breaker import ServiceBreakers

    breakers = ServiceBreakers()

    async with breakers.openrouter.guard():
        resp = await session.post(OPENROUTER_URL, ...)

    # Or manual:
    breakers.solana_rpc.check()          # raises CircuitOpen if tripped
    breakers.solana_rpc.record_success()
    breakers.solana_rpc.record_failure("timeout")

    # Health snapshot for dashboard / API:
    stats = breakers.get_all_stats()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Exceptions ───────────────────────────────────────────────────────────────


class CircuitOpen(Exception):
    """Raised when a call is attempted while the breaker is OPEN."""

    def __init__(self, service: str, reason: str, retry_after_s: float) -> None:
        self.service = service
        self.reason = reason
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Circuit OPEN for {service!r}: {reason} " f"(retry in {retry_after_s:.1f}s)"
        )


# ─── State ────────────────────────────────────────────────────────────────────


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking all calls
    HALF_OPEN = "half_open"  # Probing for recovery


# ─── Config ───────────────────────────────────────────────────────────────────


@dataclass
class CircuitBreakerConfig:
    """
    Thresholds for a single circuit breaker instance.

    failure_threshold   — consecutive failures before OPEN
    recovery_timeout_s  — seconds in OPEN before trying HALF_OPEN
    success_threshold   — consecutive successes in HALF_OPEN before CLOSED
    half_open_max_calls — max concurrent probes allowed in HALF_OPEN
    """

    failure_threshold: int = 3
    recovery_timeout_s: float = 60.0
    success_threshold: int = 2
    half_open_max_calls: int = 1


# Per-service defaults tuned for FENRIR's traffic patterns
_OPENROUTER_CFG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout_s=60.0,
    success_threshold=2,
    half_open_max_calls=1,
)
_SOLANA_RPC_CFG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout_s=30.0,
    success_threshold=3,
    half_open_max_calls=2,
)
_JUPITER_CFG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout_s=45.0,
    success_threshold=2,
    half_open_max_calls=1,
)
_TELEGRAM_CFG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout_s=120.0,
    success_threshold=1,
    half_open_max_calls=1,
)


# ─── Core circuit breaker ─────────────────────────────────────────────────────


class CircuitBreaker:
    """
    Three-state circuit breaker for a single external service.

    Thread-safe for asyncio: all state mutations hold _lock.
    """

    def __init__(
        self,
        service: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], Awaitable[None]] | None = None,
    ) -> None:
        self.service = service
        self.config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_calls = 0  # probes currently in-flight
        self._opened_at: float | None = None
        self._last_failure_reason: str = ""
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejected = 0
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────

    def check(self) -> None:
        """
        Synchronously assert the breaker is callable.
        Raises CircuitOpen if OPEN and the recovery timeout has not elapsed.
        Raises CircuitOpen if OPEN and HALF_OPEN is already at capacity.
        No-op when CLOSED or when transitioning to HALF_OPEN.
        """
        state = self._state
        if state == CircuitState.CLOSED:
            return

        if state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0.0)
            remaining = self.config.recovery_timeout_s - elapsed
            if remaining > 0:
                self._total_rejected += 1
                raise CircuitOpen(self.service, self._last_failure_reason, remaining)
            # Recovery window has elapsed — transition to HALF_OPEN.
            # _transition() needs the lock; callers that need atomicity use guard().
            # Here we just update state so the probe can proceed.
            self._state = CircuitState.HALF_OPEN
            self._consecutive_successes = 0
            self._half_open_calls = 0
            logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN (probing)", self.service)
            return

        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._total_rejected += 1
                raise CircuitOpen(
                    self.service,
                    f"HALF_OPEN at capacity ({self._half_open_calls} probe(s) in flight)",
                    0.0,
                )
            # Track the in-flight probe so max-calls limit is enforced
            # whether callers use guard() or the manual check/record pattern.
            self._half_open_calls += 1

    def record_success(self) -> None:
        """Record a successful call. May close the breaker from HALF_OPEN."""
        self._total_successes += 1
        self._consecutive_failures = 0

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls = max(0, self._half_open_calls - 1)
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.config.success_threshold:
                self._transition(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            self._consecutive_successes += 1

    def record_failure(self, reason: str = "unknown") -> None:
        """Record a failed call. May open or keep open the breaker."""
        self._total_failures += 1
        self._consecutive_successes = 0
        self._last_failure_reason = reason

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls = max(0, self._half_open_calls - 1)
            # Any failure in HALF_OPEN trips back to OPEN
            self._transition(CircuitState.OPEN)
            return

        if self._state == CircuitState.CLOSED:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.failure_threshold:
                self._transition(CircuitState.OPEN)

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        """
        Async context manager that wraps one external call.

        Calls check() before yielding and records success/failure based
        on whether the block raises. Exceptions propagate normally after
        the failure is recorded.

        Usage:
            async with breaker.guard():
                result = await some_external_call()
        """
        async with self._lock:
            self.check()  # check() now increments _half_open_calls when HALF_OPEN

        try:
            yield
        except Exception as exc:
            async with self._lock:
                self.record_failure(str(exc)[:120])
            raise
        else:
            async with self._lock:
                self.record_success()

    # ── State inspection ──────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    def get_stats(self) -> dict:
        """Snapshot for health dashboard / logs."""
        elapsed_open = (
            time.monotonic() - self._opened_at
            if self._opened_at and self._state == CircuitState.OPEN
            else None
        )
        remaining = (
            max(0.0, self.config.recovery_timeout_s - elapsed_open)
            if elapsed_open is not None
            else None
        )
        return {
            "service": self.service,
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "total_rejected": self._total_rejected,
            "last_failure_reason": self._last_failure_reason or None,
            "retry_after_s": round(remaining, 1) if remaining is not None else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout_s": self.config.recovery_timeout_s,
                "success_threshold": self.config.success_threshold,
            },
        }

    def reset(self) -> None:
        """Manually close the breaker (for admin/testing use)."""
        prev = self._state
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_calls = 0
        self._opened_at = None
        if prev != CircuitState.CLOSED:
            logger.info("CircuitBreaker[%s]: manually reset → CLOSED", self.service)

    # ── Internal ──────────────────────────────────────────────────

    def _transition(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            logger.warning(
                "CircuitBreaker[%s]: %s → OPEN after %d failure(s): %s",
                self.service,
                old_state.value.upper(),
                self.config.failure_threshold,
                self._last_failure_reason,
            )
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._half_open_calls = 0
            logger.info(
                "CircuitBreaker[%s]: HALF_OPEN → CLOSED (service recovered)",
                self.service,
            )

        if self._on_state_change is not None:
            # Schedule callback without blocking the caller.
            # Errors in the callback are swallowed so they never affect trading.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._safe_callback(old_state, new_state))
            except RuntimeError:
                pass

    async def _safe_callback(
        self,
        old_state: CircuitState,
        new_state: CircuitState,
    ) -> None:
        try:
            await self._on_state_change(self.service, old_state, new_state)  # type: ignore[misc]
        except Exception as exc:
            logger.debug(
                "CircuitBreaker[%s]: on_state_change callback failed: %s",
                self.service,
                exc,
            )


# ─── Service registry ─────────────────────────────────────────────────────────


class ServiceBreakers:
    """
    Pre-configured circuit breakers for every external service FENRIR calls.

    Instantiate once at bot startup and pass through the DI chain, or
    use as a module-level singleton.

    Attributes:
        openrouter  — OpenRouter AI / LLM calls
        solana_rpc  — Solana RPC node (reads + transaction sends)
        jupiter     — Jupiter price feeds / swap quotes
        telegram    — Telegram Bot API (alerts + polling)
    """

    def __init__(
        self,
        on_state_change: Callable[[str, CircuitState, CircuitState], Awaitable[None]] | None = None,
    ) -> None:
        self.openrouter = CircuitBreaker("OPENROUTER", _OPENROUTER_CFG, on_state_change)
        self.solana_rpc = CircuitBreaker("SOLANA_RPC", _SOLANA_RPC_CFG, on_state_change)
        self.jupiter = CircuitBreaker("JUPITER", _JUPITER_CFG, on_state_change)
        self.telegram = CircuitBreaker("TELEGRAM", _TELEGRAM_CFG, on_state_change)

    def get_all_stats(self) -> dict:
        """Aggregate snapshot of all four breakers."""
        breakers = [self.openrouter, self.solana_rpc, self.jupiter, self.telegram]
        any_open = any(b.is_open for b in breakers)
        return {
            "healthy": not any_open,
            "services": {b.service: b.get_stats() for b in breakers},
        }

    def reset_all(self) -> None:
        """Force all breakers closed (admin use only)."""
        for b in [self.openrouter, self.solana_rpc, self.jupiter, self.telegram]:
            b.reset()
