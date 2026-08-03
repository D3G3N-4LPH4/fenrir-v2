#!/usr/bin/env python3
"""
FENRIR - Arbitrage Scanner Service (Phase 4.4c-evidence, read-only)

Periodically checks a set of real tokens for cost-cleared cross-pool price
divergence and surfaces what it finds — the evidence-gathering step of the
arbitrage live path. It answers the empirical question "do actionable cross-venue
edges actually exist on this market?" before any executor is built.

Read-only by construction: it fetches DexScreener pools, runs the divergence detector
via the monitor (which emits an event only on an actionable gap), and records simple
stats. It never signs a transaction. Off by default; opt-in via config.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fenrir.trading.arbitrage_monitor import ArbitrageMonitor, dexscreener_venue_quotes

# Async source of token addresses to check this cycle (e.g. open positions + recent
# detections). Returning an empty list is fine — the cycle simply idles.
TokenSource = Callable[[], Awaitable[list[str]]]
PairFetcher = Callable[[str], Awaitable[list[dict]]]


class ArbitrageScanner:
    """Periodic read-only cross-pool divergence scanner over a live token set."""

    def __init__(
        self,
        monitor: ArbitrageMonitor,
        fetch_pairs: PairFetcher,
        token_source: TokenSource,
        interval_seconds: float = 30.0,
        max_tokens_per_cycle: int = 25,
        logger: Any = None,
    ) -> None:
        self.monitor = monitor
        self._fetch_pairs = fetch_pairs
        self._token_source = token_source
        self.interval_seconds = interval_seconds
        self.max_tokens_per_cycle = max_tokens_per_cycle
        self._logger = logger
        self._running = False
        self.cycles = 0
        self.tokens_checked = 0

    async def scan_once(self) -> int:
        """Run one cycle: check each token for cross-pool divergence. Returns the number
        of actionable opportunities surfaced this cycle."""
        self.cycles += 1
        try:
            tokens = await self._token_source()
        except Exception as e:  # noqa: BLE001 - a bad source must not kill the loop
            self._log("warning", f"arb token source error: {e}")
            return 0

        found = 0
        for token in tokens[: self.max_tokens_per_cycle]:
            try:
                quotes = await dexscreener_venue_quotes(self._fetch_pairs, token)
            except Exception as e:  # noqa: BLE001 - one token's fetch must not kill the loop
                self._log("warning", f"arb fetch error for {token[:8]}...: {e}")
                continue
            self.tokens_checked += 1
            if len(quotes) < 2:
                continue  # a single pool cannot diverge from itself
            opp = await self.monitor.evaluate_quotes(token, quotes)
            if opp is not None:
                found += 1
        return found

    async def start_scanning(self) -> None:
        """Run scan cycles forever at the configured interval (until cancelled)."""
        self._running = True
        self._log(
            "info", f"Arbitrage monitor scanning every {self.interval_seconds:.0f}s (read-only)"
        )
        while self._running:
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep the loop alive across surprises
                self._log("warning", f"arb scan cycle error: {e}")
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        self._running = False

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        method = getattr(self._logger, level, None)
        try:
            if callable(method):
                method(msg)
                return
        except TypeError:
            pass
        fallback = getattr(self._logger, "warning", None) or getattr(self._logger, "info", None)
        if callable(fallback):
            fallback(msg)
