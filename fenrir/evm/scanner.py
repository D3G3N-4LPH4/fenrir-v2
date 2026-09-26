#!/usr/bin/env python3
"""
FENRIR - EVM evaluator scanner (Phase 7.1, on-chain EVM, read-only)

Periodically runs a set of EVM tokens through the read-only EvmTokenEvaluator (the full
strategy / signal / brain machinery) and surfaces what fires as EVM_SIGNAL events. This
extends the empirical loop to EVM: it observes and reports, but never signs or sends a
transaction — on-chain EVM execution is a later, gated PR.

Token source and evaluator are injected, so the scanner is fully testable with no
network. In production the source is the operator watchlist (later: discovery hits) and
the evaluator wraps DexScreenerProvider.fetch_snapshot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fenrir.events.types import evm_signal_event

TokenSource = Callable[[], Awaitable[list[str]]]


class EvmEvaluatorScanner:
    """Read-only periodic EVM evaluation loop over a token source."""

    def __init__(
        self,
        evaluator: Any,
        token_source: TokenSource,
        enabled_chains: set[str] | None = None,
        interval_seconds: float = 60.0,
        max_tokens_per_cycle: int = 25,
        event_bus: Any = None,
        collector: Any = None,
        max_concurrent_collections: int = 20,
        logger: Any = None,
    ) -> None:
        self.evaluator = evaluator
        self._token_source = token_source
        # None → accept whatever chain the snapshot resolves to; else filter to these.
        self.enabled_chains = enabled_chains
        self.interval_seconds = interval_seconds
        self.max_tokens_per_cycle = max_tokens_per_cycle
        self._bus = event_bus
        # Optional read-only forward-price collector: a flagged EVM token's forward path
        # is recorded to JSONL for the backtester (never trades). See fenrir.backtest.
        self._collector = collector
        self._max_collections = max_concurrent_collections
        self._collect_tasks: set[asyncio.Task] = set()
        self._collecting: set[str] = set()
        self._logger = logger
        self._running = False
        self.cycles = 0
        self.tokens_evaluated = 0
        self.signals_surfaced = 0

    async def scan_once(self) -> int:
        """Evaluate this cycle's tokens; emit EVM_SIGNAL for any that a strategy flags on
        an enabled chain. Returns the number of tokens that produced ≥1 signal."""
        self.cycles += 1
        try:
            tokens = await self._token_source()
        except Exception as e:  # noqa: BLE001 - a bad source must not kill the loop
            self._log("warning", f"EVM token source error: {e}")
            return 0

        surfaced = 0
        for token in tokens[: self.max_tokens_per_cycle]:
            try:
                result = await self.evaluator.evaluate(token)
            except Exception as e:  # noqa: BLE001 - one token's error must not kill the loop
                self._log("warning", f"EVM evaluate error for {token[:10]}...: {e}")
                continue
            self.tokens_evaluated += 1
            if result is None or not result.signals:
                continue
            if self.enabled_chains is not None and result.chain not in self.enabled_chains:
                continue  # resolved to a chain we're not watching (e.g. Solana)

            surfaced += 1
            self.signals_surfaced += 1
            await self._emit(result)
            self._maybe_collect(result)
        return surfaced

    def _maybe_collect(self, result: Any) -> None:
        """Spawn a bounded, deduped background forward-price collection for a flagged
        EVM token (read-only — samples for the backtester, no trade)."""
        if self._collector is None:
            return
        token = result.token_address
        if token in self._collecting or len(self._collect_tasks) >= self._max_collections:
            return
        self._collecting.add(token)
        task = asyncio.create_task(self._run_collect(result))
        self._collect_tasks.add(task)
        task.add_done_callback(self._collect_tasks.discard)

    async def _run_collect(self, result: Any) -> None:
        try:
            await self._collector.collect(result.token_address, result.market_data, result.symbol)
        except Exception as e:  # noqa: BLE001 - collection is best-effort, never fatal
            self._log(
                "warning", f"EVM sample collection failed for {result.token_address[:10]}: {e}"
            )
        finally:
            self._collecting.discard(result.token_address)

    async def _emit(self, result: Any) -> None:
        confluent = result.confluence is not None and result.confluence.is_confluent()
        strength = (
            result.confluence.combined_strength
            if result.confluence is not None
            else (max((s.strength for s in result.signals), default=0.0))
        )
        self._log(
            "info",
            f"EVM[{result.chain}] {result.token_address[:10]}... "
            f"{result.strategy_ids} conviction={strength:.2f}"
            f"{' CONFLUENT' if confluent else ''}",
        )
        if self._bus is not None:
            await self._bus.emit(
                evm_signal_event(
                    token_address=result.token_address,
                    chain=result.chain,
                    symbol=result.symbol,
                    sources=result.strategy_ids,
                    combined_strength=strength,
                    confluent=confluent,
                )
            )

    async def start_scanning(self) -> None:
        """Run scan cycles forever at the configured interval (until cancelled)."""
        self._running = True
        self._log("info", f"EVM evaluator scanning every {self.interval_seconds:.0f}s (read-only)")
        while self._running:
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep the loop alive across surprises
                self._log("warning", f"EVM scan cycle error: {e}")
            await asyncio.sleep(self.interval_seconds)

    async def drain_collections(self, timeout: float | None = None) -> None:  # noqa: ASYNC109
        """Wait for in-flight forward-price collection tasks to finish (bounded by
        ``timeout``). Used at the end of a bounded collection run so samples flagged
        near the end are not lost. Does not cancel — that's what stop() is for."""
        pending = [t for t in self._collect_tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)

    async def stop(self) -> None:
        self._running = False
        for task in list(self._collect_tasks):
            task.cancel()
        self._collect_tasks.clear()
        self._collecting.clear()

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
