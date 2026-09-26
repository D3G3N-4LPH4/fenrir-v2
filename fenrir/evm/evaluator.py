#!/usr/bin/env python3
"""
FENRIR - EVM token evaluator (on-chain EVM, read-only)

Runs an EVM token through the FULL decision machinery — the signal strategies
(momentum, mean_reversion, …), the unified ``Signal`` normalization + confluence, and
(optionally) the AI brain — reusing the exact same code the Solana path uses. It only
READS (a DexScreener snapshot) and evaluates; it never signs or sends a transaction.
On-chain EVM execution is a later, gated PR.

The snapshot fetch and the AI brain are injected, so the evaluator is fully testable
with no network. In production the fetch is ``DexScreenerProvider.fetch_snapshot`` and
the brain is the bot's ``ClaudeBrain``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fenrir.evm.adapters import snapshot_to_market_data, snapshot_to_token_data
from fenrir.filters import MarketData
from fenrir.signals import Signal, SignalDirection, normalize_signal

# Injected read-only snapshot source: (token_address, chain) -> TokenSnapshot | None.
SnapshotFetcher = Callable[[str, Any], Awaitable[Any]]


@dataclass
class EvmEvaluation:
    """The read-only result of running one EVM token through the machinery."""

    chain: str
    token_address: str
    symbol: str
    market_data: MarketData
    signals: list[Signal] = field(default_factory=list)
    ai_decision: dict[str, Any] | None = None  # populated only when a brain is provided
    confluence: Any = None  # ConfluenceResult | None (only with an aggregator)

    @property
    def strategy_ids(self) -> list[str]:
        return [s.source for s in self.signals]

    def to_dict(self) -> dict:
        return {
            "chain": self.chain,
            "token_address": self.token_address,
            "symbol": self.symbol,
            "signals": [s.to_dict() for s in self.signals],
            "ai_decision": self.ai_decision,
            "confluent": self.confluence is not None,
        }


class EvmTokenEvaluator:
    """Evaluate EVM tokens with the existing strategies / signals / brain — read-only."""

    def __init__(
        self,
        strategies: list[Any],
        fetch_snapshot: SnapshotFetcher,
        aggregator: Any = None,
        brain: Any = None,
        logger: Any = None,
    ) -> None:
        # Only market-data-aware (signal) strategies apply to a snapshot-based read.
        self.strategies = [s for s in strategies if getattr(s, "uses_market_data", False)]
        self._fetch = fetch_snapshot
        self._aggregator = aggregator
        self._brain = brain
        self._logger = logger

    async def evaluate(self, token_address: str, chain: Any = None) -> EvmEvaluation | None:
        """Fetch the token's snapshot and run it through the strategies (+ optional AI
        brain). Returns None when the token has no usable snapshot. Read-only.

        ``chain`` filters the fetch to a chain; ``None`` lets the provider pick the
        token's most-liquid chain. The result's chain always comes from the fetched
        snapshot (authoritative), not the requested one."""
        try:
            snapshot = await self._fetch(token_address, chain)
        except Exception as e:  # noqa: BLE001 - a provider hiccup must not raise to the caller
            self._log("warning", f"EVM snapshot fetch failed for {token_address[:10]}...: {e}")
            return None
        if snapshot is None:
            return None

        market_data = snapshot_to_market_data(snapshot)
        token_data = snapshot_to_token_data(snapshot)

        signals: list[Signal] = []
        for strategy in self.strategies:
            try:
                bespoke = strategy.evaluate_token(token_data, market_data)
            except Exception as e:  # noqa: BLE001 - one strategy's error must not sink the rest
                self._log("warning", f"EVM strategy {getattr(strategy, 'strategy_id', '?')}: {e}")
                continue
            if bespoke is None:
                continue
            signal = normalize_signal(bespoke)
            signals.append(signal)
            if self._aggregator is not None:
                self._aggregator.add(signal)

        confluence = None
        if self._aggregator is not None and signals:
            confluence = self._aggregator.confluence_for(token_address, SignalDirection.LONG)

        ai_decision = await self._run_brain(token_data) if self._brain is not None else None

        chain_value = getattr(snapshot.chain, "value", str(snapshot.chain))
        return EvmEvaluation(
            chain=chain_value,
            token_address=token_address,
            symbol=snapshot.symbol,
            market_data=market_data,
            signals=signals,
            ai_decision=ai_decision,
            confluence=confluence,
        )

    async def _run_brain(self, token_data: dict) -> dict[str, Any] | None:
        """Run the AI brain's entry evaluation (read-only — the decision is returned,
        never executed). Empty positions: this is a fresh cross-chain read."""
        try:
            should_buy, analysis, amount_override = await self._brain.evaluate_entry(token_data, {})
        except Exception as e:  # noqa: BLE001 - the AI is best-effort here
            self._log("warning", f"EVM brain eval failed: {e}")
            return None
        result: dict[str, Any] = {"should_buy": bool(should_buy)}
        if analysis is not None:
            result.update(
                decision=getattr(getattr(analysis, "decision", None), "value", None),
                confidence=getattr(analysis, "confidence", None),
                risk_score=getattr(analysis, "risk_score", None),
            )
        if amount_override is not None:
            result["amount_override"] = amount_override
        return result

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
