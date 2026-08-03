#!/usr/bin/env python3
"""
FENRIR - Portfolio / multi-strategy backtest (Phase 6.2)

Runs several strategies over the SAME history and measures them together — including the
question Phase 5's confluence machinery raised but could not yet answer: do setups where
multiple independent strategies agree actually perform better than lone signals?

For each token, the strategies that entered are recorded; a token is "confluent" when at
least ``confluence_min_sources`` distinct strategies entered it. The result splits the
combined trade metrics into confluent vs. non-confluent so the edge (if any) is visible,
not assumed. Pure — reuses the drift-free SignalBacktester per strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fenrir.backtest.engine import SignalBacktester
from fenrir.backtest.metrics import compute_metrics
from fenrir.backtest.models import BacktestMetrics, BacktestResult, BacktestSample, BacktestTrade


@dataclass
class PortfolioResult:
    """Per-strategy results plus combined and confluence-split metrics."""

    per_strategy: dict[str, BacktestResult] = field(default_factory=dict)
    combined_metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    confluent_metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    non_confluent_metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    confluent_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "per_strategy": {sid: r.to_dict() for sid, r in self.per_strategy.items()},
            "combined": self.combined_metrics.to_dict(),
            "confluent": self.confluent_metrics.to_dict(),
            "non_confluent": self.non_confluent_metrics.to_dict(),
            "confluent_token_count": len(self.confluent_tokens),
        }


class PortfolioBacktester:
    """Backtest a set of strategies over shared samples, with confluence measurement."""

    def __init__(self, backtester: SignalBacktester | None = None) -> None:
        self._bt = backtester or SignalBacktester()

    def run(
        self,
        strategies: list[Any],
        samples: list[BacktestSample],
        confluence_min_sources: int = 2,
    ) -> PortfolioResult:
        per_strategy: dict[str, BacktestResult] = {}
        all_trades: list[BacktestTrade] = []
        strategies_by_token: dict[str, set[str]] = {}

        for strategy in strategies:
            result = self._bt.run(strategy, samples)
            per_strategy[result.strategy_id] = result
            for trade in result.trades:
                all_trades.append(trade)
                strategies_by_token.setdefault(trade.token_address, set()).add(trade.strategy_id)

        confluent_tokens = {
            token
            for token, sources in strategies_by_token.items()
            if len(sources) >= confluence_min_sources
        }
        confluent_trades = [t for t in all_trades if t.token_address in confluent_tokens]
        non_confluent_trades = [t for t in all_trades if t.token_address not in confluent_tokens]

        return PortfolioResult(
            per_strategy=per_strategy,
            combined_metrics=compute_metrics(all_trades),
            confluent_metrics=compute_metrics(confluent_trades),
            non_confluent_metrics=compute_metrics(non_confluent_trades),
            confluent_tokens=sorted(confluent_tokens),
        )
