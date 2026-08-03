#!/usr/bin/env python3
"""
FENRIR - Backtest models (Phase 6, backtesting rigor)

The dataclasses the backtester consumes and produces. A ``BacktestSample`` is one
candidate token: the entry-time MarketData snapshot the strategy evaluates, plus the
forward SOL/token price path used to simulate the exit. Trades and metrics come back
in ``BacktestTrade`` / ``BacktestMetrics`` / ``BacktestResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestSample:
    """One historical candidate: an entry snapshot + the price path that follows.

    ``forward_prices`` is SOL-per-token, oldest first, with ``[0]`` the entry price.
    ``frame_seconds`` is the spacing between prices, used to honor the strategy's
    max-hold in real time.
    """

    token_address: str
    token_data: dict[str, Any]
    market_data: Any  # a MarketData snapshot at entry time
    forward_prices: list[float]
    frame_seconds: float = 60.0
    symbol: str = ""


@dataclass
class BacktestTrade:
    """A single simulated round trip, exited by the strategy's own trade params."""

    token_address: str
    strategy_id: str
    entry_price: float
    exit_price: float
    exit_reason: str  # take_profit | stop_loss | trailing_stop | max_hold | end_of_data
    hold_frames: int
    pnl_pct: float
    strength: float  # normalized signal strength at entry (unified Signal)
    symbol: str = ""


@dataclass
class BacktestMetrics:
    """Rigorous summary statistics over a set of trades."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy_pct: float = 0.0  # mean PnL per trade
    total_return_pct: float = 0.0  # sum of trade PnL% (independent sizing)
    profit_factor: float = 0.0  # gross profit / gross loss (0 = undefined/no losses)
    sharpe: float = 0.0  # per-trade Sharpe (mean / stdev of returns)
    max_drawdown_pct: float = 0.0  # worst peak-to-trough of the cumulative curve

    def to_dict(self) -> dict:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "expectancy_pct": round(self.expectancy_pct, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
        }


@dataclass
class BacktestResult:
    """A strategy's backtest outcome: every trade + the summary metrics."""

    strategy_id: str
    trades: list[BacktestTrade] = field(default_factory=list)
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    samples_evaluated: int = 0
    samples_entered: int = 0

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "samples_evaluated": self.samples_evaluated,
            "samples_entered": self.samples_entered,
            "metrics": self.metrics.to_dict(),
        }
