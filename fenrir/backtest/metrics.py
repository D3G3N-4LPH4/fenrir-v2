#!/usr/bin/env python3
"""
FENRIR - Backtest metrics (Phase 6)

Rigorous summary statistics over a set of trade PnL%s: win rate, average win/loss,
expectancy, profit factor, per-trade Sharpe, and max drawdown of the cumulative curve.
Pure functions — no state, no network.
"""

from __future__ import annotations

import statistics

from fenrir.backtest.models import BacktestMetrics, BacktestTrade


def max_drawdown_pct(pnls: list[float]) -> float:
    """Worst peak-to-trough drawdown of the cumulative (summed) PnL curve, in the same
    percentage units as the inputs. 0 when the curve never draws down."""
    cumulative = 0.0
    peak = 0.0
    worst = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        worst = max(worst, peak - cumulative)
    return worst


def compute_metrics(trades: list[BacktestTrade]) -> BacktestMetrics:
    """Summarize a set of trades. Empty input yields an all-zero metrics block."""
    n = len(trades)
    if n == 0:
        return BacktestMetrics()

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss = -sum(losses)  # positive magnitude
    # Undefined when there are no losses; 0.0 signals "no denominator" to callers.
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    expectancy = statistics.fmean(pnls)
    # Per-trade Sharpe: mean return over its dispersion. 0 when <2 trades or no spread.
    sharpe = 0.0
    if n >= 2:
        stdev = statistics.stdev(pnls)
        if stdev > 0:
            sharpe = expectancy / stdev

    return BacktestMetrics(
        trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / n,
        avg_win_pct=statistics.fmean(wins) if wins else 0.0,
        avg_loss_pct=statistics.fmean(losses) if losses else 0.0,
        expectancy_pct=expectancy,
        total_return_pct=sum(pnls),
        profit_factor=profit_factor,
        sharpe=sharpe,
        max_drawdown_pct=max_drawdown_pct(pnls),
    )
