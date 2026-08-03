"""
FENRIR Backtest — rigorous, drift-free strategy evaluation.

Replays historical candidates through the same strategy ``evaluate_token`` + unified
``Signal`` + ``TradeParams`` exit logic that runs live, so backtests measure exactly
what will trade. See ``SignalBacktester``.
"""

from fenrir.backtest.engine import SignalBacktester
from fenrir.backtest.loader import load_samples, sample_from_dict, samples_from_dicts
from fenrir.backtest.metrics import compute_metrics, max_drawdown_pct
from fenrir.backtest.models import (
    BacktestMetrics,
    BacktestResult,
    BacktestSample,
    BacktestTrade,
)
from fenrir.backtest.portfolio import PortfolioBacktester, PortfolioResult

__all__ = [
    "SignalBacktester",
    "PortfolioBacktester",
    "PortfolioResult",
    "BacktestSample",
    "BacktestTrade",
    "BacktestMetrics",
    "BacktestResult",
    "compute_metrics",
    "max_drawdown_pct",
    "load_samples",
    "sample_from_dict",
    "samples_from_dicts",
]
