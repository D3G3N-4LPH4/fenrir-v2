"""
FENRIR Signals — a unified conviction abstraction over heterogeneous strategies.

``Signal`` normalizes every strategy's bespoke result and the market-neutral detectors
onto one comparable shape (source, direction, 0-1 strength). Adapters map the existing
objects on; nothing here changes the strategies themselves.
"""

from fenrir.signals.adapters import (
    normalize_arbitrage,
    normalize_signal,
    normalize_strategy_signal,
)
from fenrir.signals.aggregator import ConfluenceResult, SignalAggregator
from fenrir.signals.models import Signal, SignalDirection

__all__ = [
    "Signal",
    "SignalDirection",
    "normalize_signal",
    "normalize_strategy_signal",
    "normalize_arbitrage",
    "SignalAggregator",
    "ConfluenceResult",
]
