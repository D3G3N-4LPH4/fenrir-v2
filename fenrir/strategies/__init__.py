"""
FENRIR Trading Strategies — pluggable, self-contained strategy units.

Each strategy defines its own entry criteria, AI context, trade parameters,
budget limits, and position constraints. Strategies can be activated,
paused, and run concurrently.
"""

from fenrir.strategies.base import StrategyState, TradeParams, TradingStrategy
from fenrir.strategies.graduation import GraduationStrategy
from fenrir.strategies.sniper import (
    ConservativeSniperStrategy,
    DegenSniperStrategy,
    SniperStrategy,
)

# Strategy registry: strategy_id -> class
STRATEGY_REGISTRY: dict[str, type[TradingStrategy]] = {
    "sniper": SniperStrategy,
    "sniper_conservative": ConservativeSniperStrategy,
    "sniper_degen": DegenSniperStrategy,
    "graduation": GraduationStrategy,
}


def get_strategy_class(strategy_id: str) -> type[TradingStrategy] | None:
    """Look up a strategy class by ID."""
    return STRATEGY_REGISTRY.get(strategy_id)


def list_strategies() -> list[str]:
    """List all available strategy IDs."""
    return list(STRATEGY_REGISTRY.keys())


__all__ = [
    "TradingStrategy",
    "TradeParams",
    "StrategyState",
    "SniperStrategy",
    "ConservativeSniperStrategy",
    "DegenSniperStrategy",
    "GraduationStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy_class",
    "list_strategies",
]
