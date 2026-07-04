"""
FENRIR Trading Strategies — pluggable, self-contained strategy units.

Each strategy defines its own entry criteria, AI context, trade parameters,
budget limits, and position constraints. Strategies can be activated,
paused, and run concurrently.

The signal-oriented strategies (migration_snipe, reversal, volume_anomaly,
narrative_tracker) conform to the same ``TradingStrategy`` ABC but also expose
a richer ``evaluate_token``/``build_ai_context`` pair that gates on the
DexScreener ``MarketData`` from ``fenrir.filters``. They are registered here
(discoverable) but are *off by default* — nothing activates them until the bot
config explicitly enables them.
"""

from fenrir.strategies.base import StrategyState, TradeParams, TradingStrategy
from fenrir.strategies.graduation import GraduationStrategy
from fenrir.strategies.migration_snipe import MigrationSniperStrategy
from fenrir.strategies.narrative import NarrativeTrackerStrategy
from fenrir.strategies.reversal import ReversalStrategy
from fenrir.strategies.sniper import (
    ConservativeSniperStrategy,
    DegenSniperStrategy,
    SniperStrategy,
)
from fenrir.strategies.volume_anomaly import VolumeAnomalyStrategy

# Strategy registry: strategy_id -> class
STRATEGY_REGISTRY: dict[str, type[TradingStrategy]] = {
    "sniper": SniperStrategy,
    "sniper_conservative": ConservativeSniperStrategy,
    "sniper_degen": DegenSniperStrategy,
    "graduation": GraduationStrategy,
    "migration_snipe": MigrationSniperStrategy,
    "reversal": ReversalStrategy,
    "volume_anomaly": VolumeAnomalyStrategy,
    "narrative_tracker": NarrativeTrackerStrategy,
}

# Strategies enabled unless the operator opts in. The signal-oriented
# strategies default OFF pending pipeline wiring; the existing sniper/
# graduation strategies remain the default-on set.
DEFAULT_DISABLED_STRATEGIES: frozenset[str] = frozenset(
    {
        "migration_snipe",
        "reversal",
        "volume_anomaly",
        "narrative_tracker",
    }
)


def get_strategy_class(strategy_id: str) -> type[TradingStrategy] | None:
    """Look up a strategy class by ID."""
    return STRATEGY_REGISTRY.get(strategy_id)


def list_strategies() -> list[str]:
    """List all available strategy IDs."""
    return list(STRATEGY_REGISTRY.keys())


def is_enabled_by_default(strategy_id: str) -> bool:
    """Whether a strategy runs without an explicit operator opt-in."""
    return strategy_id in STRATEGY_REGISTRY and strategy_id not in DEFAULT_DISABLED_STRATEGIES


__all__ = [
    "TradingStrategy",
    "TradeParams",
    "StrategyState",
    "SniperStrategy",
    "ConservativeSniperStrategy",
    "DegenSniperStrategy",
    "GraduationStrategy",
    "MigrationSniperStrategy",
    "ReversalStrategy",
    "VolumeAnomalyStrategy",
    "NarrativeTrackerStrategy",
    "STRATEGY_REGISTRY",
    "DEFAULT_DISABLED_STRATEGIES",
    "get_strategy_class",
    "list_strategies",
    "is_enabled_by_default",
]
