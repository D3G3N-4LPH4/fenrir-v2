#!/usr/bin/env python3
"""
FENRIR - Discovery configuration

Assembled from ``BotConfig.build_discovery_config()`` (env-driven, opt-in). Keeps
the discovery layer self-contained: filter thresholds, scoring weights and the
universal-safety gate all live here, defaulting to the spec values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fenrir.discovery.filters import (
    DEFAULT_THRESHOLDS,
    FilterName,
    FilterThresholds,
    UniversalSafety,
)
from fenrir.discovery.models import Chain
from fenrir.discovery.scoring import ScoringWeights


def _default_thresholds() -> dict[FilterName, FilterThresholds]:
    return dict(DEFAULT_THRESHOLDS)


@dataclass
class DiscoveryConfig:
    """Runtime config for the multi-chain discovery scanner (off by default)."""

    enabled: bool = False
    chains: list[Chain] = field(default_factory=lambda: [Chain.SOLANA])
    # Jupiter feeds the Solana adapter discovers from (deduped). ``recent`` is the
    # newly-created-pairs feed that gives Low/Mid Cap filters genuinely fresh
    # launches; ``toptrending``/``toporganicscore`` cover established momentum.
    solana_categories: list[str] = field(
        default_factory=lambda: ["toptrending", "toporganicscore", "recent"]
    )
    filters: list[FilterName] = field(
        default_factory=lambda: [
            FilterName.LOW_CAP_ALPHA,
            FilterName.MID_CAP_MOMENTUM,
            FilterName.HIGH_CAP,
        ]
    )
    interval_seconds: float = 120.0
    max_candidates_per_cycle: int = 25
    cooldown_minutes: float = 30.0
    # Emit an alert / surface a discovery when its overall score is at/above this.
    min_alert_score: float = 70.0

    thresholds: dict[FilterName, FilterThresholds] = field(default_factory=_default_thresholds)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    universal_safety: UniversalSafety = field(default_factory=UniversalSafety)
