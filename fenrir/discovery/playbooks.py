#!/usr/bin/env python3
"""
FENRIR - Playbook tagging for discovery candidates.

Runs a TokenSnapshot through the market-data strategies' ``evaluate_token()``
in read-only mode (strategies are evaluated, never executed — no trades, no
orders, no position state) and reports which playbook(s) the token fits, with
per-playbook conviction and cross-strategy confluence.

The six market-data strategies gate on ``MarketData`` fields via ``getattr``;
``TokenSnapshot`` carries the same field names, so a snapshot can serve as the
market-data object directly. Launch-event strategies (sniper, graduation) are
excluded — they evaluate raw launch events, not market snapshots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fenrir.config import BotConfig
from fenrir.signals.adapters import normalize_strategy_signal
from fenrir.signals.aggregator import SignalAggregator
from fenrir.signals.models import SignalDirection
from fenrir.strategies import get_strategy_class
from fenrir.strategies.base import TradingStrategy

logger = logging.getLogger("FENRIR.Playbooks")

# strategy_id -> class must exist in the registry and expose evaluate_token().
PLAYBOOK_STRATEGY_IDS: tuple[str, ...] = (
    "momentum",
    "volume_anomaly",
    "migration_snipe",
    "narrative_tracker",
    "mean_reversion",
    "reversal",
)


@dataclass
class PlaybookMatch:
    """One strategy's read on a token."""

    strategy_id: str
    display_name: str
    strength: float  # 0-1 conviction, cross-strategy comparable
    rationale: str


@dataclass
class PlaybookTags:
    """Everything the tagger concluded about one snapshot."""

    matches: list[PlaybookMatch] = field(default_factory=list)
    confluent: bool = False  # >=2 independent strategies agree (LONG)
    combined_strength: float = 0.0  # noisy-OR over independent sources
    sources: list[str] = field(default_factory=list)

    @property
    def playbook_ids(self) -> list[str]:
        return [m.strategy_id for m in self.matches]

    def as_dict(self) -> dict:
        return {
            "playbooks": [
                {
                    "strategy_id": m.strategy_id,
                    "display_name": m.display_name,
                    "strength": round(m.strength, 3),
                    "rationale": m.rationale,
                }
                for m in self.matches
            ],
            "confluent": self.confluent,
            "combined_strength": round(self.combined_strength, 3),
        }


class PlaybookTagger:
    """Read-only strategy evaluation for discovery snapshots.

    Strategies are instantiated with a default ``BotConfig`` (no secrets, no
    network) and flipped to ``state.active`` purely so ``evaluate_token()``
    runs its gates — trading entry points are never called.
    """

    def __init__(self, strategy_ids: tuple[str, ...] = PLAYBOOK_STRATEGY_IDS) -> None:
        config = BotConfig()
        self._strategies: list[tuple[str, TradingStrategy]] = []
        for sid in strategy_ids:
            cls: Any = get_strategy_class(sid)  # concrete ctor takes a BotConfig
            if cls is None:
                logger.warning("playbook strategy %s not in registry — skipping", sid)
                continue
            if not hasattr(cls, "evaluate_token"):
                logger.warning("playbook strategy %s has no evaluate_token — skipping", sid)
                continue
            try:
                strat = cls(config)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("could not instantiate playbook %s: %s", sid, exc)
                continue
            strat.state.active = True  # read-only evaluation mode
            self._strategies.append((sid, strat))
        self._aggregator = SignalAggregator()

    @property
    def strategy_ids(self) -> list[str]:
        return [sid for sid, _ in self._strategies]

    def tag(self, snapshot) -> PlaybookTags:
        """Evaluate one TokenSnapshot against every playbook strategy."""
        token_data = {
            "token_address": snapshot.token_address,
            "symbol": snapshot.symbol,
            "name": snapshot.name,
        }
        self._aggregator.clear()
        tags = PlaybookTags()
        for sid, strat in self._strategies:
            try:
                sig = strat.evaluate_token(token_data, snapshot)
            except Exception as exc:
                # One misbehaving strategy never kills the tagging run.
                logger.debug("playbook %s raised on %s: %s", sid, snapshot.token_address[:12], exc)
                continue
            if sig is None:
                continue
            try:
                norm = normalize_strategy_signal(sig, source=sid)
            except Exception as exc:
                logger.debug("could not normalize %s signal: %s", sid, exc)
                continue
            self._aggregator.add(norm)
            display = getattr(strat, "display_name", sid)
            tags.matches.append(
                PlaybookMatch(
                    strategy_id=sid,
                    display_name=display,
                    strength=norm.strength,
                    rationale=norm.rationale,
                )
            )
        conf = self._aggregator.confluence_for(snapshot.token_address, SignalDirection.LONG)
        if conf is not None:
            tags.sources = conf.sources
            tags.combined_strength = conf.combined_strength
            tags.confluent = conf.is_confluent(min_sources=2)
        # Strongest conviction first.
        tags.matches.sort(key=lambda m: -m.strength)
        return tags
