#!/usr/bin/env python3
"""
FENRIR - Unified Signal model (Phase 5, signal generalization)

Every strategy and market-neutral detector today produces its own bespoke result —
MomentumSignal / MeanReversionSignal / VolumeAnomalySignal / ArbOpportunity / … —
each with a differently-named 0-1 score and its own shape. That makes them impossible
to compare, rank, or combine across strategies.

``Signal`` is the common denominator: a normalized, source-tagged, directional
conviction on a token. Adapters (``fenrir.signals.adapters``) map each bespoke result
onto it, so downstream consumers (ranking, confluence, alerting) can treat a momentum
long and an arbitrage divergence uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class SignalDirection(Enum):
    """Which way the signal leans."""

    LONG = "long"  # buy / accumulate (all current entry strategies)
    SHORT = "short"  # sell / fade (reserved; no spot-short venue yet)
    NEUTRAL = "neutral"  # market-neutral (arbitrage, market-making)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class Signal:
    """A normalized conviction from any strategy or detector.

    ``strength`` is the single cross-strategy comparable: a 0-1 conviction, higher =
    stronger. ``source`` identifies the producing strategy/detector, ``metadata`` keeps
    the original details, and ``rationale`` is a short human summary.
    """

    source: str
    token_address: str
    direction: SignalDirection
    strength: float  # 0-1, clamped
    rationale: str = ""
    symbol: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        # Strength is the comparable axis — keep it in [0, 1] no matter what fed it.
        self.strength = max(0.0, min(1.0, float(self.strength)))

    @property
    def is_actionable(self) -> bool:
        """A positive-conviction signal (strength above zero)."""
        return self.strength > 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "token_address": self.token_address,
            "direction": self.direction.value,
            "strength": self.strength,
            "rationale": self.rationale,
            "symbol": self.symbol,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
