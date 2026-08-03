#!/usr/bin/env python3
"""
FENRIR - Signal Aggregator (Phase 5.2, confluence + cross-strategy ranking)

Now that heterogeneous strategies emit a common ``Signal`` (Phase 5.1), they can be
combined. The aggregator collects signals over a rolling window and answers two
questions the single-strategy world could not:

  - CONFLUENCE: are multiple *independent* strategies flagging the same token the same
    way? Independent agreement is worth more than any one strategy shouting — the
    combined conviction is a noisy-OR over the strongest signal per source, so two
    mediocre-but-independent reads beat one strong-but-lonely one.
  - RANKING: across every flagged token, which are the highest-conviction on the shared
    0-1 strength axis?

Pure and additive — it consumes ``Signal`` objects (or raw objects via
``normalize_signal``) and holds no execution state. Stale signals expire by TTL so the
window reflects only recent conviction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from fenrir.signals.adapters import normalize_signal
from fenrir.signals.models import Signal, SignalDirection


@dataclass
class ConfluenceResult:
    """Combined conviction on one token in one direction, across distinct sources."""

    token_address: str
    direction: SignalDirection
    sources: list[str]  # distinct producing strategies, sorted
    combined_strength: float  # noisy-OR over the per-source strongest signals
    max_strength: float  # the single strongest contributing signal
    signals: list[Signal] = field(default_factory=list)  # the per-source strongest

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def is_confluent(self, min_sources: int = 2) -> bool:
        """True when at least ``min_sources`` distinct strategies agree."""
        return self.source_count >= min_sources


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SignalAggregator:
    """Rolling-window collector for confluence detection and cross-strategy ranking."""

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        now_fn: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self._now = now_fn
        self._signals: list[Signal] = []

    # ── Ingestion ──────────────────────────────────────────────────────

    def add(self, signal: Signal) -> None:
        """Add an already-normalized signal to the window."""
        self._signals.append(signal)

    def ingest(self, obj: object) -> Signal:
        """Normalize a raw strategy signal / opportunity and add it. Returns the Signal."""
        signal = normalize_signal(obj)
        self.add(signal)
        return signal

    def clear(self) -> None:
        self._signals = []

    def _prune(self, now: datetime | None = None) -> None:
        current = now or self._now()
        cutoff = current - timedelta(seconds=self.ttl_seconds)
        self._signals = [s for s in self._signals if s.timestamp >= cutoff]

    @property
    def active_count(self) -> int:
        self._prune()
        return len(self._signals)

    # ── Confluence + ranking ───────────────────────────────────────────

    def confluence_for(
        self,
        token_address: str,
        direction: SignalDirection = SignalDirection.LONG,
        now: datetime | None = None,
    ) -> ConfluenceResult | None:
        """Combined conviction on a token in a direction, or None if nothing live.

        One strategy counts once (its strongest live signal), so a chatty strategy
        cannot fake confluence. The combined strength is a noisy-OR — it rises with
        each independent source and saturates toward 1.
        """
        self._prune(now)
        relevant = [
            s
            for s in self._signals
            if s.token_address == token_address and s.direction == direction and s.strength > 0.0
        ]
        if not relevant:
            return None

        strongest_by_source: dict[str, Signal] = {}
        for s in relevant:
            current = strongest_by_source.get(s.source)
            if current is None or s.strength > current.strength:
                strongest_by_source[s.source] = s

        signals = list(strongest_by_source.values())
        product = 1.0
        for s in signals:
            product *= 1.0 - s.strength
        combined = 1.0 - product

        return ConfluenceResult(
            token_address=token_address,
            direction=direction,
            sources=sorted(strongest_by_source.keys()),
            combined_strength=combined,
            max_strength=max(s.strength for s in signals),
            signals=signals,
        )

    def ranked(
        self,
        direction: SignalDirection = SignalDirection.LONG,
        min_sources: int = 1,
        now: datetime | None = None,
    ) -> list[ConfluenceResult]:
        """Every flagged token in a direction (with at least ``min_sources`` distinct
        strategies), ranked by combined conviction — confluence first, then strength."""
        self._prune(now)
        tokens = {
            s.token_address for s in self._signals if s.direction == direction and s.strength > 0.0
        }
        results: list[ConfluenceResult] = []
        for token in tokens:
            result = self.confluence_for(token, direction, now)
            if result is not None and result.source_count >= min_sources:
                results.append(result)

        results.sort(
            key=lambda r: (r.combined_strength, r.source_count, r.max_strength),
            reverse=True,
        )
        return results
