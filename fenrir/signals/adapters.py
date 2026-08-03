#!/usr/bin/env python3
"""
FENRIR - Signal normalization adapters (Phase 5)

Map each strategy's bespoke signal and the arbitrage opportunity onto the common
``Signal``. Deliberately DUCK-TYPED — no strategy/detector classes are imported here,
so this module stays decoupled and additive: strategies keep returning their own
objects, and normalization is centralized.

A strategy signal is recognized by its ``metadata["strategy"]`` tag and a 0-1 score
property; the per-strategy score attribute is mapped below, with a best-effort fallback
so a newly-added strategy still normalizes (its first ``*_score`` / ``*_strength``
property) without a code change here.
"""

from __future__ import annotations

from typing import Any

from fenrir.signals.models import Signal, SignalDirection

# strategy_id -> the 0-1 conviction property on its bespoke signal.
_STRATEGY_SCORE_ATTR = {
    "momentum": "momentum_score",
    "mean_reversion": "reversion_score",
    "volume_anomaly": "anomaly_score",
    "reversal": "recovery_strength",
    "narrative_tracker": "narrative_momentum_score",
    "migration_snipe": "urgency_score",
}

# Net edge (bps) that maps to full strength for an arbitrage divergence.
_ARB_STRENGTH_CAP_BPS = 300.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_strength(sig: Any, source: str) -> float:
    """The bespoke signal's 0-1 conviction. Uses the mapped attribute, else the first
    ``*_score`` / ``*_strength`` property (so unmapped/new strategies still normalize)."""
    attr = _STRATEGY_SCORE_ATTR.get(source)
    if attr and hasattr(sig, attr):
        try:
            return _clamp01(float(getattr(sig, attr)))
        except (TypeError, ValueError):
            pass
    for name in dir(sig):
        if name.startswith("_"):
            continue
        if name.endswith(("_score", "_strength")):
            try:
                return _clamp01(float(getattr(sig, name)))
            except (TypeError, ValueError):
                continue
    return 0.0


def normalize_strategy_signal(sig: Any, source: str | None = None) -> Signal:
    """Normalize a directional strategy signal (momentum, mean_reversion, …) to a
    LONG ``Signal``. ``source`` overrides the ``metadata["strategy"]`` tag."""
    metadata = dict(getattr(sig, "metadata", {}) or {})
    src = source or metadata.get("strategy")
    if not src:
        raise ValueError("cannot determine signal source (no metadata['strategy'])")
    strength = _extract_strength(sig, src)
    return Signal(
        source=src,
        token_address=getattr(sig, "token_address", "") or "",
        direction=SignalDirection.LONG,
        strength=strength,
        rationale=f"{src} signal (strength {strength:.2f})",
        symbol=getattr(sig, "symbol", "") or metadata.get("symbol", "") or "",
        metadata=metadata,
    )


def normalize_arbitrage(opp: Any) -> Signal:
    """Normalize an ``ArbOpportunity`` to a NEUTRAL (market-neutral) ``Signal``.
    Strength scales the net edge, saturating at ``_ARB_STRENGTH_CAP_BPS``."""
    strength = _clamp01(opp.net_edge_bps / _ARB_STRENGTH_CAP_BPS)
    return Signal(
        source="arbitrage",
        token_address=opp.token_address,
        direction=SignalDirection.NEUTRAL,
        strength=strength,
        rationale=(f"buy {opp.buy_venue} → sell {opp.sell_venue} (net {opp.net_edge_bps:.0f}bps)"),
        metadata={
            "buy_venue": opp.buy_venue,
            "sell_venue": opp.sell_venue,
            "net_edge_bps": opp.net_edge_bps,
            "est_profit_sol": opp.est_profit_sol,
            "size_sol": opp.size_sol,
        },
    )


def normalize_signal(obj: Any) -> Signal:
    """Dispatch to the right adapter for any known signal/opportunity object."""
    # ArbOpportunity: identified by its costed-divergence fields.
    if hasattr(obj, "net_edge_bps") and hasattr(obj, "buy_venue"):
        return normalize_arbitrage(obj)
    # Strategy signal: identified by a metadata tag.
    if hasattr(obj, "metadata"):
        return normalize_strategy_signal(obj)
    raise TypeError(f"cannot normalize object of type {type(obj).__name__}")
