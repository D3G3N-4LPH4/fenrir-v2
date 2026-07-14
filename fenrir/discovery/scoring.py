#!/usr/bin/env python3
"""
FENRIR - Discovery scoring engine

Weighted 0–100 scoring over the chain-agnostic :class:`TokenSnapshot`. Each token
gets six component scores (momentum, safety, liquidity, holder, community, risk)
and a weighted ``overall``. Weights are configurable via :class:`ScoringWeights`.

All scorers are pure functions of the snapshot and degrade gracefully when optional
fields are ``None`` (fall back to a neutral 50 so a missing provider doesn't zero a
token out). ``risk`` is inverted (0 = safe, 100 = risky) and contributes to
``overall`` as ``100 - risk``.
"""

from __future__ import annotations

from dataclasses import dataclass

from fenrir.discovery.models import ScoreBreakdown, TokenSnapshot

NEUTRAL = 50.0


def _scale(value: float, lo: float, hi: float) -> float:
    """Linear map ``value`` in [lo, hi] → [0, 100], clamped."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(100.0, (value - lo) / (hi - lo) * 100.0))


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


@dataclass
class ScoringWeights:
    """Relative weights for the ``overall`` score (need not sum to 1 — normalized)."""

    momentum: float = 0.25
    safety: float = 0.25
    liquidity: float = 0.20
    holder: float = 0.15
    community: float = 0.05
    risk: float = 0.10  # weight on the (100 - risk) contribution

    def total(self) -> float:
        return (
            self.momentum + self.safety + self.liquidity + self.holder + self.community + self.risk
        )


class ScoringEngine:
    """Compute component + overall scores for a token snapshot."""

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score(self, snap: TokenSnapshot) -> ScoreBreakdown:
        momentum = self._momentum(snap)
        safety = self._safety(snap)
        liquidity = self._liquidity(snap)
        holder = self._holder(snap)
        community = self._community(snap)
        risk = self._risk(snap)

        w = self.weights
        total = w.total() or 1.0
        overall = (
            w.momentum * momentum
            + w.safety * safety
            + w.liquidity * liquidity
            + w.holder * holder
            + w.community * community
            + w.risk * (100.0 - risk)
        ) / total

        return ScoreBreakdown(
            overall=_clamp(overall),
            momentum=momentum,
            safety=safety,
            liquidity=liquidity,
            holder=holder,
            community=community,
            risk=risk,
        )

    # ── Component scorers (each 0–100) ────────────────────────────────

    @staticmethod
    def _momentum(snap: TokenSnapshot) -> float:
        vol = _scale(snap.volume_24h_usd, 2_000, 1_000_000)
        pressure = snap.buy_pressure_24h * 100.0  # 0.5 → 50
        trend = _scale(snap.price_change_24h_pct, -30.0, 60.0)
        return _clamp((vol + pressure + trend) / 3.0)

    @staticmethod
    def _safety(snap: TokenSnapshot) -> float:
        s = snap.safety
        score = 100.0
        # Bad/unknown contract states subtract; unknowns are lighter than negatives.
        score -= {True: 0.0, False: 40.0, None: 10.0}[s.mint_disabled]
        score -= {True: 0.0, False: 30.0, None: 8.0}[s.freeze_disabled]
        score -= {True: 0.0, False: 40.0, None: 10.0}[s.lp_locked_or_burned]
        score -= {True: 100.0, False: 0.0, None: 5.0}[s.honeypot]
        score -= {True: 0.0, False: 15.0, None: 5.0}[s.contract_verified]
        if s.blacklist_present is True:
            score -= 25.0
        if s.ownership_renounced is True:
            score += 5.0  # bonus
        for tax in (s.buy_tax_pct, s.sell_tax_pct):
            if tax is not None and tax > 10.0:
                score -= min(30.0, (tax - 10.0) * 2.0)
        if s.risk_score is not None:  # RugCheck-style: higher = riskier
            score -= _scale(s.risk_score, 0.0, 100.0) * 0.4
        return _clamp(score)

    @staticmethod
    def _liquidity(snap: TokenSnapshot) -> float:
        depth = _scale(snap.liquidity_usd, 1_000, 250_000)
        ratio = _scale(snap.liquidity_to_mcap, 0.02, 0.20)  # 2%–20% of mcap
        return _clamp((depth + ratio) / 2.0)

    @staticmethod
    def _holder(snap: TokenSnapshot) -> float:
        count = _scale(snap.holder_count, 25, 3_000) if snap.holder_count is not None else NEUTRAL
        dist = 100.0
        if snap.top_holder_pct is not None:
            dist -= max(0.0, snap.top_holder_pct - 5.0) * 4.0  # penalize >5%
        if snap.dev_wallet_pct is not None:
            dist -= max(0.0, snap.dev_wallet_pct - 3.0) * 4.0
        return _clamp((count + _clamp(dist)) / 2.0)

    @staticmethod
    def _community(snap: TokenSnapshot) -> float:
        socials = sum(33.4 for x in (snap.twitter, snap.telegram, snap.website) if x)
        if snap.organic_score is not None:
            socials = (socials + _scale(snap.organic_score, 0.0, 100.0)) / 2.0
        return _clamp(socials)

    @staticmethod
    def _risk(snap: TokenSnapshot) -> float:
        """0 = safe, 100 = risky. Driven by concentration + hostile-contract flags."""
        risk = 0.0
        risk += max(0.0, (snap.sniper_pct or 0.0)) * 1.5
        risk += max(0.0, (snap.bundle_pct or 0.0)) * 2.0
        risk += max(0.0, (snap.insider_pct or 0.0)) * 2.0
        risk += max(0.0, (snap.dev_wallet_pct or 0.0) - 3.0) * 2.0
        if snap.safety.honeypot is True:
            risk += 100.0
        if snap.safety.blacklist_present is True:
            risk += 40.0
        if snap.safety.mint_disabled is False:
            risk += 25.0
        if snap.safety.risk_score is not None:
            risk += _scale(snap.safety.risk_score, 0.0, 100.0) * 0.5
        return _clamp(risk)
