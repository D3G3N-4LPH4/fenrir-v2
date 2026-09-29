#!/usr/bin/env python3
"""
FENRIR Strategy: Flush Recovery (Post-Flush Second Leg)

The kioto $100M-runner pattern, as a playbook tagger: every big Robinhood
runner follows the same structure — initial traction rally, euphoria fade,
then a -60% to -95% flush that shakes out short-term buyers and migrates
supply to strong holders (plus a "team" accumulating for the next run). When
new demand returns it hits a thin sell side and the move goes parabolic.
"Robinhood is a holders chain. Not a flippers chain."

What makes this distinct from the existing dip playbooks:
  - ``mean_reversion`` fades a 1h oversold dislocation on a 1h-48h-old token
    for a ~2h bounce trade.
  - ``reversal`` plays a 15m-2h-old launch drawing down from its ATH.
  - ``range_rotation`` buys the bottom of a -20%..-60% 24h range in a settled
    (12h+) community coin.
  - ``flush_recovery`` goes deeper: the -60%..-95% flush leg itself, entered
    only once the knife has stopped (1h/5m stabilized) AND buyers are stepping
    back in AND the holder base survived the flush (holders didn't flee —
    the supply-migration leg of the pattern). Time horizon is days to weeks;
    the target is the second leg, not a bounce.

Entry logic:
  - Age: ≥ 6h (the flush leg takes time to play out)
  - In the flush: 24h change between -95% and -60% (flushed, not dead)
  - Not still dumping: 1h change ≥ -10% and 5m change ≥ -3%
  - Buyers stepping back in: 1h buy/sell ≥ 1.05 (or 5m buy pressure ≥ 0.52)
  - Still a real coin: liquidity ≥ $50k, 24h volume ≥ $200k, 1h volume ≥ $25k
  - Supply migrated, not concentrated: top holder ≤ 25% when known
  - Holder resilience: poll-over-poll holder growth ≥ 0.85 when acceleration
    data is available (fail-open when missing — the flush did not nuke the
    holder base, i.e. supply moved to strong hands instead of exiting)

Exit logic (conceptual — the strategy ships OFF, tagging only):
  - Take profit: +200% (the second leg, not a scalp)
  - Trailing stop: 30%
  - Hard stop: -35% — the bottom broke, the thesis is wrong
  - Max hold: 14 days

Risk: HIGH — post-flush bottoms can keep bleeding into dead coins; the
stabilization + buy-edge + holder-resilience gates exist so the tag fires on
flushes that are turning, not on coins still being distributed into the bid.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) with
the richer ``evaluate_token`` / ``FlushRecoverySignal`` machinery gated on the
DexScreener ``MarketData`` produced by ``fenrir.filters``. Off by default
(opt-in); the scout uses it read-only as a playbook tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy

logger = logging.getLogger("FENRIR.FlushRecovery")


@dataclass
class FlushRecoveryConfig:
    """Tunable parameters for the flush recovery strategy."""

    # The flush leg takes time — no fresh launches.
    min_age_minutes: float = 360.0  # 6 hours
    # The flush: 24h change must sit inside this band (flushed, not dead).
    max_flush_24h_pct: float = -60.0  # change_24h must be <= this
    min_flush_24h_pct: float = -95.0  # change_24h must be >= this
    # Not still dumping right now.
    min_price_change_1h_pct: float = -10.0
    min_price_change_5m_pct: float = -3.0
    # Buyers stepping back in at the lows.
    min_buy_sell_ratio_1h: float = 1.05
    min_buy_pressure_5m: float = 0.52
    # Still a real coin: deep enough to trade, tape still alive.
    min_liquidity_usd: float = 50_000.0
    min_volume_24h_usd: float = 200_000.0
    min_volume_1h_usd: float = 25_000.0
    # Supply migrated to strong hands, not concentrated in one dumper.
    max_top_holder_pct: float = 25.0  # applied only when holder data exists
    # Holder resilience: holder count vs the previous scout poll. The flush
    # must not have nuked the holder base (Sparsity's "supply held" idea).
    # Fail-open when acceleration data is unavailable.
    min_holder_resilience: float = 0.85
    # Exit plan (conceptual — tagging only).
    take_profit_pct: float = 200.0
    trailing_stop_pct: float = 30.0
    stop_loss_pct: float = 35.0
    max_hold_days: float = 14.0
    # AI confidence threshold.
    ai_min_confidence: float = 0.62
    # Daily budget (0 = fall back to the shared per-strategy default).
    daily_budget_sol: float = 0.0


@dataclass
class FlushRecoverySignal:
    """Signal for a post-flush second-leg opportunity."""

    token_address: str
    pair_address: str
    age_minutes: float
    market_cap_usd: float
    price_usd: float
    liquidity_usd: float
    volume_24h_usd: float
    volume_1h_usd: float
    buy_sell_ratio_1h: float | None
    buy_pressure_5m: float
    price_change_5m_pct: float
    price_change_1h_pct: float
    price_change_24h_pct: float
    top_holder_pct: float | None
    holder_resilience: float | None
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def flush_score(self) -> float:
        """0-1 score for how clean the post-flush setup is.

        Blends flush depth (24h -60%→-95% mapped 0→1), stabilization (5m
        -3%→+3% mapped 0→1), returning buy edge (1h ratio 1.05→1.8 mapped
        0→1), and holder resilience (0.85→1.2 mapped 0→1; neutral 0.5 when
        the acceleration data is missing).
        """
        dip = min(1.0, max(0.0, (-self.price_change_24h_pct - 60.0) / 35.0))
        stab = min(1.0, max(0.0, (self.price_change_5m_pct + 3.0) / 6.0))
        ratio = self.buy_sell_ratio_1h or 0.0
        edge = min(1.0, max(0.0, (ratio - 1.05) / 0.75))
        if self.holder_resilience is None:
            resil = 0.5
        else:
            resil = min(1.0, max(0.0, (self.holder_resilience - 0.85) / 0.35))
        return dip * 0.30 + stab * 0.25 + edge * 0.30 + resil * 0.15


class FlushRecoveryStrategy(TradingStrategy):
    """
    Flush recovery (post-flush second leg) strategy.

    Tags coins sitting in a -60%..-95% 24h flush that has stabilized, with
    buyers stepping back in and the holder base intact — the setup that
    precedes the parabolic second leg. Read-only in the scout; execution
    stays off.
    """

    strategy_id = "flush_recovery"
    display_name = "Flush Recovery"
    description = (
        "Tags a post-flush second-leg setup (-60%..-95% 24h flush, stabilized "
        "short windows, buyers returning, holder base intact) — the kioto "
        "$100M-runner structure. HIGH risk."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig, params: FlushRecoveryConfig | None = None) -> None:
        super().__init__()
        self.config = config
        self.params = params or FlushRecoveryConfig()

        self._params = TradeParams(
            buy_amount_sol=config.buy_amount_sol,
            max_slippage_bps=config.max_slippage_bps,
            stop_loss_pct=self.params.stop_loss_pct,
            take_profit_pct=self.params.take_profit_pct,
            trailing_stop_pct=self.params.trailing_stop_pct,
            max_position_age_minutes=int(self.params.max_hold_days * 24 * 60),
            priority_fee_lamports=config.priority_fee_lamports,
            ai_min_confidence=self.params.ai_min_confidence,
            ai_temperature=config.ai_temperature,
            ai_entry_timeout=config.ai_entry_timeout_seconds,
        )

    # ── ABC interface ──────────────────────────────────────────────────

    async def should_evaluate(self, token_data: dict) -> bool:
        """Cheap pre-filter on token_data only. Real gating is in evaluate_token."""
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: FLUSH RECOVERY (POST-FLUSH SECOND LEG)\n"
            "You are evaluating a coin that just flushed -60% to -95% — the setup "
            "is the second leg AFTER the flush, not the flush itself.\n"
            "Key factors for this strategy:\n"
            "- The flush must be real (24h -60% to -95%) but the coin must not be "
            "dead: the knife has stopped (1h/5m stabilized) and buyers are "
            "stepping back in\n"
            "- The critical confirmation is holder resilience: the flush must "
            "have migrated supply to strong hands, not nuked the holder base\n"
            "- Red flags: 1h still dumping, no buy edge at the lows, holder "
            "count collapsing, one whale holding the supply\n"
            "- Green flags: buy/sell flipping positive at the lows, holder "
            "count steady or growing through the flush, thin sell side\n"
            "- Time horizon: days to weeks; target the second leg (+200%) — if "
            "the bottom breaks (-35%), the thesis is wrong\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    # ── Rich signal machinery (used by the market-data stage) ──────────

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> FlushRecoverySignal | None:
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        age_minutes = getattr(market_data, "age_minutes", 0.0)
        mcap = getattr(market_data, "market_cap_usd", 0.0)
        price_usd = getattr(market_data, "price_usd", 0.0)
        liq = getattr(market_data, "liquidity_usd", 0.0)
        vol_24h = getattr(market_data, "volume_24h_usd", 0.0)
        vol_1h = getattr(market_data, "volume_1h_usd", 0.0)
        change_5m = getattr(market_data, "price_change_5m_pct", 0.0)
        change_1h = getattr(market_data, "price_change_1h_pct", 0.0)
        change_24h = getattr(market_data, "price_change_24h_pct", 0.0)
        ratio_1h = getattr(market_data, "buy_sell_ratio_1h", None)
        pressure_5m = getattr(market_data, "buy_pressure_5m", 0.5)
        top_holder = getattr(market_data, "top_holder_pct", None)
        resilience = getattr(market_data, "accel_holder_growth", None)
        pair_address = getattr(market_data, "pair_address", "") or ""

        # The flush leg takes time — silent skip if too young.
        if age_minutes < self.params.min_age_minutes:
            return None

        failures = []

        # In the flush — deep enough to be the real flush…
        if change_24h > self.params.max_flush_24h_pct:
            failures.append(
                f"24h change {change_24h:+.1f}% — not a flush "
                f"(want ≤ {self.params.max_flush_24h_pct:.0f}%)"
            )
        # …but not a dead coin.
        if change_24h < self.params.min_flush_24h_pct:
            failures.append(
                f"24h change {change_24h:+.0f}% < floor {self.params.min_flush_24h_pct:.0f}% "
                "(dead coin, not a flush)"
            )

        # Not still dumping right now — the second leg starts at stabilization.
        if change_1h < self.params.min_price_change_1h_pct:
            failures.append(
                f"1h change {change_1h:+.1f}% — still dumping "
                f"(want ≥ {self.params.min_price_change_1h_pct:.0f}%)"
            )
        if change_5m < self.params.min_price_change_5m_pct:
            failures.append(
                f"5m change {change_5m:+.1f}% — no stabilization yet "
                f"(want ≥ {self.params.min_price_change_5m_pct:.0f}%)"
            )

        # Buyers stepping back in at the lows.
        edge_1h = ratio_1h is not None and ratio_1h >= self.params.min_buy_sell_ratio_1h
        edge_5m = pressure_5m >= self.params.min_buy_pressure_5m
        if not (edge_1h or edge_5m):
            r = f"{ratio_1h:.2f}" if ratio_1h is not None else "n/a"
            failures.append(
                f"no buy edge at the bottom (1h b/s {r}, 5m pressure {pressure_5m:.2f})"
            )

        # Still a real coin: deep enough to trade, tape still alive.
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < min ${self.params.min_liquidity_usd:,.0f}")
        if vol_24h < self.params.min_volume_24h_usd:
            failures.append(f"Vol24h ${vol_24h:,.0f} < min ${self.params.min_volume_24h_usd:,.0f}")
        if vol_1h < self.params.min_volume_1h_usd:
            failures.append(f"Vol(1h) ${vol_1h:,.0f} < min ${self.params.min_volume_1h_usd:,.0f}")
        if top_holder is not None and top_holder > self.params.max_top_holder_pct:
            failures.append(
                f"top holder {top_holder:.1f}% > {self.params.max_top_holder_pct:.0f}% "
                "(supply not migrated)"
            )

        # Holder resilience: the flush must have migrated supply to strong
        # hands, not nuked the holder base. Fail-open when the acceleration
        # data is missing (holder coverage varies by chain/provider).
        if resilience is not None and resilience < self.params.min_holder_resilience:
            failures.append(
                f"holder resilience {resilience:.2f}x < {self.params.min_holder_resilience:.2f}x "
                "(holders fled the flush — no supply migration)"
            )

        if failures:
            logger.debug(f"Flush recovery reject {token_address[:8]}...: {' | '.join(failures)}")
            return None

        signal = FlushRecoverySignal(
            token_address=token_address,
            pair_address=pair_address,
            age_minutes=age_minutes,
            market_cap_usd=mcap,
            price_usd=price_usd,
            liquidity_usd=liq,
            volume_24h_usd=vol_24h,
            volume_1h_usd=vol_1h,
            buy_sell_ratio_1h=ratio_1h,
            buy_pressure_5m=pressure_5m,
            price_change_5m_pct=change_5m,
            price_change_1h_pct=change_1h,
            price_change_24h_pct=change_24h,
            top_holder_pct=top_holder,
            holder_resilience=resilience,
            metadata={
                "strategy": self.strategy_id,
                "stop_loss_pct": self.params.stop_loss_pct,
                "take_profit_pct": self.params.take_profit_pct,
                "trailing_stop_pct": self.params.trailing_stop_pct,
                "max_hold_days": self.params.max_hold_days,
                "ai_min_confidence": self.params.ai_min_confidence,
            },
        )

        logger.info(
            f"Flush recovery SIGNAL {token_address[:8]}... | "
            f"age={age_minutes/60:.1f}h 24h={change_24h:+.1f}% 5m={change_5m:+.1f}% "
            f"b/s={ratio_1h} resil={resilience} flush={signal.flush_score:.2f}"
        )
        return signal

    def build_ai_context(self, signal: FlushRecoverySignal) -> str:
        """Per-signal context injected into the AI prompt for this candidate."""
        return "\n".join(
            [
                f"Flush recovery setup: {signal.token_address[:8]}...",
                f"Age {signal.age_minutes/60:.1f}h, 24h {signal.price_change_24h_pct:+.1f}%, "
                f"1h {signal.price_change_1h_pct:+.1f}%, 5m {signal.price_change_5m_pct:+.1f}%",
                f"1h buy/sell {signal.buy_sell_ratio_1h}, holder resilience "
                f"{signal.holder_resilience}, LP ${signal.liquidity_usd:,.0f}, "
                f"flush score {signal.flush_score:.2f}",
                f"Plan: +{signal.metadata['take_profit_pct']:.0f}% target, "
                f"-{signal.metadata['stop_loss_pct']:.0f}% stop, "
                f"{signal.metadata['max_hold_days']:.0f}d max hold.",
            ]
        )
