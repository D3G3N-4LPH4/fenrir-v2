#!/usr/bin/env python3
"""
FENRIR Strategy: Range Rotation (Community-Coin Rebalancing)

The nobrainflip rebalancing play, as a playbook tagger: buy the bottom of the
range in a *settled* community coin and hold for the next leg, instead of
chasing the top of someone else's 3x. The structural edge is range position,
not prediction — a coin that keeps making ranges, bought at the bottom of one,
has ~5x better reward/risk than the same coin bought after its run.

What makes this distinct from the existing dip playbooks:
  - ``mean_reversion`` fades a 1h oversold dislocation on a 1h-48h-old token
    for a ~2h bounce trade.
  - ``reversal`` plays a 15m-2h-old launch drawing down from its ATH.
  - ``range_rotation`` wants a *settled* coin (12h+, no upper bound — months-old
    community coins qualify) sitting in a deep 24h dip with buyers stepping back
    in, deep liquidity to size, and distributed supply (the dev is long gone).
    Time horizon is days, not hours.

Entry logic:
  - Age: ≥ 12h (a coin with ranges behind it, not a fresh launch)
  - In a real dip: 24h change between -60% and -20% (deep enough to matter,
    not so deep the coin is dead)
  - Not in freefall: 1h change ≥ -15% and 5m change ≥ -2%
    (the knife has slowed — rotation happens at range bottoms, not mid-dump)
  - Buyers stepping back in: 1h buy/sell ≥ 1.1 (or 5m buy pressure ≥ 0.55)
  - Community-coin structure: liquidity ≥ $100k (deep enough to size),
    1h volume ≥ $25k (still alive), top holder ≤ 15% when known
    (distributed — nobody sitting on a dump)

Exit logic (conceptual — the strategy ships OFF, tagging only):
  - Take profit: +150% (the next range leg, not a scalp)
  - Trailing stop: 25%
  - Hard stop: -30% — the range bottom broke, the thesis is wrong
  - Max hold: 7 days

Risk: MEDIUM-HIGH — range bottoms can become new downtrends; the stabilization
+ buy-edge + distribution gates exist so the tag fires on bottoming ranges, not
on coins bleeding out. A coin that keeps making lower ranges will keep taking
the stop.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) with
the richer ``evaluate_token`` / ``RangeRotationSignal`` machinery gated on the
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

logger = logging.getLogger("FENRIR.RangeRotation")


@dataclass
class RangeRotationConfig:
    """Tunable parameters for the range rotation strategy."""

    # Settled-coin age floor (minutes). No ceiling — months-old coins qualify.
    min_age_minutes: float = 720.0  # 12 hours
    # The dip: 24h change must sit inside this band (deep, not dead).
    max_dip_24h_pct: float = -20.0  # change_24h must be <= this
    min_dip_24h_pct: float = -60.0  # change_24h must be >= this
    # Not in freefall right now.
    min_price_change_1h_pct: float = -15.0
    min_price_change_5m_pct: float = -2.0
    # Buyers stepping back in at the bottom.
    min_buy_sell_ratio_1h: float = 1.1
    min_buy_pressure_5m: float = 0.55
    # Community-coin structure: deep, alive, distributed.
    min_liquidity_usd: float = 100_000.0
    min_volume_1h_usd: float = 25_000.0
    max_top_holder_pct: float = 15.0  # applied only when holder data exists
    # Exit plan (conceptual — tagging only).
    take_profit_pct: float = 150.0
    trailing_stop_pct: float = 25.0
    stop_loss_pct: float = 30.0
    max_hold_days: float = 7.0
    # AI confidence threshold.
    ai_min_confidence: float = 0.62
    # Daily budget (0 = fall back to the shared per-strategy default).
    daily_budget_sol: float = 0.0


@dataclass
class RangeRotationSignal:
    """Signal for a range-bottom rotation opportunity on a settled coin."""

    token_address: str
    pair_address: str
    age_minutes: float
    market_cap_usd: float
    price_usd: float
    liquidity_usd: float
    volume_1h_usd: float
    buy_sell_ratio_1h: float | None
    buy_pressure_5m: float
    price_change_5m_pct: float
    price_change_1h_pct: float
    price_change_24h_pct: float
    top_holder_pct: float | None
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rotation_score(self) -> float:
        """0-1 score for how attractive the range-bottom setup is.

        Blends dip depth (24h -20%→-60% mapped 0→1), stabilization (5m
        -2%→+3% mapped 0→1), returning buy edge (1h ratio 1.1→1.8 mapped
        0→1), and liquidity depth ($100k→$500k mapped 0→1, capped).
        """
        dip = min(1.0, max(0.0, (-self.price_change_24h_pct - 20.0) / 40.0))
        stab = min(1.0, max(0.0, (self.price_change_5m_pct + 2.0) / 5.0))
        ratio = self.buy_sell_ratio_1h or 0.0
        edge = min(1.0, max(0.0, (ratio - 1.1) / 0.7))
        depth = min(1.0, max(0.0, (self.liquidity_usd - 100_000.0) / 400_000.0))
        return dip * 0.35 + stab * 0.25 + edge * 0.25 + depth * 0.15


class RangeRotationStrategy(TradingStrategy):
    """
    Range rotation (community-coin rebalancing) strategy.

    Tags settled, liquid, distributed coins sitting at the bottom of a deep
    24h range with buyers stepping back in — the "rotate into the laggard at
    range bottom" setup. Read-only in the scout; execution stays off.
    """

    strategy_id = "range_rotation"
    display_name = "Range Rotation"
    description = (
        "Tags a settled community coin (12h+, deep liquidity, distributed "
        "supply) bottoming in a deep 24h range with buyers returning — the "
        "rebalancing rotation setup. MEDIUM-HIGH risk."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig, params: RangeRotationConfig | None = None) -> None:
        super().__init__()
        self.config = config
        self.params = params or RangeRotationConfig()

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
            "# STRATEGY CONTEXT: RANGE ROTATION (COMMUNITY-COIN REBALANCING)\n"
            "You are evaluating a settled community coin sitting at the bottom of a "
            "deep range — the rotation setup is to buy the laggard at range bottom, "
            "not to chase the runner after its leg.\n"
            "Key factors for this strategy:\n"
            "- The coin must be settled (12h+ old, deep liquidity, distributed holders "
            "— the dev is long gone), not a fresh launch\n"
            "- The dip must be deep (24h -20% to -60%) but the bleeding must have "
            "slowed (1h/5m stabilizing) with buyers stepping back in\n"
            "- Red flags: a fresh launch masquerading as a dip, a 5m still dumping, "
            "thin liquidity, concentrated supply (someone waiting to dump on the bounce)\n"
            "- Green flags: months-old coin, prior ranges that recovered, buy/sell "
            "flipping positive at the lows, deep order books\n"
            "- Time horizon: days; take the next range leg (+150% target) — if the "
            "range bottom breaks (-30%), the thesis is wrong\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    # ── Rich signal machinery (used by the market-data stage) ──────────

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> RangeRotationSignal | None:
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        age_minutes = getattr(market_data, "age_minutes", 0.0)
        mcap = getattr(market_data, "market_cap_usd", 0.0)
        price_usd = getattr(market_data, "price_usd", 0.0)
        liq = getattr(market_data, "liquidity_usd", 0.0)
        vol_1h = getattr(market_data, "volume_1h_usd", 0.0)
        change_5m = getattr(market_data, "price_change_5m_pct", 0.0)
        change_1h = getattr(market_data, "price_change_1h_pct", 0.0)
        change_24h = getattr(market_data, "price_change_24h_pct", 0.0)
        ratio_1h = getattr(market_data, "buy_sell_ratio_1h", None)
        pressure_5m = getattr(market_data, "buy_pressure_5m", 0.5)
        top_holder = getattr(market_data, "top_holder_pct", None)
        pair_address = getattr(market_data, "pair_address", "") or ""

        # Settled coin — silent skip if too young.
        if age_minutes < self.params.min_age_minutes:
            return None

        failures = []

        # In a real dip — deep enough to matter…
        if change_24h > self.params.max_dip_24h_pct:
            failures.append(
                f"24h change {change_24h:+.1f}% — not a range bottom "
                f"(want ≤ {self.params.max_dip_24h_pct:.0f}%)"
            )
        # …but not dead.
        if change_24h < self.params.min_dip_24h_pct:
            failures.append(
                f"24h change {change_24h:+.0f}% < floor {self.params.min_dip_24h_pct:.0f}% "
                "(dead coin, not a dip)"
            )

        # Not in freefall right now — rotation happens at bottoms, not mid-dump.
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

        # Community-coin structure: deep, alive, distributed.
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < min ${self.params.min_liquidity_usd:,.0f}")
        if vol_1h < self.params.min_volume_1h_usd:
            failures.append(f"Vol(1h) ${vol_1h:,.0f} < min ${self.params.min_volume_1h_usd:,.0f}")
        if top_holder is not None and top_holder > self.params.max_top_holder_pct:
            failures.append(
                f"top holder {top_holder:.1f}% > {self.params.max_top_holder_pct:.0f}% "
                "(supply not distributed)"
            )

        if failures:
            logger.debug(f"Range rotation reject {token_address[:8]}...: {' | '.join(failures)}")
            return None

        signal = RangeRotationSignal(
            token_address=token_address,
            pair_address=pair_address,
            age_minutes=age_minutes,
            market_cap_usd=mcap,
            price_usd=price_usd,
            liquidity_usd=liq,
            volume_1h_usd=vol_1h,
            buy_sell_ratio_1h=ratio_1h,
            buy_pressure_5m=pressure_5m,
            price_change_5m_pct=change_5m,
            price_change_1h_pct=change_1h,
            price_change_24h_pct=change_24h,
            top_holder_pct=top_holder,
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
            f"Range rotation SIGNAL {token_address[:8]}... | "
            f"age={age_minutes/60:.1f}h 24h={change_24h:+.1f}% 5m={change_5m:+.1f}% "
            f"b/s={ratio_1h} rotation={signal.rotation_score:.2f}"
        )
        return signal

    def build_ai_context(self, signal: RangeRotationSignal) -> str:
        """Per-signal context injected into the AI prompt for this candidate."""
        return "\n".join(
            [
                f"Range rotation setup: {signal.token_address[:8]}...",
                f"Age {signal.age_minutes/60:.1f}h, 24h {signal.price_change_24h_pct:+.1f}%, "
                f"1h {signal.price_change_1h_pct:+.1f}%, 5m {signal.price_change_5m_pct:+.1f}%",
                f"1h buy/sell {signal.buy_sell_ratio_1h}, LP ${signal.liquidity_usd:,.0f}, "
                f"rotation score {signal.rotation_score:.2f}",
                f"Plan: +{signal.metadata['take_profit_pct']:.0f}% target, "
                f"-{signal.metadata['stop_loss_pct']:.0f}% stop, "
                f"{signal.metadata['max_hold_days']:.0f}d max hold.",
            ]
        )
