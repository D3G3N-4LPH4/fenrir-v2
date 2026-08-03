#!/usr/bin/env python3
"""
FENRIR Strategy: Momentum Continuation (Trend Riding)

Rides an established, *accelerating* uptrend rather than sniping a launch or
buying a dip. The thesis: a token already moving up on rising volume and
buy-side dominance tends to keep moving long enough for a trend trade — as long
as it is not yet parabolic (blow-off top).

Entry logic:
  - Age: past the launch chaos but still early (30 min – 12 h)
  - Uptrend intact NOW: 1h change ≥ +12% AND 5m change ≥ +1% (still rising,
    not rolling over)
  - Not overextended: 24h change ≤ +400% (avoid buying the blow-off top)
  - Volume accelerating: the recent 5-min pace exceeds the trailing hourly pace
    (volume_5m/5min > volume_1h/60min × 1.2)
  - Buy-side dominance: 5m buy pressure ≥ 0.55
  - Sufficient liquidity for a clean exit

Exit logic:
  - Take profit: +60% (trend trade, not a moonshot hold)
  - Trailing stop: 15% — momentum reverses fast; trail to lock the run
  - Hard stop: -15%
  - Max hold: 3 hours

Risk: MODERATE-HIGH — trend-following buys strength, so a sharp reversal hits
before the trailing stop can react; the acceleration + buy-pressure gates and
the parabolic cap exist to avoid entering exhaustion moves.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) with the
richer ``evaluate_token`` / ``MomentumSignal`` machinery gated on the DexScreener
``MarketData`` produced by ``fenrir.filters``. Off by default (opt-in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy

logger = logging.getLogger("FENRIR.Momentum")


@dataclass
class MomentumConfig:
    """Tunable parameters for the momentum continuation strategy."""

    # Age window (minutes): past launch chaos, still with runway ahead.
    min_age_minutes: float = 30.0
    max_age_minutes: float = 720.0  # 12 hours
    # Uptrend gates — the move must still be alive right now.
    min_price_change_1h_pct: float = 12.0
    min_price_change_5m_pct: float = 1.0
    # Overextension cap — refuse to chase a parabolic blow-off top.
    max_price_change_24h_pct: float = 400.0
    # Volume acceleration: recent 5m pace vs. trailing hourly pace.
    min_volume_acceleration: float = 1.2
    # Absolute 1h volume floor (organic participation).
    min_volume_1h_usd: float = 50_000.0
    # Buy-side dominance over the last 5 minutes.
    min_buy_pressure_5m: float = 0.55
    # Liquidity floor for a clean exit.
    min_liquidity_usd: float = 50_000.0
    # Exit plan.
    take_profit_pct: float = 60.0
    trailing_stop_pct: float = 15.0
    stop_loss_pct: float = 15.0
    max_hold_hours: float = 3.0
    # AI confidence threshold.
    ai_min_confidence: float = 0.60
    # Daily budget (0 = fall back to the shared per-strategy default).
    daily_budget_sol: float = 0.0


@dataclass
class MomentumSignal:
    """Signal for a momentum continuation opportunity."""

    token_address: str
    pair_address: str
    age_minutes: float
    market_cap_usd: float
    price_usd: float
    liquidity_usd: float
    volume_5m_usd: float
    volume_1h_usd: float
    volume_acceleration: float
    buy_pressure_5m: float
    price_change_5m_pct: float
    price_change_1h_pct: float
    price_change_24h_pct: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def momentum_score(self) -> float:
        """0-1 score for how strong the momentum is.

        Blends trend strength (1h change, normalized to a +50% cap), volume
        acceleration (how much hotter the last 5m ran vs. the hourly pace, capped
        at 3x), and buy-side dominance (0.5→1.0 pressure mapped to 0→1).
        """
        trend_score = min(1.0, max(0.0, self.price_change_1h_pct) / 50.0)
        accel_score = min(1.0, max(0.0, self.volume_acceleration - 1.0) / 2.0)
        pressure_score = min(1.0, max(0.0, (self.buy_pressure_5m - 0.5) / 0.5))
        return (trend_score * 0.45) + (accel_score * 0.35) + (pressure_score * 0.20)


class MomentumStrategy(TradingStrategy):
    """
    Momentum continuation (trend-riding) strategy.

    Enters tokens already trending up on accelerating volume and buy-side
    dominance — provided the move is not yet parabolic — and rides the trend
    with a trailing stop.
    """

    strategy_id = "momentum"
    display_name = "Momentum Continuation"
    description = (
        "Rides an established, accelerating uptrend (1h +12%+, volume building, "
        "buyers dominant) that is not yet parabolic, trailing the run for a ~60% "
        "trend trade. MODERATE-HIGH risk."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig) -> None:
        super().__init__()
        self.config = config
        self.params = MomentumConfig()

        self._params = TradeParams(
            buy_amount_sol=config.buy_amount_sol,
            max_slippage_bps=config.max_slippage_bps,
            stop_loss_pct=self.params.stop_loss_pct,
            take_profit_pct=self.params.take_profit_pct,
            trailing_stop_pct=self.params.trailing_stop_pct,
            max_position_age_minutes=int(self.params.max_hold_hours * 60),
            priority_fee_lamports=config.priority_fee_lamports,
            ai_min_confidence=self.params.ai_min_confidence,
            ai_temperature=config.ai_temperature,
            ai_entry_timeout=config.ai_entry_timeout_seconds,
        )

    # ── ABC interface ──────────────────────────────────────────────────

    async def should_evaluate(self, token_data: dict) -> bool:
        """Cheap pre-filter on token_data only. Momentum depends on price/volume
        trends that require a MarketData snapshot, so the real gating happens in
        ``evaluate_token``."""
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: MOMENTUM CONTINUATION\n"
            "You are evaluating a token that is already trending up on rising "
            "volume, entered to ride the trend — not a launch snipe or a dip buy.\n"
            "Key factors for this strategy:\n"
            "- The move must still be alive: recent 5m and 1h both positive\n"
            "- Volume should be accelerating (last 5 min hotter than the hourly pace)\n"
            "- Buyers must dominate sellers (buy pressure)\n"
            "- Red flags: a parabolic 24h run nearing exhaustion, volume that is "
            "fading rather than building, buy pressure driven by a few large wallets "
            "(distribution risk), thin liquidity that will slip the exit\n"
            "- Green flags: steady higher-timeframe uptrend, broad two-sided flow "
            "with a buy skew, volume expanding into the move, deep liquidity\n"
            "- Time horizon: minutes to a few hours; ride with a trailing stop and "
            "exit on target or trend break — momentum reverses fast\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    # ── Rich signal machinery (used by the market-data stage) ──────────

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> MomentumSignal | None:
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        age_minutes = getattr(market_data, "age_minutes", 0.0)
        mcap = getattr(market_data, "market_cap_usd", 0.0)
        price_usd = getattr(market_data, "price_usd", 0.0)
        liq = getattr(market_data, "liquidity_usd", 0.0)
        vol_5m = getattr(market_data, "volume_5m_usd", 0.0)
        vol_1h = getattr(market_data, "volume_1h_usd", 0.0)
        change_5m = getattr(market_data, "price_change_5m_pct", 0.0)
        change_1h = getattr(market_data, "price_change_1h_pct", 0.0)
        change_24h = getattr(market_data, "price_change_24h_pct", 0.0)
        buy_pressure = getattr(market_data, "buy_pressure_5m", 0.5)
        pair_address = getattr(market_data, "pair_address", "") or ""

        # Age gate — silent skip if outside window.
        if not (self.params.min_age_minutes <= age_minutes <= self.params.max_age_minutes):
            return None

        # Volume acceleration: recent per-minute pace vs. the trailing hourly pace.
        # 1h pace is the baseline; if there is no hourly volume there is no trend.
        hourly_pace = vol_1h / 60.0
        recent_pace = vol_5m / 5.0
        acceleration = recent_pace / hourly_pace if hourly_pace > 0 else 0.0

        failures = []

        # Uptrend must still be intact right now.
        if change_1h < self.params.min_price_change_1h_pct:
            failures.append(
                f"1h change {change_1h:+.1f}% < min +{self.params.min_price_change_1h_pct:.0f}%"
            )
        if change_5m < self.params.min_price_change_5m_pct:
            failures.append(
                f"5m change {change_5m:+.1f}% — stalling "
                f"(want ≥ +{self.params.min_price_change_5m_pct:.0f}%)"
            )

        # Overextension — do not chase a parabolic move.
        if change_24h > self.params.max_price_change_24h_pct:
            failures.append(
                f"24h change {change_24h:+.0f}% > max +{self.params.max_price_change_24h_pct:.0f}% "
                "(parabolic — exhaustion risk)"
            )

        # Volume must be accelerating.
        if acceleration < self.params.min_volume_acceleration:
            failures.append(
                f"Vol accel {acceleration:.2f}x < min {self.params.min_volume_acceleration:.1f}x "
                "(volume fading)"
            )

        # Absolute participation floor.
        if vol_1h < self.params.min_volume_1h_usd:
            failures.append(f"Vol(1h) ${vol_1h:,.0f} < min ${self.params.min_volume_1h_usd:,.0f}")

        # Buy-side dominance.
        if buy_pressure < self.params.min_buy_pressure_5m:
            failures.append(
                f"Buy pressure {buy_pressure:.2f} < min {self.params.min_buy_pressure_5m:.2f}"
            )

        # Liquidity for exit.
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < min ${self.params.min_liquidity_usd:,.0f}")

        if failures:
            logger.debug(f"Momentum reject {token_address[:8]}...: {' | '.join(failures)}")
            return None

        signal = MomentumSignal(
            token_address=token_address,
            pair_address=pair_address,
            age_minutes=age_minutes,
            market_cap_usd=mcap,
            price_usd=price_usd,
            liquidity_usd=liq,
            volume_5m_usd=vol_5m,
            volume_1h_usd=vol_1h,
            volume_acceleration=acceleration,
            buy_pressure_5m=buy_pressure,
            price_change_5m_pct=change_5m,
            price_change_1h_pct=change_1h,
            price_change_24h_pct=change_24h,
            metadata={
                "strategy": self.strategy_id,
                "stop_loss_pct": self.params.stop_loss_pct,
                "take_profit_pct": self.params.take_profit_pct,
                "trailing_stop_pct": self.params.trailing_stop_pct,
                "max_hold_hours": self.params.max_hold_hours,
                "ai_min_confidence": self.params.ai_min_confidence,
            },
        )

        logger.info(
            f"Momentum SIGNAL {token_address[:8]}... | "
            f"age={age_minutes:.0f}m 1h={change_1h:+.1f}% 5m={change_5m:+.1f}% "
            f"accel={acceleration:.2f}x buyp={buy_pressure:.2f} "
            f"momentum={signal.momentum_score:.2f}"
        )
        return signal

    def build_ai_context(self, signal: MomentumSignal) -> str:
        """Per-signal context injected into the AI prompt for this candidate."""
        return "\n".join(
            [
                "=== MOMENTUM CONTINUATION EVALUATION ===",
                f"Strategy: {self.display_name}",
                f"Token: {signal.token_address}",
                f"Token age: {signal.age_minutes:.0f} minutes",
                f"Market cap: ${signal.market_cap_usd:,.0f}",
                f"Liquidity: ${signal.liquidity_usd:,.0f}",
                f"5-min price change: {signal.price_change_5m_pct:+.1f}%",
                f"1-hour price change: {signal.price_change_1h_pct:+.1f}% "
                f"({'STRONG' if signal.price_change_1h_pct > 30 else 'MODERATE'} trend)",
                f"24-hour price change: {signal.price_change_24h_pct:+.0f}%",
                f"Volume (5m/1h): ${signal.volume_5m_usd:,.0f} / ${signal.volume_1h_usd:,.0f}",
                f"Volume acceleration: {signal.volume_acceleration:.2f}x hourly pace "
                f"({'building' if signal.volume_acceleration >= 1.5 else 'steady'})",
                f"Buy pressure (5m): {signal.buy_pressure_5m:.2f}",
                f"Momentum score: {signal.momentum_score:.2f}/1.00",
                "",
                "EXIT PLAN (trend trade):",
                f"  Take profit: +{signal.metadata['take_profit_pct']:.0f}%",
                f"  Trailing stop: {signal.metadata['trailing_stop_pct']:.0f}%",
                f"  Hard stop: -{signal.metadata['stop_loss_pct']:.0f}%",
                f"  Max hold: {signal.metadata['max_hold_hours']:.0f} hours",
                "",
                "MOMENTUM RISK FACTORS:",
                "  - Trend-following buys strength; a sharp reversal can hit before the trail reacts",
                "  - A parabolic 24h run may be nearing exhaustion, not continuation",
                "  - Volume acceleration from a few large wallets can signal distribution",
                "  - Solana network congestion can stall the exit on a fast reversal",
                "",
                "DECISION: BUY or SKIP with confidence 0.0-1.0.",
                "Only ride momentum that is still building — skip exhaustion moves.",
            ]
        )
