#!/usr/bin/env python3
"""
FENRIR Strategy: Volume-to-Market Cap Anomaly (Day Trading)

Finds mid-cap meme coins ($500k-$3M) where 24h volume significantly
exceeds market cap — a signal of intense interest and high volatility
suitable for quick 30-50% scalps.

Entry logic:
  - Market cap: $500k to $3M
  - Token age: > 6 hours (survived initial dump)
  - 24h volume / market cap ratio > 1.5 (volume exceeds mcap by 50%+)
  - Buy on minor pullback: 5m price change negative (dip entry)
  - Sufficient liquidity for clean exit

Exit logic:
  - Target: 30-50% profit (scalp, not a hold)
  - Trailing stop: 12% (tight — this is a volume scalp)
  - Max hold: 4 hours

Risk: MODERATE — high liquidity reduces slippage risk but sudden
Solana network congestion can stall exits.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) while
retaining the richer ``evaluate_token`` / ``VolumeAnomalySignal`` machinery,
which gates on the DexScreener ``MarketData`` produced by ``fenrir.filters``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy

logger = logging.getLogger("FENRIR.VolumeAnomaly")


@dataclass
class VolumeAnomalyConfig:
    """Tunable parameters for the volume anomaly strategy."""

    # Token age minimum (hours) — must have survived initial dump
    min_age_hours: float = 6.0
    max_age_hours: float = 72.0
    # Market cap range
    min_market_cap_usd: float = 500_000.0
    max_market_cap_usd: float = 3_000_000.0
    # Volume/mcap ratio threshold
    min_volume_to_mcap_ratio: float = 1.5  # 24h vol > 1.5x market cap
    # Minimum 24h volume absolute
    min_volume_24h_usd: float = 500_000.0
    # Minimum liquidity for clean exit
    min_liquidity_usd: float = 100_000.0
    # Dip entry: 5m price change should be slightly negative
    max_entry_price_change_5m_pct: float = 0.0  # not pumping right now
    min_entry_price_change_5m_pct: float = -15.0  # not in freefall
    # Tight trailing stop for scalp
    trailing_stop_pct: float = 12.0
    # Take profit target
    take_profit_pct: float = 40.0
    # Hard stop-loss
    stop_loss_pct: float = 15.0
    # Max hold (hours)
    max_hold_hours: float = 4.0
    # AI confidence threshold
    ai_min_confidence: float = 0.60
    # Daily budget
    daily_budget_sol: float = 0.0


@dataclass
class VolumeAnomalySignal:
    """Signal for a volume anomaly scalp opportunity."""

    token_address: str
    pair_address: str
    age_hours: float
    market_cap_usd: float
    volume_24h_usd: float
    volume_to_mcap_ratio: float
    liquidity_usd: float
    price_usd: float
    price_change_5m_pct: float
    price_change_1h_pct: float
    price_change_24h_pct: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def anomaly_score(self) -> float:
        """
        0-1 score for how strong the volume anomaly is.
        Higher = more unusual volume relative to size.
        """
        ratio_score = min(1.0, (self.volume_to_mcap_ratio - 1.0) / 9.0)  # caps at 10x
        liq_score = min(1.0, self.liquidity_usd / 500_000.0)
        # Slight negative 5m = ideal dip entry
        entry_score = 1.0 if -10 <= self.price_change_5m_pct <= 0 else 0.5
        return (ratio_score * 0.5) + (liq_score * 0.3) + (entry_score * 0.2)


class VolumeAnomalyStrategy(TradingStrategy):
    """
    Volume-to-market-cap anomaly scalping strategy.

    Identifies mid-cap tokens with disproportionately high trading volume
    and enters on minor pullbacks for quick 30-50% scalps.
    """

    strategy_id = "volume_anomaly"
    display_name = "Volume Anomaly Scalper"
    description = (
        "Scalps mid-cap tokens ($500k-$3M) whose 24h volume exceeds market "
        "cap by 1.5x+, entering on a minor pullback for a quick 30-50% move. "
        "Tight trailing stop. MODERATE risk."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig) -> None:
        super().__init__()
        self.config = config
        self.params = VolumeAnomalyConfig()

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
        """
        Cheap pre-filter on token_data only. The volume anomaly depends on
        24h volume / market-cap ratio which requires market data, so the real
        gating happens in ``evaluate_token`` once a MarketData snapshot exists.
        """
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: VOLUME ANOMALY SCALP\n"
            "You are evaluating a mid-cap token whose 24h volume is unusually "
            "high relative to its market cap, entered on a minor pullback.\n"
            "Key factors for this strategy:\n"
            "- Target a quick 30-50% scalp — this is not a hold\n"
            "- A high volume/mcap ratio signals intense interest and volatility\n"
            "- Red flags: volume may be wash trading (check buy/sell balance), "
            "high ratio can indicate distribution rather than accumulation, "
            "mid-caps can gap down on broader market moves\n"
            "- Green flags: deep liquidity for clean exit, organic two-sided "
            "flow, a shallow dip entry rather than a chase\n"
            "- Time horizon: minutes to a few hours; exit on target or trail\n"
            "- Solana network congestion can stall exits — factor that in\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    # ── Rich signal machinery (used by the market-data stage) ──────────

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> VolumeAnomalySignal | None:
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        age_minutes = getattr(market_data, "age_minutes", 0.0)
        age_hours = age_minutes / 60.0
        mcap = getattr(market_data, "market_cap_usd", 0.0)
        vol_24h = getattr(market_data, "volume_24h_usd", 0.0)
        liq = getattr(market_data, "liquidity_usd", 0.0)
        price_usd = getattr(market_data, "price_usd", 0.0)
        change_5m = getattr(market_data, "price_change_5m_pct", 0.0)
        change_1h = getattr(market_data, "price_change_1h_pct", 0.0)
        change_24h = getattr(market_data, "price_change_24h_pct", 0.0)
        pair_address = getattr(market_data, "pair_address", "") or ""

        # Age gate — silent skip if outside window
        if not (self.params.min_age_hours <= age_hours <= self.params.max_age_hours):
            return None

        failures = []

        # Market cap range
        if mcap < self.params.min_market_cap_usd:
            failures.append(f"MCap ${mcap:,.0f} < min ${self.params.min_market_cap_usd:,.0f}")
        if mcap > self.params.max_market_cap_usd:
            failures.append(f"MCap ${mcap:,.0f} > max ${self.params.max_market_cap_usd:,.0f}")

        # Volume anomaly ratio
        ratio = vol_24h / mcap if mcap > 0 else 0.0
        if ratio < self.params.min_volume_to_mcap_ratio:
            failures.append(
                f"Vol/MCap ratio {ratio:.2f}x < min {self.params.min_volume_to_mcap_ratio:.1f}x"
            )

        # Absolute volume floor
        if vol_24h < self.params.min_volume_24h_usd:
            failures.append(
                f"Vol(24h) ${vol_24h:,.0f} < min ${self.params.min_volume_24h_usd:,.0f}"
            )

        # Liquidity for exit
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < min ${self.params.min_liquidity_usd:,.0f}")

        # Dip entry timing
        if change_5m > self.params.max_entry_price_change_5m_pct:
            failures.append(
                f"5m change {change_5m:+.1f}% — not a dip "
                f"(want ≤ {self.params.max_entry_price_change_5m_pct:.0f}%)"
            )
        if change_5m < self.params.min_entry_price_change_5m_pct:
            failures.append(
                f"5m change {change_5m:+.1f}% — freefall "
                f"(want ≥ {self.params.min_entry_price_change_5m_pct:.0f}%)"
            )

        if failures:
            logger.debug(f"Volume anomaly reject {token_address[:8]}...: {' | '.join(failures)}")
            return None

        signal = VolumeAnomalySignal(
            token_address=token_address,
            pair_address=pair_address,
            age_hours=age_hours,
            market_cap_usd=mcap,
            volume_24h_usd=vol_24h,
            volume_to_mcap_ratio=ratio,
            liquidity_usd=liq,
            price_usd=price_usd,
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
            f"Volume anomaly SIGNAL {token_address[:8]}... | "
            f"age={age_hours:.1f}h mcap=${mcap:,.0f} "
            f"vol/mcap={ratio:.1f}x vol24h=${vol_24h:,.0f} "
            f"5m={change_5m:+.1f}% anomaly={signal.anomaly_score:.2f}"
        )
        return signal

    def build_ai_context(self, signal: VolumeAnomalySignal) -> str:
        """Per-signal context injected into the AI prompt for this candidate."""
        return "\n".join(
            [
                "=== VOLUME ANOMALY SCALP EVALUATION ===",
                f"Strategy: {self.display_name}",
                f"Token: {signal.token_address}",
                f"Token age: {signal.age_hours:.1f} hours",
                f"Market cap: ${signal.market_cap_usd:,.0f}",
                f"24h volume: ${signal.volume_24h_usd:,.0f}",
                f"Volume/MCap ratio: {signal.volume_to_mcap_ratio:.2f}x "
                f"({'STRONG' if signal.volume_to_mcap_ratio > 3 else 'MODERATE'} anomaly)",
                f"Liquidity: ${signal.liquidity_usd:,.0f}",
                f"5-min price change: {signal.price_change_5m_pct:+.1f}% (dip entry)",
                f"1-hour price change: {signal.price_change_1h_pct:+.1f}%",
                f"24-hour price change: {signal.price_change_24h_pct:+.1f}%",
                f"Anomaly score: {signal.anomaly_score:.2f}/1.00",
                "",
                "EXIT PLAN (scalp):",
                f"  Take profit: +{signal.metadata['take_profit_pct']:.0f}%",
                f"  Trailing stop: {signal.metadata['trailing_stop_pct']:.0f}%",
                f"  Hard stop: -{signal.metadata['stop_loss_pct']:.0f}%",
                f"  Max hold: {signal.metadata['max_hold_hours']:.0f} hours",
                "",
                "SCALP RISK FACTORS:",
                "  - Solana network congestion risk on exit",
                "  - Volume may be wash trading (check buy/sell ratio)",
                "  - Mid-cap tokens can gap down on broader market moves",
                "  - High vol/mcap ratio sometimes indicates distribution, not accumulation",
                "",
                "DECISION: BUY or SKIP with confidence 0.0-1.0.",
                "Target quick 30-50% scalp only. High confidence required for entry.",
            ]
        )
