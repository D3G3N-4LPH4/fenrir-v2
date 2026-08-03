#!/usr/bin/env python3
"""
FENRIR Strategy: Mean Reversion (Oversold Bounce)

The conceptual inverse of the momentum strategy. Instead of buying strength, it
fades a sharp *oversold dislocation* on the bet that price reverts toward its
recent mean — but only once the drop is stabilizing. The cardinal rule is "don't
catch a falling knife": a token still in freefall is rejected; entry requires the
5-minute action to have flattened or turned up while buyers step back in.

Entry logic:
  - Age: an established token (1 h – 48 h) so there is a mean to revert to
  - Oversold: 1h change ≤ -15% (a real dislocation)…
  - …but not a collapse: 1h change ≥ -60% and 24h change ≥ -80% (skip dead/rugged
    tokens that will not bounce)
  - Stabilizing NOW: 5m change ≥ -3% (the knife has slowed, ideally turning up)
  - Buyers returning: 5m buy pressure ≥ 0.45 (sellers no longer overwhelming)
  - Enough 1h volume + liquidity for a clean bounce and exit

Exit logic:
  - Take profit: +30% (a bounce to the mean, not a new trend)
  - Trailing stop: 10%
  - Hard stop: -12% — if it keeps falling, the reversion thesis is simply wrong
  - Max hold: 2 hours

Risk: HIGH — buying weakness means the dislocation can deepen (the knife keeps
falling); the stabilization + buy-pressure gates and the collapse floor exist to
avoid entering an ongoing capitulation, and the tight hard stop caps the downside
when the bounce does not come.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) with the
richer ``evaluate_token`` / ``MeanReversionSignal`` machinery gated on the
DexScreener ``MarketData`` produced by ``fenrir.filters``. Off by default (opt-in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy

logger = logging.getLogger("FENRIR.MeanReversion")


@dataclass
class MeanReversionConfig:
    """Tunable parameters for the mean reversion strategy."""

    # Age window (minutes): established enough to have a mean to revert to.
    min_age_minutes: float = 60.0
    max_age_minutes: float = 2880.0  # 48 hours
    # Oversold trigger: the 1h drop must be at least this deep (a real dislocation).
    max_price_change_1h_pct: float = -15.0  # change_1h must be <= this
    # …but not a collapse — floors that reject dead/rugged tokens.
    min_price_change_1h_pct: float = -60.0  # change_1h must be >= this
    min_price_change_24h_pct: float = -80.0  # change_24h must be >= this
    # Stabilization: the last 5 minutes must not still be crashing.
    min_price_change_5m_pct: float = -3.0
    # Buyers returning (sellers no longer overwhelming).
    min_buy_pressure_5m: float = 0.45
    # Participation + exit liquidity floors.
    min_volume_1h_usd: float = 50_000.0
    min_liquidity_usd: float = 50_000.0
    # Exit plan.
    take_profit_pct: float = 30.0
    trailing_stop_pct: float = 10.0
    stop_loss_pct: float = 12.0
    max_hold_hours: float = 2.0
    # AI confidence threshold.
    ai_min_confidence: float = 0.62
    # Daily budget (0 = fall back to the shared per-strategy default).
    daily_budget_sol: float = 0.0


@dataclass
class MeanReversionSignal:
    """Signal for an oversold mean-reversion bounce opportunity."""

    token_address: str
    pair_address: str
    age_minutes: float
    market_cap_usd: float
    price_usd: float
    liquidity_usd: float
    volume_1h_usd: float
    buy_pressure_5m: float
    price_change_5m_pct: float
    price_change_1h_pct: float
    price_change_24h_pct: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reversion_score(self) -> float:
        """0-1 score for how attractive the bounce setup is.

        Blends dislocation depth (how oversold on 1h, normalized between -15% and
        -50%), stabilization (5m recovering, -3%→+5% mapped to 0→1), and buy-side
        recovery (0.45→0.75 pressure mapped to 0→1).
        """
        dislocation = min(1.0, max(0.0, (-self.price_change_1h_pct - 15.0) / 35.0))
        stabilization = min(1.0, max(0.0, (self.price_change_5m_pct + 3.0) / 8.0))
        pressure = min(1.0, max(0.0, (self.buy_pressure_5m - 0.45) / 0.30))
        return (dislocation * 0.40) + (stabilization * 0.35) + (pressure * 0.25)


class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion (oversold bounce) strategy.

    Enters established tokens that have dislocated sharply lower but are showing
    signs of stabilization and returning buyers, betting on a reversion toward the
    recent mean — with a tight hard stop for when the bounce does not come.
    """

    strategy_id = "mean_reversion"
    display_name = "Mean Reversion"
    description = (
        "Fades a sharp oversold dislocation (1h -15%+ but stabilizing, buyers "
        "returning) on an established token, betting on a bounce toward the mean "
        "for a ~30% trade with a tight stop. HIGH risk."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig) -> None:
        super().__init__()
        self.config = config
        self.params = MeanReversionConfig()

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
        """Cheap pre-filter on token_data only. The dislocation + stabilization gates
        need a MarketData snapshot, so the real gating happens in ``evaluate_token``."""
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: MEAN REVERSION (OVERSOLD BOUNCE)\n"
            "You are evaluating an established token that has dropped sharply and may "
            "be bottoming — entered to fade the dislocation, not to chase a trend.\n"
            "Key factors for this strategy:\n"
            "- The drop must be real (oversold) but stabilizing — the 5m action has "
            "flattened or turned up, not still crashing (never catch a falling knife)\n"
            "- Buyers should be returning (buy pressure recovering)\n"
            "- Red flags: a token that is dead or rugging rather than dislocated, a 5m "
            "still in freefall, thin liquidity, a 'bounce' that is just a lower high in "
            "a continued downtrend\n"
            "- Green flags: a clear capitulation wick with volume, sellers exhausting, "
            "the 5m turning positive, deep liquidity, an intact higher-timeframe base\n"
            "- Time horizon: minutes to ~2 hours; take the bounce to the mean and exit — "
            "if it keeps falling the thesis is wrong, respect the tight stop\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    # ── Rich signal machinery (used by the market-data stage) ──────────

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> MeanReversionSignal | None:
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
        buy_pressure = getattr(market_data, "buy_pressure_5m", 0.5)
        pair_address = getattr(market_data, "pair_address", "") or ""

        # Age gate — silent skip if outside window.
        if not (self.params.min_age_minutes <= age_minutes <= self.params.max_age_minutes):
            return None

        failures = []

        # Oversold dislocation — must have dropped enough on 1h…
        if change_1h > self.params.max_price_change_1h_pct:
            failures.append(
                f"1h change {change_1h:+.1f}% — not oversold "
                f"(want ≤ {self.params.max_price_change_1h_pct:.0f}%)"
            )
        # …but not a collapse (dead/rugged tokens do not bounce).
        if change_1h < self.params.min_price_change_1h_pct:
            failures.append(
                f"1h change {change_1h:+.0f}% < floor {self.params.min_price_change_1h_pct:.0f}% "
                "(collapse, not a dislocation)"
            )
        if change_24h < self.params.min_price_change_24h_pct:
            failures.append(
                f"24h change {change_24h:+.0f}% < floor "
                f"{self.params.min_price_change_24h_pct:.0f}% (dead token)"
            )

        # Stabilization — do not catch a falling knife.
        if change_5m < self.params.min_price_change_5m_pct:
            failures.append(
                f"5m change {change_5m:+.1f}% — still crashing "
                f"(want ≥ {self.params.min_price_change_5m_pct:.0f}%)"
            )

        # Buyers returning.
        if buy_pressure < self.params.min_buy_pressure_5m:
            failures.append(
                f"Buy pressure {buy_pressure:.2f} < min {self.params.min_buy_pressure_5m:.2f}"
            )

        # Participation + exit liquidity.
        if vol_1h < self.params.min_volume_1h_usd:
            failures.append(f"Vol(1h) ${vol_1h:,.0f} < min ${self.params.min_volume_1h_usd:,.0f}")
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < min ${self.params.min_liquidity_usd:,.0f}")

        if failures:
            logger.debug(f"Mean reversion reject {token_address[:8]}...: {' | '.join(failures)}")
            return None

        signal = MeanReversionSignal(
            token_address=token_address,
            pair_address=pair_address,
            age_minutes=age_minutes,
            market_cap_usd=mcap,
            price_usd=price_usd,
            liquidity_usd=liq,
            volume_1h_usd=vol_1h,
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
            f"Mean reversion SIGNAL {token_address[:8]}... | "
            f"age={age_minutes:.0f}m 1h={change_1h:+.1f}% 5m={change_5m:+.1f}% "
            f"buyp={buy_pressure:.2f} reversion={signal.reversion_score:.2f}"
        )
        return signal

    def build_ai_context(self, signal: MeanReversionSignal) -> str:
        """Per-signal context injected into the AI prompt for this candidate."""
        return "\n".join(
            [
                "=== MEAN REVERSION (OVERSOLD BOUNCE) EVALUATION ===",
                f"Strategy: {self.display_name}",
                f"Token: {signal.token_address}",
                f"Token age: {signal.age_minutes:.0f} minutes",
                f"Market cap: ${signal.market_cap_usd:,.0f}",
                f"Liquidity: ${signal.liquidity_usd:,.0f}",
                f"1-hour price change: {signal.price_change_1h_pct:+.1f}% "
                f"({'DEEP' if signal.price_change_1h_pct < -30 else 'MODERATE'} dislocation)",
                f"5-min price change: {signal.price_change_5m_pct:+.1f}% "
                f"({'turning up' if signal.price_change_5m_pct >= 0 else 'stabilizing'})",
                f"24-hour price change: {signal.price_change_24h_pct:+.1f}%",
                f"1h volume: ${signal.volume_1h_usd:,.0f}",
                f"Buy pressure (5m): {signal.buy_pressure_5m:.2f}",
                f"Reversion score: {signal.reversion_score:.2f}/1.00",
                "",
                "EXIT PLAN (bounce to mean):",
                f"  Take profit: +{signal.metadata['take_profit_pct']:.0f}%",
                f"  Trailing stop: {signal.metadata['trailing_stop_pct']:.0f}%",
                f"  Hard stop: -{signal.metadata['stop_loss_pct']:.0f}%",
                f"  Max hold: {signal.metadata['max_hold_hours']:.0f} hours",
                "",
                "MEAN-REVERSION RISK FACTORS:",
                "  - Buying weakness: the dislocation can deepen (the knife keeps falling)",
                "  - The 'bounce' may be a lower high in a continued downtrend",
                "  - A sharp drop can be a rug/exploit in progress, not an overreaction",
                "  - Respect the tight hard stop — the thesis is invalidated below it",
                "",
                "DECISION: BUY or SKIP with confidence 0.0-1.0.",
                "Only fade dislocations that are stabilizing — never a falling knife.",
            ]
        )
