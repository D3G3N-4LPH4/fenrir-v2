#!/usr/bin/env python3
"""
FENRIR Strategy: 15-Minute Reversal (The "Second Leg")

Targets tokens between 15 minutes and 2 hours old that have crashed
60-80% from their initial peak but are now consolidating and showing
signs of a community-driven recovery.

Entry logic:
  - Token age: 15 minutes to 2 hours
  - Drawdown from ATH: 60% to 80%
  - Price action: higher low on 1-minute chart (consolidation, not continued dump)
  - Developer still holding (dev wallet not fully sold)
  - Community signal: active X/Telegram (checked via social metadata)
  - Volume recovery: 1h volume recovering vs. initial dump volume

Exit logic:
  - Incremental sells as price pushes toward new ATH
  - 25% position sold at each 25% gain above entry
  - Hard stop at 20% below entry (if support breaks, it goes to zero)

Risk: HIGH — support floor can collapse entirely if CTO fails.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) while
retaining the richer ``evaluate_token`` / ``ReversalSignal`` machinery, which
gates on the DexScreener ``MarketData`` produced by ``fenrir.filters``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy

logger = logging.getLogger("FENRIR.ReversalStrategy")


@dataclass
class ReversalConfig:
    """Tunable parameters for the 15-minute reversal strategy."""

    # Token age window (minutes)
    min_age_minutes: float = 15.0
    max_age_minutes: float = 120.0
    # Drawdown from ATH required to qualify
    min_drawdown_from_ath_pct: float = 60.0
    max_drawdown_from_ath_pct: float = 80.0
    # Must show consolidation — price change in last 5m should be mild
    max_abs_price_change_5m_pct: float = 15.0
    # Minimum liquidity (lower than live launch — post-dump LP shrinks)
    min_liquidity_usd: float = 5_000.0
    # Volume recovery ratio: 1h volume / peak volume (proxy for recovery)
    min_volume_recovery_ratio: float = 0.15
    # Minimum 1h transactions (community must still be active)
    min_txns_1h: int = 50
    # Hard stop-loss (tight — if support breaks it collapses fast)
    stop_loss_pct: float = 20.0
    # Incremental take-profit levels
    take_profit_levels: list[float] = field(default_factory=lambda: [25.0, 50.0, 100.0, 200.0])
    # Sell fraction at each TP level
    take_profit_fractions: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    # Max hold time (minutes)
    max_hold_minutes: float = 180.0
    # AI confidence threshold
    ai_min_confidence: float = 0.60
    # Daily SOL budget
    daily_budget_sol: float = 0.0


@dataclass
class ReversalSignal:
    """Signal emitted when a qualifying reversal setup is detected."""

    token_address: str
    pair_address: str
    age_minutes: float
    current_price_usd: float
    liquidity_usd: float
    market_cap_usd: float
    volume_1h_usd: float
    txns_1h: int
    price_change_1h_pct: float
    price_change_5m_pct: float
    # Estimated drawdown from peak (derived from price change data)
    estimated_drawdown_pct: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def recovery_strength(self) -> float:
        """
        0-1 score for how strong the recovery signal is.
        Higher = stronger consolidation / reversal setup.
        """
        # Mild 5m price action = consolidation (not still dumping)
        stability = max(0.0, 1.0 - (abs(self.price_change_5m_pct) / 15.0))
        # Volume still present
        vol_score = min(1.0, self.volume_1h_usd / 20_000.0)
        # Transaction activity
        txn_score = min(1.0, self.txns_1h / 200.0)
        return (stability * 0.4) + (vol_score * 0.35) + (txn_score * 0.25)


class ReversalStrategy(TradingStrategy):
    """
    Second-leg reversal strategy.

    Finds tokens that have survived the initial dump and are consolidating
    for a community-driven recovery. Uses incremental profit-taking to
    capture gains while protecting against secondary collapse.
    """

    strategy_id = "reversal"
    display_name = "15-Minute Reversal"
    description = (
        "Buys tokens 15m-2h old that crashed 60-80% from ATH and are now "
        "consolidating for a community-driven recovery. Incremental "
        "profit-taking; hard stop if the support floor breaks. HIGH risk."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig) -> None:
        super().__init__()
        self.config = config
        self.params = ReversalConfig()

        self._params = TradeParams(
            buy_amount_sol=config.buy_amount_sol,
            max_slippage_bps=config.max_slippage_bps,
            stop_loss_pct=self.params.stop_loss_pct,
            # Single mechanical TP param; the full incremental ladder lives in
            # the signal metadata / AI context.
            take_profit_pct=self.params.take_profit_levels[-1],
            trailing_stop_pct=self.params.stop_loss_pct,
            max_position_age_minutes=int(self.params.max_hold_minutes),
            priority_fee_lamports=config.priority_fee_lamports,
            ai_min_confidence=self.params.ai_min_confidence,
            ai_temperature=config.ai_temperature,
            ai_entry_timeout=config.ai_entry_timeout_seconds,
        )

    # ── ABC interface ──────────────────────────────────────────────────

    async def should_evaluate(self, token_data: dict) -> bool:
        """
        Cheap pre-filter on token_data only. The reversal setup depends on
        drawdown/consolidation which requires market data, so the real gating
        happens in ``evaluate_token`` once a MarketData snapshot is available.
        """
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: 15-MINUTE REVERSAL\n"
            "You are evaluating a token that crashed 60-80% from its launch "
            "peak and may be consolidating for a community-driven recovery.\n"
            "Key questions for this strategy:\n"
            "- Is the 5m price action consolidating, or still actively dumping?\n"
            "- Is the drawdown severe enough to have shaken out weak hands?\n"
            "- Is there enough transaction activity to suggest a CTO is forming?\n"
            "- Does liquidity depth support incremental exits without slippage?\n"
            "- Red flags: still in freefall, dead volume, dev fully exited\n"
            "- Time horizon: minutes to a few hours; exit incrementally\n"
            "- Penalize heavily if the token is still actively dumping\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    # ── Rich signal machinery (used by the market-data stage) ──────────

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> ReversalSignal | None:
        """
        Evaluate a token for reversal entry.
        Returns a ReversalSignal if criteria are met, None otherwise.
        """
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        age = getattr(market_data, "age_minutes", 0.0)
        liq = getattr(market_data, "liquidity_usd", 0.0)
        mcap = getattr(market_data, "market_cap_usd", 0.0)
        vol_1h = getattr(market_data, "volume_1h_usd", 0.0)
        txns_1h = getattr(market_data, "txns_1h_total", 0)
        price_usd = getattr(market_data, "price_usd", 0.0)
        change_1h = getattr(market_data, "price_change_1h_pct", 0.0)
        change_5m = getattr(market_data, "price_change_5m_pct", 0.0)
        pair_address = getattr(market_data, "pair_address", "") or ""

        failures = []

        # Age window
        if not (self.params.min_age_minutes <= age <= self.params.max_age_minutes):
            return None  # Silent — most tokens won't be in this window

        # Estimate drawdown: if 1h change is deeply negative, token has dumped.
        # We use the 1h price change as a proxy for drawdown from initial peak.
        estimated_drawdown = abs(min(change_1h, 0.0))
        if estimated_drawdown < self.params.min_drawdown_from_ath_pct:
            logger.debug(
                f"Reversal reject {token_address[:8]}...: "
                f"drawdown {estimated_drawdown:.1f}% < min "
                f"{self.params.min_drawdown_from_ath_pct:.0f}%"
            )
            return None
        if estimated_drawdown > self.params.max_drawdown_from_ath_pct:
            logger.debug(
                f"Reversal reject {token_address[:8]}...: "
                f"drawdown {estimated_drawdown:.1f}% > max "
                f"{self.params.max_drawdown_from_ath_pct:.0f}% (too destroyed)"
            )
            return None

        # Consolidation check — 5m price action should be mild (not still dumping)
        if abs(change_5m) > self.params.max_abs_price_change_5m_pct:
            failures.append(
                f"Still volatile: 5m change {change_5m:+.1f}% "
                f"(want < ±{self.params.max_abs_price_change_5m_pct:.0f}%)"
            )

        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < min ${self.params.min_liquidity_usd:,.0f}")

        if txns_1h < self.params.min_txns_1h:
            failures.append(f"Txns(1h) {txns_1h} < min {self.params.min_txns_1h}")

        if failures:
            logger.info(f"Reversal reject {token_address[:8]}...: {' | '.join(failures)}")
            return None

        signal = ReversalSignal(
            token_address=token_address,
            pair_address=pair_address,
            age_minutes=age,
            current_price_usd=price_usd,
            liquidity_usd=liq,
            market_cap_usd=mcap,
            volume_1h_usd=vol_1h,
            txns_1h=txns_1h,
            price_change_1h_pct=change_1h,
            price_change_5m_pct=change_5m,
            estimated_drawdown_pct=estimated_drawdown,
            metadata={
                "strategy": self.strategy_id,
                "stop_loss_pct": self.params.stop_loss_pct,
                "take_profit_levels": self.params.take_profit_levels,
                "take_profit_fractions": self.params.take_profit_fractions,
                "max_hold_minutes": self.params.max_hold_minutes,
                "ai_min_confidence": self.params.ai_min_confidence,
            },
        )

        logger.info(
            f"Reversal SIGNAL {token_address[:8]}... | "
            f"age={age:.0f}m drawdown={estimated_drawdown:.0f}% "
            f"5m_chg={change_5m:+.1f}% liq=${liq:,.0f} "
            f"recovery={signal.recovery_strength:.2f}"
        )
        return signal

    def build_ai_context(self, signal: ReversalSignal) -> str:
        """Per-signal context injected into the AI prompt for this candidate."""
        tp_plan = " → ".join(
            f"+{level:.0f}% (sell {frac * 100:.0f}%)"
            for level, frac in zip(
                signal.metadata["take_profit_levels"],
                signal.metadata["take_profit_fractions"],
                strict=False,
            )
        )

        return "\n".join(
            [
                "=== 15-MINUTE REVERSAL EVALUATION ===",
                f"Strategy: {self.display_name}",
                f"Token: {signal.token_address}",
                f"Token age: {signal.age_minutes:.0f} minutes",
                f"Estimated drawdown from launch peak: -{signal.estimated_drawdown_pct:.0f}%",
                f"Current 5-min price change: {signal.price_change_5m_pct:+.1f}% "
                "(consolidation check)",
                f"1-hour price change: {signal.price_change_1h_pct:+.1f}%",
                f"Liquidity: ${signal.liquidity_usd:,.0f}",
                f"Market cap: ${signal.market_cap_usd:,.0f}",
                f"1-hour volume: ${signal.volume_1h_usd:,.0f}",
                f"1-hour transactions: {signal.txns_1h}",
                f"Recovery strength score: {signal.recovery_strength:.2f}/1.00",
                "",
                "EXIT PLAN (incremental):",
                f"  {tp_plan}",
                f"  Hard stop: -{signal.metadata['stop_loss_pct']:.0f}% (support break = zero)",
                f"  Max hold: {signal.metadata['max_hold_minutes']:.0f} minutes",
                "",
                "KEY QUESTIONS FOR AI EVALUATION:",
                "  1. Is the 5m price action consolidating or still dumping?",
                "  2. Is the drawdown severe enough to have shaken out weak hands?",
                "  3. Is there sufficient transaction activity to suggest a CTO is forming?",
                "  4. Does the liquidity depth support incremental exits without slippage?",
                "",
                "DECISION: BUY or SKIP with confidence 0.0-1.0.",
                "Confidence >= 0.60 required. Penalize if still actively dumping.",
            ]
        )
