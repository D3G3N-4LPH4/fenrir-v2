#!/usr/bin/env python3
"""
FENRIR Strategy: Volatility Breakout

Buys the vertical move that every other strategy refuses to chase. The thesis:
when a token prints +40% or more in an hour on expanding, buy-driven volume,
the move is often the START of repricing (listing, KOL wave, narrative catch)
rather than the end — and the crowd waiting for a dip never gets filled.

Entry logic:
  - Breakout: 1h price change between +40% and +400% (beyond that it's one wick)
  - Still alive: 5m change ≥ 0% (not already dumping)
  - Buy-driven: 1h buy/sell ≥ 1.3 and 24h buys > sells (not a short-squeeze wick)
  - Volume confirms: ≥10% of 24h volume in the last hour, 24h volume ≥ $50k
  - Exit exists: liquidity ≥ $10k

Exit logic:
  - Take profit: +80% from entry (breakouts extend, but don't get greedy)
  - Trailing stop: 20% — breakouts reverse hard; lock the extension
  - Hard stop: -20%
  - Max hold: 4 hours

Risk: VERY HIGH — buying vertical candles means entering with the weakest
hands alongside you. The buy-driven and volume-confirmation gates exist to
separate repricing from one-print wicks; they don't always succeed.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) with
``evaluate_token`` gated on a market-data snapshot. Off by default (opt-in);
used read-only by the scout's playbook tagger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy

logger = logging.getLogger("FENRIR.VolatilityBreakout")


@dataclass
class VolatilityBreakoutConfig:
    """Tunable parameters for the volatility breakout strategy."""

    # Breakout window: the vertical move other strategies reject.
    min_price_change_1h_pct: float = 40.0
    max_price_change_1h_pct: float = 400.0
    # Still alive right now (not already dumping).
    min_price_change_5m_pct: float = 0.0
    # Buy-driven, not a wick.
    min_buy_sell_ratio_1h: float = 1.3
    require_buys_exceed_sells_24h: bool = True
    # Volume confirmation.
    min_volume_1h_share: float = 0.10
    min_volume_24h_usd: float = 50_000.0
    # Exit exists.
    min_liquidity_usd: float = 10_000.0
    max_age_minutes: float = 48 * 60.0
    # Exit plan.
    take_profit_pct: float = 80.0
    trailing_stop_pct: float = 20.0
    stop_loss_pct: float = 20.0
    max_hold_hours: float = 4.0
    ai_min_confidence: float = 0.55
    daily_budget_sol: float = 0.0


@dataclass
class VolatilityBreakoutSignal:
    """Signal for a buy-driven vertical breakout still in motion."""

    token_address: str
    pair_address: str
    age_minutes: float
    market_cap_usd: float
    price_usd: float
    liquidity_usd: float
    price_change_5m_pct: float
    price_change_1h_pct: float
    buy_sell_ratio_1h: float
    volume_1h_share: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def breakout_score(self) -> float:
        """0-1 conviction: move magnitude, buy-dominance, and volume heat."""
        move_score = min(1.0, max(0.0, (self.price_change_1h_pct - 40.0) / 160.0))
        edge = self.buy_sell_ratio_1h
        edge_score = min(1.0, max(0.0, (min(edge, 5.0) - 1.3) / 3.7))
        heat_score = min(1.0, max(0.0, (self.volume_1h_share - 0.10) / 0.40))
        return (move_score * 0.4) + (edge_score * 0.35) + (heat_score * 0.25)


class VolatilityBreakoutStrategy(TradingStrategy):
    """
    Volatility breakout: enter a +40%-to-+400% hourly vertical move while it is
    still buy-driven and volume-confirmed, riding the repricing leg with a
    tight trail.
    """

    strategy_id = "volatility_breakout"
    display_name = "Volatility Breakout"
    description = (
        "Buys the vertical 1h move (+40% to +400%) other strategies reject, "
        "provided it is buy-driven and volume-confirmed, riding repricing "
        "with a +80% target and 20% trail. VERY HIGH risk."
    )

    budget_sol = 0.75
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig, params: VolatilityBreakoutConfig | None = None) -> None:
        super().__init__()
        self.config = config
        self.params = params or VolatilityBreakoutConfig()
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

    async def should_evaluate(self, token_data: dict) -> bool:
        """Breakout needs a market snapshot; real gating is in evaluate_token."""
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: VOLATILITY BREAKOUT\n"
            "You are evaluating a token mid-vertical-move (+40% or more in the "
            "last hour) — entered to ride repricing, not to catch a dip.\n"
            "Key factors:\n"
            "- The move must be buy-driven: 1h buy/sell ≥ 1.3x with 24h buys "
            "exceeding sells. A wick on thin two-sided flow is a trap.\n"
            "- Volume must confirm: a real share of the day's volume printed "
            "during the move\n"
            "- The 5m must still be non-negative: entering a move that already "
            "rolled over is catching a falling knife\n"
            "- Red flags: single-wallet-driven volume, top holders distributing "
            "into strength, liquidity too thin for the size\n"
            "- Green flags: news/listing/KOL catalyst, broadening buyer base, "
            "higher lows forming on the pullbacks\n"
            "- Time horizon: hours. Trail tight — breakouts reverse as fast as "
            "they start. Take the extension, don't marry it.\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> VolatilityBreakoutSignal | None:
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        age_minutes = getattr(market_data, "age_minutes", 0.0) or 0.0
        mcap = getattr(market_data, "market_cap_usd", 0.0) or 0.0
        price_usd = getattr(market_data, "price_usd", 0.0) or 0.0
        liq = getattr(market_data, "liquidity_usd", 0.0) or 0.0
        change_5m = getattr(market_data, "price_change_5m_pct", 0.0) or 0.0
        change_1h = getattr(market_data, "price_change_1h_pct", 0.0) or 0.0
        ratio = getattr(market_data, "buy_sell_ratio_1h", None)
        share = getattr(market_data, "volume_1h_share", None)
        vol_24h = getattr(market_data, "volume_24h_usd", 0.0) or 0.0
        buys_24h = getattr(market_data, "txns_24h_buys", 0) or 0
        sells_24h = getattr(market_data, "txns_24h_sells", 0) or 0
        pair_address = getattr(market_data, "pair_address", "") or ""

        failures = []
        if not (
            self.params.min_price_change_1h_pct <= change_1h <= self.params.max_price_change_1h_pct
        ):
            failures.append(
                f"1h {change_1h:+.0f}% outside breakout window "
                f"(+{self.params.min_price_change_1h_pct:.0f}%..+{self.params.max_price_change_1h_pct:.0f}%)"
            )
        if change_5m < self.params.min_price_change_5m_pct:
            failures.append(f"5m {change_5m:+.1f}% — move already rolling over")
        if ratio is None:
            failures.append("1h buy/sell unavailable")
        elif ratio < self.params.min_buy_sell_ratio_1h:
            r = f"{ratio:.2f}" if ratio != float("inf") else "all-buys"
            failures.append(
                f"1h buy/sell {r} < {self.params.min_buy_sell_ratio_1h}x (not buy-driven)"
            )
        if self.params.require_buys_exceed_sells_24h and buys_24h <= sells_24h:
            failures.append(f"24h buys {buys_24h} <= sells {sells_24h}")
        if share is None:
            failures.append("1h volume share unavailable")
        elif share < self.params.min_volume_1h_share:
            failures.append(f"1h vol share {share:.1%} < {self.params.min_volume_1h_share:.0%}")
        if vol_24h < self.params.min_volume_24h_usd:
            failures.append(f"Vol24h ${vol_24h:,.0f} < ${self.params.min_volume_24h_usd:,.0f}")
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < ${self.params.min_liquidity_usd:,.0f}")
        if age_minutes > self.params.max_age_minutes:
            failures.append(f"age {age_minutes:.0f}m > {self.params.max_age_minutes:.0f}m")

        if failures:
            logger.debug(
                "VolatilityBreakout reject %s...: %s", token_address[:8], " | ".join(failures)
            )
            return None

        # The failure guard above returns when either is None; safe to coerce now.
        assert ratio is not None and share is not None

        signal = VolatilityBreakoutSignal(
            token_address=token_address,
            pair_address=pair_address,
            age_minutes=age_minutes,
            market_cap_usd=mcap,
            price_usd=price_usd,
            liquidity_usd=liq,
            price_change_5m_pct=change_5m,
            price_change_1h_pct=change_1h,
            buy_sell_ratio_1h=float(ratio),
            volume_1h_share=float(share),
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
            "VolatilityBreakout SIGNAL %s... | 1h=%+.0f%% 5m=%+.1f%% b/s=%.1f",
            token_address[:8],
            change_1h,
            change_5m,
            float(ratio) if ratio != float("inf") else 999.0,
        )
        return signal
