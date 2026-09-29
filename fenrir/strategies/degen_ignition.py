#!/usr/bin/env python3
"""
FENRIR Strategy: Degen Ignition

Catches a launch in its first hour when buy pressure detonates — the moment a
trench coin goes from "just created" to "everyone is aping". The thesis: the
earliest sustained buy imbalance on a fresh pair precedes the first leg up, and
entering inside the first 45 minutes captures the steepest part of the curve.

Entry logic:
  - Age: ≤ 60 minutes (ideally < 30)
  - Tiny but real: mcap $500–$50k, liquidity ≥ $1k
  - Ignition: 1h buy/sell ratio ≥ 2.0 (or all-buys), ≥10 buys in the last hour
  - Tape is live: ≥8% of 24h volume printed in the last hour
  - Not already dead: 5m price change ≥ -10% (not dumping into the bid)

Exit logic:
  - Take profit: +150% (degen runner — let it work)
  - Trailing stop: 25% — trench coins retrace violently; give it room but
    cut when the ignition fizzles
  - Hard stop: -30%
  - Max hold: 90 minutes — ignition either works fast or it doesn't

Risk: EXTREME — most trench launches go to zero. The buy-edge and live-tape
gates filter for genuine demand, but this is lottery-ticket sizing by design.

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

logger = logging.getLogger("FENRIR.DegenIgnition")


@dataclass
class DegenIgnitionConfig:
    """Tunable parameters for the degen ignition strategy."""

    max_age_minutes: float = 60.0
    min_market_cap_usd: float = 500.0
    max_market_cap_usd: float = 50_000.0
    min_liquidity_usd: float = 1_000.0
    # Ignition: buy-side detonation over the last hour.
    min_buy_sell_ratio_1h: float = 2.0
    min_buys_1h: int = 10
    # Tape must be live: share of 24h volume in the last hour.
    min_volume_1h_share: float = 0.08
    # Not already dumping into the bid.
    min_price_change_5m_pct: float = -10.0
    # Exit plan.
    take_profit_pct: float = 150.0
    trailing_stop_pct: float = 25.0
    stop_loss_pct: float = 30.0
    max_hold_minutes: float = 90.0
    ai_min_confidence: float = 0.50
    daily_budget_sol: float = 0.0


@dataclass
class DegenIgnitionSignal:
    """Signal for a trench launch whose buy pressure just detonated."""

    token_address: str
    pair_address: str
    age_minutes: float
    market_cap_usd: float
    price_usd: float
    liquidity_usd: float
    buys_1h: int
    sells_1h: int
    buy_sell_ratio_1h: float
    volume_1h_share: float
    price_change_5m_pct: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ignition_score(self) -> float:
        """0-1 conviction: buy-edge strength, tape heat, and freshness."""
        edge = self.buy_sell_ratio_1h
        edge_score = min(1.0, max(0.0, (min(edge, 10.0) - 2.0) / 8.0))
        heat_score = min(1.0, max(0.0, (self.volume_1h_share - 0.08) / 0.42))
        fresh_score = min(1.0, max(0.0, (60.0 - self.age_minutes) / 60.0))
        return (edge_score * 0.5) + (heat_score * 0.3) + (fresh_score * 0.2)


class DegenIgnitionStrategy(TradingStrategy):
    """
    Degen ignition: enter a minutes-old launch the moment buy pressure
    detonates (1h buy/sell ≥ 2x on a live tape), riding the first leg with a
    wide trail and a hard time stop.
    """

    strategy_id = "degen_ignition"
    display_name = "Degen Ignition"
    description = (
        "Apes a minutes-old trench launch when 1h buy pressure detonates "
        "(≥2x buy/sell on a live tape), targeting the first leg up with a "
        "+150% take-profit and a 90-minute time stop. EXTREME risk."
    )

    budget_sol = 0.5
    max_concurrent_positions = 5
    uses_market_data = True

    def __init__(self, config: BotConfig, params: DegenIgnitionConfig | None = None) -> None:
        super().__init__()
        self.config = config
        self.params = params or DegenIgnitionConfig()
        self._params = TradeParams(
            buy_amount_sol=config.buy_amount_sol,
            max_slippage_bps=config.max_slippage_bps,
            stop_loss_pct=self.params.stop_loss_pct,
            take_profit_pct=self.params.take_profit_pct,
            trailing_stop_pct=self.params.trailing_stop_pct,
            max_position_age_minutes=int(self.params.max_hold_minutes),
            priority_fee_lamports=config.priority_fee_lamports,
            ai_min_confidence=self.params.ai_min_confidence,
            ai_temperature=config.ai_temperature,
            ai_entry_timeout=config.ai_entry_timeout_seconds,
        )

    async def should_evaluate(self, token_data: dict) -> bool:
        """Ignition needs a market snapshot; real gating is in evaluate_token."""
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: DEGEN IGNITION\n"
            "You are evaluating a minutes-old trench launch whose buy pressure "
            "just detonated — entered to catch the first leg up, not a trend.\n"
            "Key factors:\n"
            "- The ignition must be real: sustained buy imbalance (1h buy/sell ≥ 2x), "
            "not one wallet market-buying\n"
            "- The tape must be live: meaningful share of volume in the last hour\n"
            "- Red flags: dev still holding a large bag, snipers/bundlers dominating "
            "early supply, 5m already rolling over, liquidity too thin to exit\n"
            "- Green flags: broad buyer count, rising holder count, buy pressure "
            "building across consecutive windows\n"
            "- Time horizon: under 90 minutes. Ignition works fast or not at all — "
            "exit on target, trail, or time stop. Lottery-ticket sizing.\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> DegenIgnitionSignal | None:
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        age_minutes = getattr(market_data, "age_minutes", 0.0) or 0.0
        mcap = getattr(market_data, "market_cap_usd", 0.0) or 0.0
        price_usd = getattr(market_data, "price_usd", 0.0) or 0.0
        liq = getattr(market_data, "liquidity_usd", 0.0) or 0.0
        buys_1h = getattr(market_data, "txns_1h_buys", 0) or 0
        sells_1h = getattr(market_data, "txns_1h_sells", 0) or 0
        ratio = getattr(market_data, "buy_sell_ratio_1h", None)
        share = getattr(market_data, "volume_1h_share", None)
        change_5m = getattr(market_data, "price_change_5m_pct", 0.0) or 0.0
        pair_address = getattr(market_data, "pair_address", "") or ""

        failures = []
        if age_minutes > self.params.max_age_minutes:
            failures.append(f"age {age_minutes:.0f}m > {self.params.max_age_minutes:.0f}m")
        if not (self.params.min_market_cap_usd <= mcap <= self.params.max_market_cap_usd):
            failures.append(f"mcap ${mcap:,.0f} outside degen window")
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < ${self.params.min_liquidity_usd:,.0f}")
        if ratio is None:
            failures.append("1h buy/sell unavailable")
        elif ratio < self.params.min_buy_sell_ratio_1h:
            r = f"{ratio:.2f}" if ratio != float("inf") else "all-buys"
            failures.append(f"1h buy/sell {r} < {self.params.min_buy_sell_ratio_1h}x (no ignition)")
        if buys_1h < self.params.min_buys_1h:
            failures.append(f"1h buys {buys_1h} < {self.params.min_buys_1h}")
        if share is None:
            failures.append("1h volume share unavailable")
        elif share < self.params.min_volume_1h_share:
            failures.append(f"1h vol share {share:.1%} < {self.params.min_volume_1h_share:.0%}")
        if change_5m < self.params.min_price_change_5m_pct:
            failures.append(f"5m {change_5m:+.0f}% — dumping into the bid")

        if failures:
            logger.debug("DegenIgnition reject %s...: %s", token_address[:8], " | ".join(failures))
            return None

        signal = DegenIgnitionSignal(
            token_address=token_address,
            pair_address=pair_address,
            age_minutes=age_minutes,
            market_cap_usd=mcap,
            price_usd=price_usd,
            liquidity_usd=liq,
            buys_1h=buys_1h,
            sells_1h=sells_1h,
            buy_sell_ratio_1h=float(ratio),
            volume_1h_share=float(share),
            price_change_5m_pct=change_5m,
            metadata={
                "strategy": self.strategy_id,
                "stop_loss_pct": self.params.stop_loss_pct,
                "take_profit_pct": self.params.take_profit_pct,
                "trailing_stop_pct": self.params.trailing_stop_pct,
                "max_hold_minutes": self.params.max_hold_minutes,
                "ai_min_confidence": self.params.ai_min_confidence,
            },
        )
        logger.info(
            "DegenIgnition SIGNAL %s... | age=%.0fm mcap=$%.0f b/s=%.1f buys1h=%d",
            token_address[:8], age_minutes, mcap,
            float(ratio) if ratio != float("inf") else 999.0, buys_1h,
        )
        return signal
