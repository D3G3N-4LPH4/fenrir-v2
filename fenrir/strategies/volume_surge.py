#!/usr/bin/env python3
"""
FENRIR Strategy: Volume Surge

The higher-cap volume trade. The thesis: when an established token ($2M–$25M
market cap) prints violent volume turnover — half its market cap or more
changing hands in a day — on buy-leaning flow with deep liquidity, the tape
can absorb real size and the repricing leg has room to run. This is the gap
between volatility_breakout (caps at $2M) and high_cap (rejects anything
printing +30%/1h): coins like the $3.6M-cap runners doing $3M–$10M daily
volume that neither filter will touch.

Entry logic:
  - Established: market cap $2M–$25M, age ≥ 12h (not a fresh launch)
  - The metric: 24h volume ≥ $2M AND 24h turnover (vol/mcap) 0.5x–20x.
    Below 0.5x the tape is too quiet for the thesis; above 20x is wash/churn.
  - Still alive: ≥5% of the day's volume printed in the last hour
  - Not distributing: 1h buy/sell ≥ 1.0 (parity — per-pair flow splits across
    venues, so no strong edge demanded) and 24h buys exceed sells
  - Exit exists: liquidity ≥ $150k
  - Not the terminal wick: 1h change ≤ +80% (no minimum — the token may be
    breaking out or basing into the flow; volume is the signal)

Exit logic:
  - Take profit: +60% from entry (higher caps extend less than trenches)
  - Trailing stop: 15% — volume surges reverse hard when the flow dries up
  - Hard stop: -15%
  - Max hold: 12 hours

Risk: HIGH — volume can be rented (wash, coordinated churn) and the edge
decays fast once turnover normalizes. The turnover band and the 1h-share
freshness gate exist to keep entries inside the active window; they don't
always succeed.

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

logger = logging.getLogger("FENRIR.VolumeSurge")


@dataclass
class VolumeSurgeConfig:
    """Tunable parameters for the volume surge strategy."""

    # Established band: above volatility_breakout's ceiling, below blue-chip.
    min_market_cap_usd: float = 2_000_000.0
    max_market_cap_usd: float = 25_000_000.0
    min_age_minutes: float = 12 * 60.0
    # The core metric: absolute + relative volume.
    min_volume_24h_usd: float = 2_000_000.0
    min_turnover_24h: float = 0.5
    max_turnover_24h: float = 20.0
    # Tape alive right now.
    min_volume_1h_share: float = 0.05
    # Not distributing.
    min_buy_sell_ratio_1h: float = 1.0
    require_buys_exceed_sells_24h: bool = True
    # Exit exists.
    min_liquidity_usd: float = 150_000.0
    # Heat is fine; the terminal wick is not.
    max_price_change_1h_pct: float = 80.0
    # Exit plan.
    take_profit_pct: float = 60.0
    trailing_stop_pct: float = 15.0
    stop_loss_pct: float = 15.0
    max_hold_hours: float = 12.0
    ai_min_confidence: float = 0.55
    daily_budget_sol: float = 0.0


@dataclass
class VolumeSurgeSignal:
    """Signal for an established token with violent buy-leaning volume turnover."""

    token_address: str
    pair_address: str
    age_minutes: float
    market_cap_usd: float
    price_usd: float
    liquidity_usd: float
    price_change_1h_pct: float
    turnover_24h: float
    volume_24h_usd: float
    volume_1h_share: float
    buy_sell_ratio_1h: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def surge_score(self) -> float:
        """0-1 conviction: turnover depth, buy-dominance, and volume heat."""
        turn_score = min(1.0, max(0.0, (self.turnover_24h - 0.5) / 4.5))
        edge = self.buy_sell_ratio_1h
        edge_score = min(1.0, max(0.0, (min(edge, 3.0) - 1.0) / 2.0))
        heat_score = min(1.0, max(0.0, (self.volume_1h_share - 0.05) / 0.45))
        return (turn_score * 0.45) + (edge_score * 0.25) + (heat_score * 0.30)


class VolumeSurgeStrategy(TradingStrategy):
    """
    Volume surge: enter an established ($2M–$25M) token printing violent
    volume turnover (≥0.5x daily) on non-sell-dominated flow, riding the
    repricing leg with a 15% trail.
    """

    strategy_id = "volume_surge"
    display_name = "Volume Surge"
    description = (
        "Trades established $2M–$25M tokens with violent volume turnover "
        "(≥50% of mcap traded daily) on buy-leaning flow — the higher-cap "
        "volume trade volatility_breakout can't reach and high_cap won't "
        "touch. +60% target, 15% trail. HIGH risk."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig, params: VolumeSurgeConfig | None = None) -> None:
        super().__init__()
        self.config = config
        self.params = params or VolumeSurgeConfig()
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
        """Surge needs a market snapshot; real gating is in evaluate_token."""
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: VOLUME SURGE\n"
            "You are evaluating an established token ($2M–$25M) printing "
            "violent volume turnover — half its market cap or more changed "
            "hands today. The tape can absorb real size here, unlike the "
            "trenches.\n"
            "Key factors:\n"
            "- Turnover is the signal: 0.5x–20x daily. Below 0.5x the thesis "
            "is gone; above 20x it is probably wash or coordinated churn.\n"
            "- Flow must not be sell-dominated: 1h buy/sell at parity or "
            "better, 24h buys exceeding sells. Heavy volume INTO selling is "
            "distribution, not accumulation.\n"
            "- The 1h share must stay elevated: when the fresh-volume share "
            "decays, the surge is over — exit, don't average.\n"
            "- Red flags: turnover concentrated in one venue/wallet, top "
            "holders distributing into the volume, 1h change beyond +80% "
            "(terminal wick)\n"
            "- Green flags: broadening buyer base across venues, liquidity "
            "deepening with the volume, higher lows on pullbacks\n"
            "- Time horizon: hours. Trail 15% — surges reverse as fast as "
            "they start when the flow dries up.\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> VolumeSurgeSignal | None:
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        age_minutes = getattr(market_data, "age_minutes", 0.0) or 0.0
        mcap = getattr(market_data, "market_cap_usd", 0.0) or 0.0
        price_usd = getattr(market_data, "price_usd", 0.0) or 0.0
        liq = getattr(market_data, "liquidity_usd", 0.0) or 0.0
        change_1h = getattr(market_data, "price_change_1h_pct", 0.0) or 0.0
        ratio = getattr(market_data, "buy_sell_ratio_1h", None)
        share = getattr(market_data, "volume_1h_share", None)
        vol_24h = getattr(market_data, "volume_24h_usd", 0.0) or 0.0
        buys_24h = getattr(market_data, "txns_24h_buys", 0) or 0
        sells_24h = getattr(market_data, "txns_24h_sells", 0) or 0
        pair_address = getattr(market_data, "pair_address", "") or ""

        turnover = (vol_24h / mcap) if mcap > 0 else 0.0

        failures = []
        if not (self.params.min_market_cap_usd <= mcap <= self.params.max_market_cap_usd):
            failures.append(
                f"mcap ${mcap:,.0f} outside "
                f"${self.params.min_market_cap_usd:,.0f}–${self.params.max_market_cap_usd:,.0f}"
            )
        if age_minutes < self.params.min_age_minutes:
            failures.append(f"age {age_minutes:.0f}m < {self.params.min_age_minutes:.0f}m")
        if vol_24h < self.params.min_volume_24h_usd:
            failures.append(f"Vol24h ${vol_24h:,.0f} < ${self.params.min_volume_24h_usd:,.0f}")
        if not (self.params.min_turnover_24h <= turnover <= self.params.max_turnover_24h):
            failures.append(
                f"turnover {turnover:.2f}x outside "
                f"{self.params.min_turnover_24h:.1f}x–{self.params.max_turnover_24h:.0f}x"
            )
        if ratio is None:
            failures.append("1h buy/sell unavailable")
        elif ratio < self.params.min_buy_sell_ratio_1h:
            r = f"{ratio:.2f}" if ratio != float("inf") else "all-buys"
            failures.append(
                f"1h buy/sell {r} < {self.params.min_buy_sell_ratio_1h}x (sell-dominated)"
            )
        if self.params.require_buys_exceed_sells_24h and buys_24h <= sells_24h:
            failures.append(f"24h buys {buys_24h} <= sells {sells_24h}")
        if share is None:
            failures.append("1h volume share unavailable")
        elif share < self.params.min_volume_1h_share:
            failures.append(f"1h vol share {share:.1%} < {self.params.min_volume_1h_share:.0%}")
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < ${self.params.min_liquidity_usd:,.0f}")
        if change_1h > self.params.max_price_change_1h_pct:
            failures.append(
                f"1h {change_1h:+.0f}% > +{self.params.max_price_change_1h_pct:.0f}% (terminal wick)"
            )

        if failures:
            logger.debug("VolumeSurge reject %s...: %s", token_address[:8], " | ".join(failures))
            return None
        assert ratio is not None and share is not None

        signal = VolumeSurgeSignal(
            token_address=token_address,
            pair_address=pair_address,
            age_minutes=age_minutes,
            market_cap_usd=mcap,
            price_usd=price_usd,
            liquidity_usd=liq,
            price_change_1h_pct=change_1h,
            turnover_24h=turnover,
            volume_24h_usd=vol_24h,
            volume_1h_share=float(share),
            buy_sell_ratio_1h=float(ratio),
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
            "VolumeSurge SIGNAL %s... | mcap=$%.1fM turn=%.1fx 1h=%+.0f%% b/s=%.2f",
            token_address[:8],
            mcap / 1e6,
            turnover,
            change_1h,
            float(ratio) if ratio != float("inf") else 999.0,
        )
        return signal
