#!/usr/bin/env python3
"""
FENRIR Strategy: Migration Snipe (Ultra-Early)

Targets tokens at the exact moment they graduate from the pump.fun bonding
curve and launch onto Raydium. Buys within 1-2 minutes of pool opening and
rides the initial migration pump.

Entry logic:
  - Token must be a confirmed pump.fun → Raydium migration
  - Pool must be < MAX_ENTRY_MINUTES old
  - Liquidity must clear the minimum threshold (migration bundlers inflate this)
  - Security hard-filters must pass (mint revoked, freeze revoked)
  - No dev bundle detected (top-10 holder check)

Exit logic:
  - Stage 1: Take profit at 2x (recover initial investment)
  - Stage 2: Let remainder ("moon bag") ride with a trailing stop
  - Hard stop-loss at 35% (migrations can dump immediately if bundled)

Risk: EXTREME — developer bundle dumps are common on migration.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) while
retaining the richer ``evaluate_token`` / ``MigrationSignal`` machinery, which
gates on the DexScreener ``MarketData`` produced by ``fenrir.filters``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy

logger = logging.getLogger("FENRIR.MigrationSnipe")

RAYDIUM_AMM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
PUMPFUN_MIGRATION_PROGRAM = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"

# DexScreener dexId values for the AMMs pump.fun tokens migrate to. Verified on
# real MigrateV2 txs: modern graduations land on PumpSwap (pump's own AMM); some
# legacy migrations land on Raydium. Compared case-insensitively.
SUPPORTED_MIGRATION_DEXES = frozenset({"raydium", "pumpswap"})


@dataclass
class MigrationSniperConfig:
    """Tunable parameters for the migration snipe strategy."""

    # Maximum age of Raydium pool at entry (minutes)
    max_entry_minutes: float = 2.0
    # Minimum liquidity in USD at migration
    min_liquidity_usd: float = 10_000.0
    # Maximum liquidity (very high = suspicious bundle)
    max_liquidity_usd: float = 500_000.0
    # Minimum market cap at migration
    min_market_cap_usd: float = 30_000.0
    # Maximum market cap at migration
    max_market_cap_usd: float = 150_000.0
    # Minimum 5-minute volume after migration
    min_volume_5m_usd: float = 15_000.0
    # Minimum transactions in first 5 minutes
    min_txns_5m: int = 100
    # Stage 1 take-profit (recover initial capital)
    take_profit_stage1_pct: float = 100.0  # 2x = 100% gain
    # Stage 1: sell this fraction of position
    stage1_sell_fraction: float = 0.5  # sell 50%, keep 50% as moon bag
    # Stage 2 trailing stop on moon bag
    moon_bag_trailing_stop_pct: float = 25.0
    # Hard stop-loss
    stop_loss_pct: float = 35.0
    # Max hold time (minutes) before forced exit
    max_hold_minutes: float = 60.0
    # AI minimum confidence to approve entry
    ai_min_confidence: float = 0.65
    # Daily SOL budget for this strategy
    daily_budget_sol: float = 0.0  # 0 = use global budget


@dataclass
class MigrationSignal:
    """Signal emitted when a qualifying migration is detected."""

    token_address: str
    pair_address: str
    pool_age_minutes: float
    liquidity_usd: float
    market_cap_usd: float
    volume_5m_usd: float
    txns_5m: int
    price_usd: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def urgency_score(self) -> float:
        """
        0-1 score reflecting how urgently to act.
        Higher = more urgent (younger pool, higher volume).
        """
        age_score = max(0.0, 1.0 - (self.pool_age_minutes / 2.0))
        vol_score = min(1.0, self.volume_5m_usd / 50_000.0)
        return (age_score * 0.7) + (vol_score * 0.3)


class MigrationSniperStrategy(TradingStrategy):
    """
    Ultra-early migration snipe strategy.

    Monitors for pump.fun → Raydium migrations and enters within the
    first 2 minutes. Uses a two-stage exit: recover capital at 2x,
    then trail the moon bag.
    """

    strategy_id = "migration_snipe"
    display_name = "Migration Sniper"
    description = (
        "Snipes pump.fun → Raydium migrations within the first ~2 minutes of "
        "the pool opening. Two-stage exit: recover capital at 2x, then trail "
        "the moon bag. EXTREME risk — dev bundle dumps are common."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3
    uses_market_data = True

    def __init__(self, config: BotConfig) -> None:
        super().__init__()
        self.config = config
        self.params = MigrationSniperConfig()

        self._params = TradeParams(
            buy_amount_sol=config.buy_amount_sol,
            max_slippage_bps=config.max_slippage_bps,
            stop_loss_pct=self.params.stop_loss_pct,
            take_profit_pct=self.params.take_profit_stage1_pct,
            trailing_stop_pct=self.params.moon_bag_trailing_stop_pct,
            max_position_age_minutes=int(self.params.max_hold_minutes),
            priority_fee_lamports=config.priority_fee_lamports,
            ai_min_confidence=self.params.ai_min_confidence,
            ai_temperature=config.ai_temperature,
            ai_entry_timeout=config.ai_entry_timeout_seconds,
        )

    # ── ABC interface ──────────────────────────────────────────────────

    async def should_evaluate(self, token_data: dict) -> bool:
        """
        Cheap pre-filter on token_data only (no market fetch).

        Migration snipes target tokens that have *completed* the bonding
        curve. If we can see the curve state and it is not yet complete,
        skip; otherwise let the market-data stage confirm the Raydium pool.
        """
        curve_state = token_data.get("bonding_curve_state")
        if curve_state is not None and not curve_state.complete:
            return False
        return True

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: MIGRATION SNIPE\n"
            "You are evaluating a token at the moment it graduates from the "
            "pump.fun bonding curve onto Raydium (ULTRA-EARLY, first ~2 min).\n"
            "Key factors for this strategy:\n"
            "- Speed matters: the pool just opened, decide fast\n"
            "- Exit plan: recover capital at 2x, trail the remaining moon bag\n"
            "- Red flags: developer bundle dumps (common on migration), bot "
            "wash-trading inflating volume, thin liquidity, suspiciously high LP\n"
            "- Green flags: organic buy pressure, healthy holder distribution, "
            "mint/freeze authority revoked\n"
            "- Time horizon: minutes; this is a scalp on the migration pump\n"
            "- Given EXTREME risk, require high confidence to enter\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    # ── Rich signal machinery (used by the market-data stage) ──────────

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> MigrationSignal | None:
        """
        Evaluate a token for migration snipe entry.
        Returns a MigrationSignal if criteria are met, None otherwise.
        """
        if not self.state.active:
            return None

        if market_data is None:
            logger.debug(f"No market data for {token_data.get('token_address', '?')[:8]}...")
            return None

        age = getattr(market_data, "age_minutes", float("inf"))
        liq = getattr(market_data, "liquidity_usd", 0.0)
        mcap = getattr(market_data, "market_cap_usd", 0.0)
        vol_5m = getattr(market_data, "volume_5m_usd", 0.0)
        txns_5m = getattr(market_data, "txns_5m_total", 0)
        price_usd = getattr(market_data, "price_usd", 0.0)
        pair_address = getattr(market_data, "pair_address", "") or ""
        dex_id = getattr(market_data, "dex_id", "") or ""

        token_address = token_data.get("token_address", "")
        failures = []

        # Pool age gate — must be ultra-fresh
        if age > self.params.max_entry_minutes:
            logger.debug(
                f"Migration reject {token_address[:8]}...: "
                f"pool too old ({age:.1f}m > {self.params.max_entry_minutes}m)"
            )
            return None

        # Must be on a supported migration AMM (PumpSwap or Raydium)
        if dex_id.lower() not in SUPPORTED_MIGRATION_DEXES:
            logger.debug(
                f"Migration reject {token_address[:8]}...: "
                f"unsupported migration AMM (dex={dex_id})"
            )
            return None

        # Liquidity checks
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < min ${self.params.min_liquidity_usd:,.0f}")
        if liq > self.params.max_liquidity_usd:
            failures.append(
                f"LP ${liq:,.0f} > max ${self.params.max_liquidity_usd:,.0f} (possible bundle)"
            )

        # Market cap checks
        if mcap < self.params.min_market_cap_usd:
            failures.append(f"MCap ${mcap:,.0f} < min ${self.params.min_market_cap_usd:,.0f}")
        if mcap > self.params.max_market_cap_usd:
            failures.append(f"MCap ${mcap:,.0f} > max ${self.params.max_market_cap_usd:,.0f}")

        # Volume momentum
        if vol_5m < self.params.min_volume_5m_usd:
            failures.append(f"Vol(5m) ${vol_5m:,.0f} < min ${self.params.min_volume_5m_usd:,.0f}")

        # Transaction count
        if txns_5m < self.params.min_txns_5m:
            failures.append(f"Txns(5m) {txns_5m} < min {self.params.min_txns_5m}")

        if failures:
            logger.info(f"Migration reject {token_address[:8]}...: {' | '.join(failures)}")
            return None

        signal = MigrationSignal(
            token_address=token_address,
            pair_address=pair_address,
            pool_age_minutes=age,
            liquidity_usd=liq,
            market_cap_usd=mcap,
            volume_5m_usd=vol_5m,
            txns_5m=txns_5m,
            price_usd=price_usd,
            metadata={
                "strategy": self.strategy_id,
                "stop_loss_pct": self.params.stop_loss_pct,
                "take_profit_pct": self.params.take_profit_stage1_pct,
                "trailing_stop_pct": self.params.moon_bag_trailing_stop_pct,
                "stage1_sell_fraction": self.params.stage1_sell_fraction,
                "max_hold_minutes": self.params.max_hold_minutes,
                "ai_min_confidence": self.params.ai_min_confidence,
            },
        )

        logger.info(
            f"Migration SIGNAL {token_address[:8]}... | "
            f"age={age:.1f}m liq=${liq:,.0f} mcap=${mcap:,.0f} "
            f"vol5m=${vol_5m:,.0f} txns={txns_5m} "
            f"urgency={signal.urgency_score:.2f}"
        )
        return signal

    def build_ai_context(
        self,
        signal: MigrationSignal,
        security_result: Any | None = None,
    ) -> str:
        """Per-signal context injected into the AI prompt for this candidate."""
        lines = [
            "=== MIGRATION SNIPE EVALUATION ===",
            f"Strategy: {self.display_name}",
            f"Token: {signal.token_address}",
            f"Pool age: {signal.pool_age_minutes:.1f} minutes (ULTRA-EARLY)",
            f"Liquidity: ${signal.liquidity_usd:,.0f}",
            f"Market cap: ${signal.market_cap_usd:,.0f}",
            f"5-min volume: ${signal.volume_5m_usd:,.0f}",
            f"5-min transactions: {signal.txns_5m}",
            f"Urgency score: {signal.urgency_score:.2f}/1.00",
            "",
            "EXIT PLAN:",
            f"  Stage 1: Sell {signal.metadata['stage1_sell_fraction'] * 100:.0f}% at "
            f"+{signal.metadata['take_profit_pct']:.0f}% (recover capital)",
            f"  Stage 2: Trail remaining moon bag with "
            f"{signal.metadata['trailing_stop_pct']:.0f}% trailing stop",
            f"  Stop loss: -{signal.metadata['stop_loss_pct']:.0f}%",
            f"  Max hold: {signal.metadata['max_hold_minutes']:.0f} minutes",
            "",
            "RISK FACTORS TO EVALUATE:",
            "  - Developer bundle dump risk (common on migration)",
            "  - Bot wash-trading inflating volume metrics",
            "  - Thin liquidity enabling large price impact on exit",
            "  - First-mover advantage vs. dev dump timing",
        ]

        if security_result:
            lines.append("")
            lines.append("SECURITY CHECKS:")
            if security_result.details.get("top10_holder_pct") is not None:
                lines.append(
                    f"  Top-10 holders: {security_result.details['top10_holder_pct']:.1f}% of supply"
                )
            if security_result.details.get("lp_burned_pct") is not None:
                lines.append(f"  LP burned: {security_result.details['lp_burned_pct']:.1f}%")

        lines.extend(
            [
                "",
                "DECISION REQUIRED:",
                "Respond with BUY or SKIP and a confidence score 0.0-1.0.",
                "Given EXTREME risk, confidence >= 0.65 required to enter.",
            ]
        )

        return "\n".join(lines)
