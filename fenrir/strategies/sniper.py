#!/usr/bin/env python3
"""
FENRIR - Sniper Strategy

The default FENRIR strategy: detect fresh pump.fun launches and
snipe them immediately via the bonding curve.

This extracts the existing bot logic into a self-contained strategy
unit. Entry criteria: fresh launch, meets liquidity/mcap filters,
not yet migrated to Raydium.
"""

from fenrir.config import TRADING_PRESETS, BotConfig, TradingMode
from fenrir.strategies.base import TradeParams, TradingStrategy


class SniperStrategy(TradingStrategy):
    """
    Fresh launch sniper.

    Evaluates every new token that passes the monitor's basic filters.
    Relies on the AI Brain for quality assessment. Fast timeout,
    bias toward action on high-confidence signals.
    """

    strategy_id = "sniper"
    display_name = "Launch Sniper"
    description = (
        "Snipes fresh pump.fun launches via direct bonding curve buys. "
        "Evaluates with AI for quality signals, exits on mechanical "
        "triggers with AI override capability."
    )

    def __init__(self, config: BotConfig):
        super().__init__()
        self.config = config

        # Apply mode presets
        preset = TRADING_PRESETS.get(config.mode, {})

        # Strategy-specific limits from config.
        # Prefer the explicit sniper_daily_budget_sol when set; fall back to
        # 10 × buy_amount_sol only as a last resort so that tuning buy size
        # doesn't silently balloon the daily spend cap.
        if config.sniper_daily_budget_sol > 0:
            self.budget_sol = config.sniper_daily_budget_sol
        else:
            self.budget_sol = config.buy_amount_sol * 10  # legacy auto default
        self.max_concurrent_positions = 5

        # Build trade params from config + presets
        self._params = TradeParams(
            buy_amount_sol=config.buy_amount_sol,
            max_slippage_bps=config.max_slippage_bps,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            max_position_age_minutes=config.max_position_age_minutes,
            priority_fee_lamports=config.priority_fee_lamports,
            ai_min_confidence=config.ai_min_confidence_to_buy,
            ai_temperature=preset.get("ai_temperature", config.ai_temperature),
            ai_entry_timeout=config.ai_entry_timeout_seconds,
        )

    async def should_evaluate(self, token_data: dict) -> bool:
        """
        Evaluate all fresh launches that passed the monitor's criteria.

        The monitor already filtered by min liquidity and max market cap.
        The sniper is interested in everything that survived those filters
        and hasn't already migrated.
        """
        # Skip if already migrated to Raydium
        curve_state = token_data.get("bonding_curve_state")
        if curve_state and curve_state.complete:
            return False

        # Skip if too far along (>50% migrated = price already moved)
        if curve_state and curve_state.get_migration_progress() > 50:
            return False

        return True

    def get_ai_context(self) -> str:
        """Context for AI: evaluate as a fresh launch sniper."""
        return (
            "# STRATEGY CONTEXT: LAUNCH SNIPER\n"
            "You are evaluating a FRESH pump.fun token launch for immediate entry.\n"
            "Key factors for this strategy:\n"
            "- Speed matters: this token just launched, you need to decide quickly\n"
            "- Focus on: token name/symbol quality, description legitimacy, "
            "initial liquidity depth, creator reputation\n"
            "- Red flags: copycat names, zero-effort descriptions, "
            "suspicious creator history, extremely low liquidity\n"
            "- Green flags: original concept, engaged community signals, "
            "healthy initial liquidity (3-20 SOL), known good creators\n"
            "- Position sizing: smaller for unknown creators, larger for "
            "high-conviction setups\n"
            "- Time horizon: minutes to hours, not days\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params


class ConservativeSniperStrategy(SniperStrategy):
    """
    Conservative variant: tighter filters, smaller positions.
    """

    strategy_id = "sniper_conservative"
    display_name = "Conservative Sniper"

    def __init__(self, config: BotConfig):
        override = BotConfig.from_mode(TradingMode.CONSERVATIVE)
        super().__init__(override)
        self.budget_sol = 0.5  # Smaller daily budget
        self.max_concurrent_positions = 2
        self._params.ai_min_confidence = 0.75  # Higher bar

    async def should_evaluate(self, token_data: dict) -> bool:
        if not await super().should_evaluate(token_data):
            return False

        # Additional filter: require higher liquidity
        liq = token_data.get("initial_liquidity_sol", 0)
        return liq >= 5.0


class DegenSniperStrategy(SniperStrategy):
    """
    Degen variant: wider filters, larger positions, lower confidence bar.
    """

    strategy_id = "sniper_degen"
    display_name = "Degen Sniper"

    def __init__(self, config: BotConfig):
        override = BotConfig.from_mode(TradingMode.DEGEN)
        super().__init__(override)
        self.budget_sol = 5.0
        self.max_concurrent_positions = 8
        self._params.ai_min_confidence = 0.4

    async def should_evaluate(self, token_data: dict) -> bool:
        # Degen evaluates everything that hasn't migrated
        curve_state = token_data.get("bonding_curve_state")
        if curve_state and curve_state.complete:
            return False
        return True
