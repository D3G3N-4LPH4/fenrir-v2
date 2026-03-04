#!/usr/bin/env python3
"""
FENRIR - Graduation Strategy

Buy tokens that are approaching Raydium migration (the "graduation").
When a pump.fun token raises ~85 SOL, it migrates to Raydium with
a DEX pool — this often triggers a second wave of buying as the token
becomes accessible to DEX aggregators and wider market.

This strategy targets tokens at 50-85% migration progress, betting
on the graduation pump. Higher liquidity = lower risk, but also
lower potential upside compared to fresh sniping.
"""

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy


class GraduationStrategy(TradingStrategy):
    """
    Migration play: buy tokens approaching Raydium graduation.

    Entry: bonding curve at 50-90% completion
    Exit: post-migration pump (take profit) or reversion (stop loss)
    Risk: lower than fresh sniping (token already has proven demand)
    """

    strategy_id = "graduation"
    display_name = "Graduation Play"
    description = (
        "Buys tokens approaching Raydium migration (50-90% bonding curve "
        "completion). Bets on the graduation pump when the token becomes "
        "available on DEX aggregators."
    )

    # Graduation plays are lower risk, allow more budget
    budget_sol = 2.0
    max_concurrent_positions = 4

    def __init__(self, config: BotConfig):
        super().__init__()
        self.config = config

        # Graduation plays use slightly different params:
        # - Tighter slippage (more liquidity available)
        # - Wider take profit (graduation pumps can be large)
        # - Shorter position age (graduation happens fast or not at all)
        self._params = TradeParams(
            buy_amount_sol=config.buy_amount_sol * 1.5,  # Slightly larger positions
            max_slippage_bps=300,  # Tighter — more liquidity at this stage
            stop_loss_pct=20.0,
            take_profit_pct=150.0,  # Graduation pumps can be significant
            trailing_stop_pct=12.0,
            max_position_age_minutes=20,  # Graduation is quick or doesn't happen
            priority_fee_lamports=config.priority_fee_lamports,
            ai_min_confidence=0.55,  # Lower bar — these are inherently safer
            ai_temperature=0.3,
            ai_entry_timeout=5.0,
        )

        # Migration progress thresholds
        self.min_migration_pct = 50.0
        self.max_migration_pct = 90.0
        self.min_liquidity_sol = 5.0  # Higher floor for graduation plays

    async def should_evaluate(self, token_data: dict) -> bool:
        """
        Only evaluate tokens that are 50-90% through the bonding curve.
        """
        curve_state = token_data.get("bonding_curve_state")
        if not curve_state:
            return False

        # Must not already be migrated
        if curve_state.complete:
            return False

        # Check migration progress is in our sweet spot
        progress = curve_state.get_migration_progress()
        if progress < self.min_migration_pct or progress > self.max_migration_pct:
            return False

        # Require minimum liquidity (these are established tokens)
        liq = token_data.get("initial_liquidity_sol", 0)
        if liq < self.min_liquidity_sol:
            return False

        return True

    def get_ai_context(self) -> str:
        """Context for AI: evaluate as a graduation/migration play."""
        return (
            "# STRATEGY CONTEXT: GRADUATION PLAY\n"
            "You are evaluating a token that is approaching Raydium migration.\n"
            f"The bonding curve is {self.min_migration_pct:.0f}-{self.max_migration_pct:.0f}% "
            "complete, meaning this token has already attracted significant buying.\n\n"
            "Key factors for this strategy:\n"
            "- Momentum analysis: is buying velocity increasing or slowing?\n"
            "- The graduation thesis: when a token migrates to Raydium, it becomes "
            "accessible to Jupiter aggregators and wider market, often triggering "
            "a second wave of buying\n"
            "- Risk is LOWER than fresh sniping: this token has proven demand\n"
            "- Focus on: migration progress rate, remaining SOL to graduation, "
            "community engagement, whether token has genuine staying power\n"
            "- Red flags: stalled progress (stuck at same % for a while), "
            "single-whale driven, no community presence\n"
            "- Green flags: accelerating progress, active social channels, "
            "organic holder distribution, original concept with meme potential\n"
            "- Position sizing: can be slightly larger than sniping (lower risk)\n"
            "- Time horizon: minutes (graduation should happen quickly)\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params
