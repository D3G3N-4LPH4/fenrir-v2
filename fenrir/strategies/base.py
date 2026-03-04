#!/usr/bin/env python3
"""
FENRIR - Strategy Base Class

Abstract interface for trading strategies. Each strategy defines:
- Entry criteria (which tokens to evaluate)
- AI context (strategy-specific instructions for Claude)
- Trade parameters (buy amount, slippage, priority fee)
- Budget and position limits

Strategies are self-contained units that can be activated, paused,
and run independently — inspired by OpenFang's "Hands" pattern.

Usage:
    class MySniperStrategy(TradingStrategy):
        strategy_id = "sniper"
        display_name = "Launch Sniper"
        ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TradeParams:
    """
    Strategy-specific trading parameters.

    These override BotConfig defaults on a per-strategy basis,
    allowing each strategy to have its own risk profile.
    """

    buy_amount_sol: float = 0.1
    max_slippage_bps: int = 500
    stop_loss_pct: float = 25.0
    take_profit_pct: float = 100.0
    trailing_stop_pct: float = 15.0
    max_position_age_minutes: int = 30
    priority_fee_lamports: int = 500_000

    # AI parameters for this strategy
    ai_min_confidence: float = 0.6
    ai_temperature: float = 0.3
    ai_entry_timeout: float = 5.0

    def to_dict(self) -> dict:
        return {
            "buy_amount_sol": self.buy_amount_sol,
            "max_slippage_bps": self.max_slippage_bps,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "max_position_age_minutes": self.max_position_age_minutes,
            "priority_fee_lamports": self.priority_fee_lamports,
            "ai_min_confidence": self.ai_min_confidence,
            "ai_temperature": self.ai_temperature,
        }


@dataclass
class StrategyState:
    """Runtime state for a strategy."""

    active: bool = True
    paused: bool = False
    positions_open: int = 0
    sol_spent_today: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0

    @property
    def win_rate_today(self) -> float:
        total = self.wins_today + self.losses_today
        return self.wins_today / total if total > 0 else 0.0


class TradingStrategy(ABC):
    """
    Abstract base class for all FENRIR trading strategies.

    Each strategy is a self-contained unit with its own:
    - Entry filter (should_evaluate)
    - AI context (get_ai_context)
    - Trade parameters (get_trade_params)
    - Budget and position limits
    """

    # ── Identity ──────────────────────────────────────────────
    strategy_id: str = "base"
    display_name: str = "Base Strategy"
    description: str = ""

    # ── Limits ────────────────────────────────────────────────
    budget_sol: float = 1.0                # Daily SOL budget
    max_concurrent_positions: int = 3       # Max open positions at once

    # ── State ─────────────────────────────────────────────────
    state: StrategyState = field(default_factory=StrategyState)

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses have required class attributes."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'strategy_id') or cls.strategy_id == "base":
            if cls.__name__ != "TradingStrategy":
                pass  # Allow abstract subclasses

    def __init__(self):
        self.state = StrategyState()

    @abstractmethod
    async def should_evaluate(self, token_data: dict) -> bool:
        """
        Pre-filter: does this token match this strategy's criteria?

        Called for every detected token before AI evaluation.
        Should be fast — no network calls, just data inspection.

        Args:
            token_data: Dict from PumpFunMonitor with token info

        Returns:
            True if the strategy wants to evaluate this token
        """

    @abstractmethod
    def get_ai_context(self) -> str:
        """
        Strategy-specific instructions injected into the AI prompt.

        This tells Claude HOW to evaluate tokens for this particular
        strategy. A sniper strategy emphasizes speed and fresh launches;
        a graduation strategy emphasizes migration progress.

        Returns:
            Text block to inject into the AI's system context
        """

    @abstractmethod
    def get_trade_params(self) -> TradeParams:
        """
        Strategy-specific trading parameters.

        Returns:
            TradeParams with this strategy's buy amount, slippage, etc.
        """

    def can_open_position(self) -> bool:
        """Check if this strategy can open another position."""
        if self.state.paused:
            return False
        if not self.state.active:
            return False
        if self.state.positions_open >= self.max_concurrent_positions:
            return False
        return True

    def can_spend(self, amount_sol: float) -> bool:
        """Check if budget allows this spend."""
        return (self.state.sol_spent_today + amount_sol) <= self.budget_sol

    def record_spend(self, amount_sol: float) -> None:
        """Record SOL spent by this strategy."""
        self.state.sol_spent_today += amount_sol
        self.state.trades_today += 1
        self.state.positions_open += 1

    def record_close(self, pnl_pct: float) -> None:
        """Record a position close."""
        self.state.positions_open = max(0, self.state.positions_open - 1)
        if pnl_pct > 0:
            self.state.wins_today += 1
        else:
            self.state.losses_today += 1

    def record_return(self, amount_sol: float) -> None:
        """Record SOL returned on a sell (reduces net spend)."""
        self.state.sol_spent_today = max(0, self.state.sol_spent_today - amount_sol)

    def reset_daily(self) -> None:
        """Reset daily counters (call at midnight or bot restart)."""
        self.state.sol_spent_today = 0.0
        self.state.trades_today = 0
        self.state.wins_today = 0
        self.state.losses_today = 0

    def pause(self) -> None:
        """Pause this strategy (no new entries, existing positions managed)."""
        self.state.paused = True

    def resume(self) -> None:
        """Resume this strategy."""
        self.state.paused = False

    def deactivate(self) -> None:
        """Fully deactivate (won't evaluate or manage positions)."""
        self.state.active = False

    def activate(self) -> None:
        """Activate this strategy."""
        self.state.active = True
        self.state.paused = False

    def get_status(self) -> dict:
        """Full status snapshot."""
        return {
            "strategy_id": self.strategy_id,
            "display_name": self.display_name,
            "active": self.state.active,
            "paused": self.state.paused,
            "budget_sol": self.budget_sol,
            "sol_spent_today": self.state.sol_spent_today,
            "budget_remaining": self.budget_sol - self.state.sol_spent_today,
            "positions_open": self.state.positions_open,
            "max_positions": self.max_concurrent_positions,
            "trades_today": self.state.trades_today,
            "win_rate_today": self.state.win_rate_today,
            "trade_params": self.get_trade_params().to_dict(),
        }
