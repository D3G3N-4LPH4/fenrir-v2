#!/usr/bin/env python3
"""
FENRIR - Budget Tracker & Capability Gates

Enforces per-strategy SOL budgets and position limits.
Acts as the last safety check before any trade execution.

The TradingEngine calls the budget tracker before signing
any transaction. If the strategy is over budget, over its
position limit, or paused — the trade is blocked.

Usage:
    tracker = BudgetTracker()

    # Before buying:
    allowed, reason = tracker.authorize_trade(strategy, amount_sol)
    if not allowed:
        logger.warning(f"Trade blocked: {reason}")
        return False

    # After buying:
    tracker.record_buy(strategy.strategy_id, amount_sol)

    # After selling:
    tracker.record_sell(strategy.strategy_id, returned_sol, pnl_pct)

    # Daily reset (midnight or bot restart):
    tracker.reset_all()
"""

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StrategyBudgetState:
    """Budget tracking state for a single strategy."""

    sol_spent: float = 0.0
    sol_returned: float = 0.0
    trades_executed: int = 0
    positions_open: int = 0
    wins: int = 0
    losses: int = 0
    last_trade_time: datetime | None = None

    @property
    def net_spent(self) -> float:
        return self.sol_spent - self.sol_returned

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0


@dataclass
class TradeAuthorization:
    """Result of an authorization check."""

    allowed: bool
    reason: str = ""
    strategy_id: str = ""
    amount_sol: float = 0.0


class BudgetTracker:
    """
    Central budget enforcement for all strategies.

    Every trade goes through authorize_trade() before execution.
    This is the final safety gate — even if the AI says BUY and
    the strategy says GO, the budget tracker can say NO.
    """

    def __init__(self):
        self._states: dict[str, StrategyBudgetState] = {}
        self._global_sol_limit: float | None = None  # Optional global cap
        self._global_sol_spent: float = 0.0
        self._reset_time: datetime = datetime.now()

    def set_global_limit(self, max_sol: float) -> None:
        """Set an absolute global SOL limit across all strategies."""
        self._global_sol_limit = max_sol

    def _get_state(self, strategy_id: str) -> StrategyBudgetState:
        """Get or create budget state for a strategy."""
        if strategy_id not in self._states:
            self._states[strategy_id] = StrategyBudgetState()
        return self._states[strategy_id]

    def authorize_trade(
        self,
        strategy_id: str,
        amount_sol: float,
        budget_sol: float,
        max_positions: int,
        is_active: bool = True,
        is_paused: bool = False,
    ) -> TradeAuthorization:
        """
        Check if a trade is authorized.

        Args:
            strategy_id: Which strategy is requesting
            amount_sol: How much SOL this trade would spend
            budget_sol: The strategy's daily budget
            max_positions: Max concurrent positions for this strategy
            is_active: Whether the strategy is active
            is_paused: Whether the strategy is paused

        Returns:
            TradeAuthorization with allowed=True/False and reason
        """
        auth = TradeAuthorization(
            allowed=False,
            strategy_id=strategy_id,
            amount_sol=amount_sol,
        )

        # Check strategy is active
        if not is_active:
            auth.reason = f"Strategy '{strategy_id}' is deactivated"
            return auth

        if is_paused:
            auth.reason = f"Strategy '{strategy_id}' is paused"
            return auth

        state = self._get_state(strategy_id)

        # Check position limit
        if state.positions_open >= max_positions:
            auth.reason = (
                f"Strategy '{strategy_id}' at position limit "
                f"({state.positions_open}/{max_positions})"
            )
            return auth

        # Check strategy budget
        if state.sol_spent + amount_sol > budget_sol:
            auth.reason = (
                f"Strategy '{strategy_id}' over budget: "
                f"{state.sol_spent:.4f} + {amount_sol:.4f} > {budget_sol:.4f} SOL"
            )
            return auth

        # Check global limit
        if self._global_sol_limit is not None:
            if self._global_sol_spent + amount_sol > self._global_sol_limit:
                auth.reason = (
                    f"Global SOL limit reached: "
                    f"{self._global_sol_spent:.4f} + {amount_sol:.4f} > "
                    f"{self._global_sol_limit:.4f} SOL"
                )
                return auth

        auth.allowed = True
        auth.reason = "authorized"
        return auth

    def record_buy(self, strategy_id: str, amount_sol: float) -> None:
        """Record a successful buy."""
        state = self._get_state(strategy_id)
        state.sol_spent += amount_sol
        state.trades_executed += 1
        state.positions_open += 1
        state.last_trade_time = datetime.now()
        self._global_sol_spent += amount_sol

    def record_sell(
        self,
        strategy_id: str,
        returned_sol: float,
        pnl_pct: float,
    ) -> None:
        """Record a sell (position close)."""
        state = self._get_state(strategy_id)
        state.sol_returned += returned_sol
        state.positions_open = max(0, state.positions_open - 1)

        if pnl_pct > 0:
            state.wins += 1
        else:
            state.losses += 1

    def reset_strategy(self, strategy_id: str) -> None:
        """Reset daily counters for a single strategy."""
        if strategy_id in self._states:
            state = self._states[strategy_id]
            state.sol_spent = 0.0
            state.sol_returned = 0.0
            state.trades_executed = 0
            state.wins = 0
            state.losses = 0
            # Note: positions_open is NOT reset (still real)

    def reset_all(self) -> None:
        """Reset daily counters for all strategies."""
        for strategy_id in list(self._states.keys()):
            self.reset_strategy(strategy_id)
        self._global_sol_spent = 0.0
        self._reset_time = datetime.now()

    def get_strategy_budget_status(self, strategy_id: str, budget_sol: float) -> dict:
        """Get budget status for a specific strategy."""
        state = self._get_state(strategy_id)
        return {
            "strategy_id": strategy_id,
            "budget_sol": budget_sol,
            "sol_spent": state.sol_spent,
            "sol_returned": state.sol_returned,
            "net_spent": state.net_spent,
            "budget_remaining": max(0, budget_sol - state.sol_spent),
            "budget_utilization": state.sol_spent / budget_sol if budget_sol > 0 else 0,
            "positions_open": state.positions_open,
            "trades_executed": state.trades_executed,
            "win_rate": state.win_rate,
            "last_trade": state.last_trade_time.isoformat() if state.last_trade_time else None,
        }

    def get_global_status(self) -> dict:
        """Get global budget status across all strategies."""
        total_spent = sum(s.sol_spent for s in self._states.values())
        total_returned = sum(s.sol_returned for s in self._states.values())
        total_positions = sum(s.positions_open for s in self._states.values())
        total_trades = sum(s.trades_executed for s in self._states.values())

        return {
            "global_sol_limit": self._global_sol_limit,
            "global_sol_spent": self._global_sol_spent,
            "total_sol_spent": total_spent,
            "total_sol_returned": total_returned,
            "net_spent": total_spent - total_returned,
            "total_positions_open": total_positions,
            "total_trades": total_trades,
            "strategies_tracked": len(self._states),
            "last_reset": self._reset_time.isoformat(),
        }
