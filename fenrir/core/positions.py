#!/usr/bin/env python3
"""
FENRIR - Position Tracking

Position management and portfolio tracking.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from fenrir.config import BotConfig
from fenrir.logger import FenrirLogger


@dataclass
class Position:
    """
    A single token position.
    Every number tells a story of risk and reward.
    """

    token_address: str
    entry_time: datetime
    entry_price: float  # Price per token in SOL
    amount_tokens: float
    amount_sol_invested: float
    peak_price: float  # For trailing stops
    current_price: float = 0.0

    def update_price(self, new_price: float):
        """Update current price and track peak."""
        self.current_price = new_price
        self.peak_price = max(self.peak_price, new_price)

    def get_pnl_percent(self) -> float:
        """Calculate profit/loss percentage."""
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    def get_pnl_sol(self) -> float:
        """Calculate profit/loss in SOL."""
        current_value = self.amount_tokens * self.current_price
        return current_value - self.amount_sol_invested

    def should_take_profit(self, target_pct: float) -> bool:
        """Greed is good, but profits are better."""
        return self.get_pnl_percent() >= target_pct

    def should_stop_loss(self, max_loss_pct: float) -> bool:
        """Cut losses before they cut you."""
        return self.get_pnl_percent() <= -max_loss_pct

    def should_trailing_stop(self, trailing_pct: float) -> bool:
        """Protect gains as they grow."""
        if self.peak_price == 0:
            return False
        drawdown = ((self.peak_price - self.current_price) / self.peak_price) * 100
        return drawdown >= trailing_pct

    def is_expired(self, max_age_minutes: int) -> bool:
        """Time-based exit - sometimes the party's over."""
        age = datetime.now() - self.entry_time
        return age > timedelta(minutes=max_age_minutes)


class PositionManager:
    """
    Portfolio manager extraordinaire.
    Tracks all positions with the precision of a Swiss watch.
    """

    def __init__(self, config: BotConfig, logger: FenrirLogger):
        self.config = config
        self.logger = logger
        self.positions: dict[str, Position] = {}

    def open_position(
        self, token_address: str, entry_price: float, amount_tokens: float, amount_sol: float
    ):
        """Open a new position with ceremony."""
        position = Position(
            token_address=token_address,
            entry_time=datetime.now(),
            entry_price=entry_price,
            amount_tokens=amount_tokens,
            amount_sol_invested=amount_sol,
            peak_price=entry_price,
            current_price=entry_price,
        )
        self.positions[token_address] = position
        self.logger.info(f"Position opened: {token_address[:8]}... | {amount_tokens:.2f} tokens")

    def close_position(self, token_address: str, reason: str) -> Position | None:
        """Close and return the position."""
        if token_address in self.positions:
            position = self.positions.pop(token_address)
            pnl = position.get_pnl_percent()
            self.logger.sell_executed(token_address, pnl, reason)
            return position
        return None

    def update_prices(self, prices: dict[str, float]):
        """Batch update all position prices."""
        for token_address, price in prices.items():
            if token_address in self.positions:
                self.positions[token_address].update_price(price)

    def check_exit_conditions(self) -> list[tuple[str, str]]:
        """
        Check all positions for exit signals.
        Returns list of (token_address, reason) tuples.
        """
        exits = []

        for token_address, position in self.positions.items():
            # Take profit
            if position.should_take_profit(self.config.take_profit_pct):
                exits.append((token_address, f"Take Profit: {position.get_pnl_percent():.2f}%"))

            # Stop loss
            elif position.should_stop_loss(self.config.stop_loss_pct):
                exits.append((token_address, f"Stop Loss: {position.get_pnl_percent():.2f}%"))

            # Trailing stop
            elif position.should_trailing_stop(self.config.trailing_stop_pct):
                exits.append(
                    (
                        token_address,
                        f"Trailing Stop: down {self.config.trailing_stop_pct}% from peak",
                    )
                )

            # Time-based exit
            elif position.is_expired(self.config.max_position_age_minutes):
                exits.append((token_address, "Max hold time reached"))

        return exits

    def get_portfolio_summary(self) -> dict:
        """Beautiful portfolio snapshot."""
        total_invested = sum(p.amount_sol_invested for p in self.positions.values())
        total_current = sum(p.amount_tokens * p.current_price for p in self.positions.values())
        total_pnl = total_current - total_invested

        return {
            "num_positions": len(self.positions),
            "total_invested_sol": total_invested,
            "current_value_sol": total_current,
            "total_pnl_sol": total_pnl,
            "total_pnl_pct": (total_pnl / total_invested * 100) if total_invested > 0 else 0,
        }
