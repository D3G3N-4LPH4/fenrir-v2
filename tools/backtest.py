#!/usr/bin/env python3
"""
FENRIR - Backtesting Framework

Test trading strategies on historical launch data before risking real money.

Features:
- Replay historical pump.fun launches
- Simulate bot behavior with different parameters
- Calculate hypothetical P&L
- Optimize parameters via grid search
- Compare strategy variants
- Generate detailed backtest reports

Input data format (JSON):
[
  {
    "timestamp": "2024-01-15T10:30:00Z",
    "token_mint": "ABC...",
    "token_symbol": "WOLF",
    "initial_liquidity_sol": 10.0,
    "price_history": [
      {"time": "2024-01-15T10:30:00Z", "price": 0.000001},
      {"time": "2024-01-15T10:31:00Z", "price": 0.000002},
      ...
    ],
    "final_outcome": "rugged|migrated|stalled",
    "peak_price": 0.000005,
    "time_to_peak_minutes": 15
  },
  ...
]
"""

import bisect
import json
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import statistics

from fenrir.config import BotConfig, TradingMode
from fenrir.core.positions import Position


class BacktestMode(Enum):
    """Backtesting modes."""
    REALISTIC = "realistic"  # Simulates realistic execution
    OPTIMISTIC = "optimistic"  # Assumes perfect fills
    PESSIMISTIC = "pessimistic"  # Adds slippage and delays


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    bot_config: BotConfig
    mode: BacktestMode = BacktestMode.REALISTIC
    execution_delay_seconds: float = 2.0
    slippage_pct: float = 1.0
    starting_capital_sol: float = 10.0
    max_concurrent_positions: int = 3
    optimize_params: bool = False
    param_grid: Optional[Dict] = None


@dataclass
class BacktestTrade:
    """A simulated trade in backtest."""
    timestamp: datetime
    trade_type: str  # "BUY" or "SELL"
    token_mint: str
    token_symbol: str
    price: float
    amount_sol: float
    amount_tokens: float
    reason: str


@dataclass
class BacktestPosition:
    """A simulated position in backtest."""
    token_mint: str
    token_symbol: str
    entry_time: datetime
    entry_price: float
    entry_amount_sol: float
    entry_amount_tokens: float

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_amount_sol: Optional[float] = None
    exit_reason: Optional[str] = None

    pnl_sol: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_time_minutes: Optional[int] = None
    peak_price: float = None

    def __post_init__(self):
        if self.peak_price is None:
            self.peak_price = self.entry_price


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    starting_capital_sol: float
    ending_capital_sol: float
    total_pnl_sol: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    positions: List[BacktestPosition]
    trades: List[BacktestTrade]
    avg_hold_time_minutes: float

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "config": asdict(self.config),
            "positions": [asdict(p) for p in self.positions],
            "trades": [asdict(t) for t in self.trades]
        }


class BacktestEngine:
    """Backtest engine for simulating trading strategies."""

    def __init__(self):
        self.historical_data: List[Dict] = []

    def load_historical_data(self, data: List[Dict]):
        self.historical_data = sorted(data, key=lambda x: x["timestamp"])

    def load_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.load_historical_data(data)

    def run_backtest(
        self,
        config: BacktestConfig,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResults:
        data = self._filter_by_date(start_date, end_date)

        if not data:
            raise ValueError("No historical data in specified range")

        capital = config.starting_capital_sol
        positions: List[BacktestPosition] = []
        closed_positions: List[BacktestPosition] = []
        trades: List[BacktestTrade] = []

        for launch in data:
            should_buy = self._should_buy_token(launch, config.bot_config)

            if should_buy and len(positions) < config.max_concurrent_positions:
                position = self._simulate_buy(launch, config, capital)

                if position:
                    positions.append(position)
                    capital -= position.entry_amount_sol

                    trades.append(BacktestTrade(
                        timestamp=position.entry_time,
                        trade_type="BUY",
                        token_mint=position.token_mint,
                        token_symbol=position.token_symbol,
                        price=position.entry_price,
                        amount_sol=position.entry_amount_sol,
                        amount_tokens=position.entry_amount_tokens,
                        reason="Entry signal"
                    ))

            for position in positions[:]:
                should_exit, reason = self._should_exit_position(
                    position, launch, config.bot_config
                )

                if should_exit:
                    self._simulate_sell(position, launch, config, reason)
                    capital += position.exit_amount_sol
                    positions.remove(position)
                    closed_positions.append(position)

                    trades.append(BacktestTrade(
                        timestamp=position.exit_time,
                        trade_type="SELL",
                        token_mint=position.token_mint,
                        token_symbol=position.token_symbol,
                        price=position.exit_price,
                        amount_sol=position.exit_amount_sol,
                        amount_tokens=position.entry_amount_tokens,
                        reason=reason
                    ))

        results = self._calculate_results(
            config, closed_positions, trades, capital,
            data[0]["timestamp"] if data else datetime.now(),
            data[-1]["timestamp"] if data else datetime.now()
        )

        return results

    def _should_buy_token(self, launch: Dict, config: BotConfig) -> bool:
        if launch["initial_liquidity_sol"] < config.min_initial_liquidity_sol:
            return False
        market_cap = launch.get("initial_market_cap_sol", 0)
        if market_cap > config.max_initial_market_cap_sol:
            return False
        return True

    def _should_exit_position(
        self, position: BacktestPosition, current_data: Dict, config: BotConfig
    ) -> Tuple[bool, str]:
        current_price = self._get_price_at_time(
            current_data, datetime.fromisoformat(current_data["timestamp"])
        )

        if not current_price:
            return False, ""

        if current_price > position.peak_price:
            position.peak_price = current_price

        current_value = position.entry_amount_tokens * current_price
        pnl_pct = ((current_value - position.entry_amount_sol) /
                   position.entry_amount_sol * 100)

        if pnl_pct >= config.take_profit_pct:
            return True, f"Take profit at {pnl_pct:.1f}%"

        if pnl_pct <= -config.stop_loss_pct:
            return True, f"Stop loss at {pnl_pct:.1f}%"

        drawdown_from_peak = ((position.peak_price - current_price) /
                             position.peak_price * 100)
        if drawdown_from_peak >= config.trailing_stop_pct:
            return True, f"Trailing stop ({drawdown_from_peak:.1f}% from peak)"

        current_time = datetime.fromisoformat(current_data["timestamp"])
        hold_time = (current_time - position.entry_time).total_seconds() / 60
        if hold_time >= config.max_position_age_minutes:
            return True, f"Max hold time ({hold_time:.0f} min)"

        return False, ""

    def _simulate_buy(
        self, launch: Dict, config: BacktestConfig, available_capital: float
    ) -> Optional[BacktestPosition]:
        buy_amount = config.bot_config.buy_amount_sol

        if buy_amount > available_capital:
            return None

        entry_time = datetime.fromisoformat(launch["timestamp"])
        entry_time += timedelta(seconds=config.execution_delay_seconds)

        entry_price = self._get_price_at_time(launch, entry_time)

        if not entry_price:
            return None

        if config.mode == BacktestMode.REALISTIC:
            entry_price *= (1 + config.slippage_pct / 100)
        elif config.mode == BacktestMode.PESSIMISTIC:
            entry_price *= (1 + config.slippage_pct * 2 / 100)

        tokens_received = buy_amount / entry_price

        return BacktestPosition(
            token_mint=launch["token_mint"],
            token_symbol=launch.get("token_symbol", "UNKNOWN"),
            entry_time=entry_time,
            entry_price=entry_price,
            entry_amount_sol=buy_amount,
            entry_amount_tokens=tokens_received,
            peak_price=entry_price
        )

    def _simulate_sell(
        self, position: BacktestPosition, launch: Dict,
        config: BacktestConfig, reason: str
    ):
        exit_time = datetime.fromisoformat(launch["timestamp"])
        exit_time += timedelta(seconds=config.execution_delay_seconds)

        exit_price = self._get_price_at_time(launch, exit_time)

        if not exit_price:
            exit_price = position.entry_price * 0.9

        if config.mode == BacktestMode.REALISTIC:
            exit_price *= (1 - config.slippage_pct / 100)
        elif config.mode == BacktestMode.PESSIMISTIC:
            exit_price *= (1 - config.slippage_pct * 2 / 100)

        exit_value = position.entry_amount_tokens * exit_price

        position.exit_time = exit_time
        position.exit_price = exit_price
        position.exit_amount_sol = exit_value
        position.exit_reason = reason
        position.pnl_sol = exit_value - position.entry_amount_sol
        position.pnl_pct = (position.pnl_sol / position.entry_amount_sol) * 100
        position.hold_time_minutes = int((exit_time - position.entry_time).total_seconds() / 60)

    def _get_price_at_time(self, launch: Dict, target_time: datetime) -> Optional[float]:
        price_history = launch.get("price_history", [])

        if not price_history:
            return launch.get("initial_price", 0.000001)

        # Build sorted timestamps list for binary search (O(log n))
        timestamps = [datetime.fromisoformat(p["time"]) for p in price_history]
        idx = bisect.bisect_left(timestamps, target_time)

        # Find closest point among neighbors
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(timestamps):
            candidates.append(idx)

        if not candidates:
            return price_history[0]["price"]

        best_idx = min(candidates, key=lambda i: abs((timestamps[i] - target_time).total_seconds()))
        return price_history[best_idx]["price"]

    def _filter_by_date(
        self, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> List[Dict]:
        filtered = self.historical_data

        if start_date:
            filtered = [d for d in filtered
                       if datetime.fromisoformat(d["timestamp"]) >= start_date]

        if end_date:
            filtered = [d for d in filtered
                       if datetime.fromisoformat(d["timestamp"]) <= end_date]

        return filtered

    def _calculate_results(
        self, config: BacktestConfig, positions: List[BacktestPosition],
        trades: List[BacktestTrade], ending_capital: float,
        start_date: datetime, end_date: datetime
    ) -> BacktestResults:
        starting_capital = config.starting_capital_sol

        if not positions:
            return BacktestResults(
                config=config, start_date=start_date, end_date=end_date,
                starting_capital_sol=starting_capital, ending_capital_sol=ending_capital,
                total_pnl_sol=0, total_return_pct=0, total_trades=0,
                winning_trades=0, losing_trades=0, win_rate=0,
                avg_win_pct=0, avg_loss_pct=0, largest_win_pct=0,
                largest_loss_pct=0, max_drawdown_pct=0, sharpe_ratio=0,
                profit_factor=0, positions=[], trades=[], avg_hold_time_minutes=0
            )

        total_pnl = ending_capital - starting_capital
        total_return = (total_pnl / starting_capital) * 100

        winners = [p for p in positions if p.pnl_sol > 0]
        losers = [p for p in positions if p.pnl_sol <= 0]

        win_rate = len(winners) / len(positions) * 100

        avg_win = statistics.mean([p.pnl_pct for p in winners]) if winners else 0
        avg_loss = statistics.mean([p.pnl_pct for p in losers]) if losers else 0

        largest_win = max((p.pnl_pct for p in positions), default=0)
        largest_loss = min((p.pnl_pct for p in positions), default=0)

        returns = [p.pnl_pct / 100 for p in positions]
        sharpe = self._calculate_sharpe(returns)

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        max_dd = self._calculate_max_drawdown(positions, starting_capital)

        avg_hold_time = statistics.mean([p.hold_time_minutes for p in positions])

        return BacktestResults(
            config=config, start_date=start_date, end_date=end_date,
            starting_capital_sol=starting_capital, ending_capital_sol=ending_capital,
            total_pnl_sol=total_pnl, total_return_pct=total_return,
            total_trades=len(positions), winning_trades=len(winners),
            losing_trades=len(losers), win_rate=win_rate,
            avg_win_pct=avg_win, avg_loss_pct=avg_loss,
            largest_win_pct=largest_win, largest_loss_pct=largest_loss,
            max_drawdown_pct=max_dd, sharpe_ratio=sharpe,
            profit_factor=profit_factor, positions=positions,
            trades=trades, avg_hold_time_minutes=avg_hold_time
        )

    def _calculate_sharpe(self, returns: List[float]) -> float:
        if len(returns) < 2:
            return 0
        avg_return = statistics.mean(returns)
        std_dev = statistics.stdev(returns)
        if std_dev == 0:
            return 0
        return (avg_return * 252) / (std_dev * (252 ** 0.5))

    def _calculate_max_drawdown(
        self, positions: List[BacktestPosition], starting_capital: float
    ) -> float:
        capital = starting_capital
        peak = capital
        max_dd = 0

        for p in sorted(positions, key=lambda x: x.exit_time):
            capital += p.pnl_sol
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100
            max_dd = max(max_dd, dd)

        return max_dd

    def generate_report(self, results: BacktestResults) -> str:
        return f"""
=====================================================================
                    BACKTEST RESULTS REPORT
=====================================================================

PERIOD: {results.start_date} to {results.end_date}

---------------------------------------------------------------------
CAPITAL & RETURNS
---------------------------------------------------------------------

Starting Capital:    {results.starting_capital_sol:.2f} SOL
Ending Capital:      {results.ending_capital_sol:.2f} SOL
Total P&L:           {results.total_pnl_sol:+.4f} SOL
Total Return:        {results.total_return_pct:+.1f}%

---------------------------------------------------------------------
TRADE STATISTICS
---------------------------------------------------------------------

Total Trades:        {results.total_trades}
Winning Trades:      {results.winning_trades} ({results.win_rate:.1f}%)
Losing Trades:       {results.losing_trades}
Average Win:         {results.avg_win_pct:+.1f}%
Average Loss:        {results.avg_loss_pct:+.1f}%
Largest Win:         {results.largest_win_pct:+.1f}%
Largest Loss:        {results.largest_loss_pct:+.1f}%

---------------------------------------------------------------------
RISK METRICS
---------------------------------------------------------------------

Sharpe Ratio:        {results.sharpe_ratio:.2f}
Profit Factor:       {results.profit_factor:.2f}x
Max Drawdown:        {results.max_drawdown_pct:.1f}%
Avg Hold Time:       {results.avg_hold_time_minutes:.0f} minutes

=====================================================================
"""


if __name__ == "__main__":
    print("FENRIR - Backtesting Framework")
    print("=" * 70)
    print("\nBacktest strategies on historical data before risking real money")
    print("\nFramework ready - load historical data to begin testing")
