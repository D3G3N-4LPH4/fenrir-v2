#!/usr/bin/env python3
"""
FENRIR - Performance Analytics

Advanced performance metrics and visualization for trading strategies.

Metrics tracked:
- Risk-adjusted returns (Sharpe, Sortino ratios)
- Drawdown analysis
- Win rate and profit factor
- Time-based patterns
- Token-specific performance
- Strategy comparison

Outputs:
- Console reports with beautiful formatting
- HTML dashboards
- CSV exports
- JSON API responses
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from fenrir.data.database import TradeDatabase, PositionRecord


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Time period
    start_date: datetime
    end_date: datetime
    days_traded: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Returns
    total_pnl_sol: float
    total_pnl_pct: float
    avg_win_sol: float
    avg_loss_sol: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    
    # Time analysis
    avg_hold_time_minutes: float
    best_time_of_day: str
    worst_time_of_day: str
    
    # Efficiency
    total_volume_sol: float
    total_gas_fees_sol: float
    gas_as_pct_of_volume: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceAnalyzer:
    """
    Analyze trading performance with professional-grade metrics.
    """
    
    def __init__(self, db: TradeDatabase):
        self.db = db
    
    def calculate_comprehensive_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics for a time period.
        """
        # Get closed positions
        positions = self.db.get_closed_positions(start_date, end_date)
        
        if not positions:
            return self._empty_metrics(start_date, end_date)
        
        # Determine date range
        actual_start = min(p.open_time for p in positions)
        actual_end = max(p.close_time for p in positions if p.close_time)
        days_traded = (actual_end - actual_start).days + 1
        
        # Basic statistics
        total_trades = len(positions)
        winners = [p for p in positions if p.pnl_sol > 0]
        losers = [p for p in positions if p.pnl_sol <= 0]
        
        win_rate = len(winners) / total_trades * 100
        
        # P&L statistics
        total_pnl_sol = sum(p.pnl_sol for p in positions)
        total_pnl_pct = sum(p.pnl_pct for p in positions)
        
        avg_win_sol = sum(p.pnl_sol for p in winners) / len(winners) if winners else 0
        avg_loss_sol = sum(p.pnl_sol for p in losers) / len(losers) if losers else 0
        
        avg_win_pct = sum(p.pnl_pct for p in winners) / len(winners) if winners else 0
        avg_loss_pct = sum(p.pnl_pct for p in losers) / len(losers) if losers else 0
        
        largest_win = max((p.pnl_pct for p in positions), default=0)
        largest_loss = min((p.pnl_pct for p in positions), default=0)
        
        # Risk metrics
        sharpe = self._calculate_sharpe_ratio(positions)
        sortino = self._calculate_sortino_ratio(positions)
        max_dd = self._calculate_max_drawdown(positions)
        profit_factor = abs(avg_win_sol / avg_loss_sol) if avg_loss_sol != 0 else 0
        
        # Time analysis
        avg_hold_time = sum(p.hold_time_minutes for p in positions) / total_trades
        best_hour, worst_hour = self._analyze_time_patterns(positions)
        
        # Volume and fees
        total_volume = sum(p.entry_amount_sol for p in positions)
        # Note: Gas fees would need to be tracked in trades table
        total_gas = 0.0
        gas_pct = (total_gas / total_volume * 100) if total_volume > 0 else 0
        
        return PerformanceMetrics(
            start_date=actual_start,
            end_date=actual_end,
            days_traded=days_traded,
            total_trades=total_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            total_pnl_sol=total_pnl_sol,
            total_pnl_pct=total_pnl_pct,
            avg_win_sol=avg_win_sol,
            avg_loss_sol=avg_loss_sol,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win_pct=largest_win,
            largest_loss_pct=largest_loss,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            profit_factor=profit_factor,
            avg_hold_time_minutes=avg_hold_time,
            best_time_of_day=best_hour,
            worst_time_of_day=worst_hour,
            total_volume_sol=total_volume,
            total_gas_fees_sol=total_gas,
            gas_as_pct_of_volume=gas_pct
        )
    
    def _calculate_sharpe_ratio(
        self,
        positions: List[PositionRecord],
        risk_free_rate: float = 0.05  # 5% annual
    ) -> float:
        """
        Calculate Sharpe ratio: (return - risk_free) / std_dev
        Measures risk-adjusted returns.
        """
        if not positions:
            return 0.0
        
        returns = [p.pnl_pct / 100 for p in positions]  # Convert to decimal
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        std_dev = statistics.stdev(returns)
        
        if std_dev == 0:
            return 0.0
        
        # Annualize assuming daily trades
        annualized_return = avg_return * 252  # 252 trading days
        annualized_std = std_dev * (252 ** 0.5)
        
        sharpe = (annualized_return - risk_free_rate) / annualized_std
        return sharpe
    
    def _calculate_sortino_ratio(
        self,
        positions: List[PositionRecord],
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate Sortino ratio: (return - risk_free) / downside_deviation
        Like Sharpe but only penalizes downside volatility.
        """
        if not positions:
            return 0.0
        
        returns = [p.pnl_pct / 100 for p in positions]
        
        if len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        
        # Only consider negative returns for downside deviation
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')  # No downside = infinite Sortino
        
        downside_dev = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0
        
        if downside_dev == 0:
            return float('inf')
        
        # Annualize
        annualized_return = avg_return * 252
        annualized_downside = downside_dev * (252 ** 0.5)
        
        sortino = (annualized_return - risk_free_rate) / annualized_downside
        return sortino
    
    def _calculate_max_drawdown(self, positions: List[PositionRecord]) -> float:
        """
        Calculate maximum drawdown: largest peak-to-trough decline.
        """
        if not positions:
            return 0.0
        
        # Sort by close time
        sorted_positions = sorted(positions, key=lambda p: p.close_time)
        
        # Calculate cumulative P&L
        cumulative_pnl = []
        total = 0.0
        
        for p in sorted_positions:
            total += p.pnl_sol
            cumulative_pnl.append(total)
        
        # Find max drawdown
        max_drawdown = 0.0
        peak = cumulative_pnl[0]
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / abs(peak) * 100 if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _analyze_time_patterns(
        self,
        positions: List[PositionRecord]
    ) -> Tuple[str, str]:
        """
        Analyze performance by time of day.
        Returns (best_hour, worst_hour).
        """
        # Group by hour of day
        hourly_pnl = defaultdict(list)
        
        for p in positions:
            hour = p.open_time.hour
            hourly_pnl[hour].append(p.pnl_pct)
        
        if not hourly_pnl:
            return ("N/A", "N/A")
        
        # Calculate average P&L per hour
        hourly_avg = {
            hour: statistics.mean(pnls)
            for hour, pnls in hourly_pnl.items()
        }
        
        best_hour = max(hourly_avg, key=hourly_avg.get)
        worst_hour = min(hourly_avg, key=hourly_avg.get)
        
        return (
            f"{best_hour:02d}:00-{best_hour+1:02d}:00",
            f"{worst_hour:02d}:00-{worst_hour+1:02d}:00"
        )
    
    def _empty_metrics(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> PerformanceMetrics:
        """Return empty metrics when no data."""
        now = datetime.now()
        return PerformanceMetrics(
            start_date=start_date or now,
            end_date=end_date or now,
            days_traded=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl_sol=0.0,
            total_pnl_pct=0.0,
            avg_win_sol=0.0,
            avg_loss_sol=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            largest_win_pct=0.0,
            largest_loss_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            avg_hold_time_minutes=0.0,
            best_time_of_day="N/A",
            worst_time_of_day="N/A",
            total_volume_sol=0.0,
            total_gas_fees_sol=0.0,
            gas_as_pct_of_volume=0.0
        )
    
    def generate_console_report(
        self,
        metrics: PerformanceMetrics
    ) -> str:
        """
        Generate beautiful console report.
        """
        report = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                     🐺  FENRIR PERFORMANCE REPORT  🐺                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

📅 PERIOD: {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')} ({metrics.days_traded} days)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TRADE STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Trades:        {metrics.total_trades}
Winning Trades:      {metrics.winning_trades} ({metrics.win_rate:.1f}%)
Losing Trades:       {metrics.losing_trades}

Average Win:         {metrics.avg_win_pct:+.1f}% ({metrics.avg_win_sol:+.4f} SOL)
Average Loss:        {metrics.avg_loss_pct:+.1f}% ({metrics.avg_loss_sol:+.4f} SOL)

Largest Win:         {metrics.largest_win_pct:+.1f}%
Largest Loss:        {metrics.largest_loss_pct:+.1f}%

Average Hold Time:   {metrics.avg_hold_time_minutes:.0f} minutes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 PROFIT & LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total P&L:           {metrics.total_pnl_sol:+.4f} SOL
Total Return:        {metrics.total_pnl_pct:+.1f}%

Total Volume:        {metrics.total_volume_sol:.2f} SOL
Total Gas Fees:      {metrics.total_gas_fees_sol:.4f} SOL ({metrics.gas_as_pct_of_volume:.2f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 RISK-ADJUSTED PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sharpe Ratio:        {metrics.sharpe_ratio:.2f}  {'(Excellent)' if metrics.sharpe_ratio > 2 else '(Good)' if metrics.sharpe_ratio > 1 else '(Fair)' if metrics.sharpe_ratio > 0 else '(Poor)'}
Sortino Ratio:       {metrics.sortino_ratio:.2f}
Profit Factor:       {metrics.profit_factor:.2f}x

Max Drawdown:        {metrics.max_drawdown_pct:.1f}%  {'(Low)' if metrics.max_drawdown_pct < 10 else '(Moderate)' if metrics.max_drawdown_pct < 25 else '(High)'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ TIME ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best Trading Time:   {metrics.best_time_of_day}
Worst Trading Time:  {metrics.worst_time_of_day}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 INSIGHTS:

"""
        # Add insights
        if metrics.win_rate > 50:
            report += "   ✅ Win rate above 50% - strategy is working\n"
        else:
            report += "   ⚠️  Win rate below 50% - consider adjusting strategy\n"
        
        if metrics.profit_factor > 2:
            report += "   ✅ Profit factor > 2x - excellent risk/reward\n"
        elif metrics.profit_factor > 1:
            report += "   ⚠️  Profit factor > 1x - profitable but could improve\n"
        else:
            report += "   ❌ Profit factor < 1x - losing money on average\n"
        
        if metrics.sharpe_ratio > 1:
            report += "   ✅ Sharpe ratio > 1 - good risk-adjusted returns\n"
        else:
            report += "   ⚠️  Sharpe ratio < 1 - returns don't justify risk\n"
        
        if metrics.max_drawdown_pct < 20:
            report += "   ✅ Low drawdown - good risk management\n"
        else:
            report += "   ⚠️  High drawdown - consider tighter stops\n"
        
        report += "\n"
        report += "═" * 75
        
        return report
    
    def export_to_json(self, metrics: PerformanceMetrics) -> str:
        """Export metrics as JSON."""
        return json.dumps(metrics.to_dict(), indent=2, default=str)
    
    def analyze_token_performance(self) -> List[Dict]:
        """
        Analyze performance by token.
        Which tokens were most profitable?
        """
        return self.db.get_best_performing_tokens(limit=20)
    
    def analyze_strategy_comparison(
        self,
        strategies: List[str]
    ) -> Dict[str, PerformanceMetrics]:
        """
        Compare performance across different strategies.
        Requires strategies to be tagged in position records.
        """
        results = {}
        
        for strategy in strategies:
            # Would need to filter positions by strategy
            # For now, return placeholder
            results[strategy] = self._empty_metrics(None, None)
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
#                              EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 FENRIR - Performance Analytics")
    print("=" * 70)
    
    # Example: Generate performance report
    db = TradeDatabase("fenrir_trades_example.db")
    analyzer = PerformanceAnalyzer(db)
    
    # Calculate metrics
    metrics = analyzer.calculate_comprehensive_metrics()
    
    # Generate report
    report = analyzer.generate_console_report(metrics)
    print(report)
    
    # Export as JSON
    json_export = analyzer.export_to_json(metrics)
    with open("performance_report.json", "w") as f:
        f.write(json_export)
    
    print("\n✅ Report generated and exported to performance_report.json")
    
    db.close()
