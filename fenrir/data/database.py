#!/usr/bin/env python3
"""
FENRIR - Trade Database & Analytics

Persistent storage for all trading activity using SQLite.
Tracks trades, positions, performance metrics for:
- Historical analysis
- Tax reporting
- Strategy optimization
- Backtesting validation

Database Schema:
- trades: Individual buy/sell transactions
- positions: Position lifecycle (open -> close)
- daily_stats: Aggregated daily performance
- configuration_snapshots: Bot config at time of trade
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Trade:
    """Individual trade record."""
    id: Optional[int] = None
    timestamp: datetime = None
    trade_type: str = None  # "BUY" or "SELL"
    token_mint: str = None
    token_symbol: str = None
    amount_sol: float = None
    amount_tokens: float = None
    price_per_token: float = None
    slippage_pct: float = None
    gas_fee_sol: float = None
    signature: str = None
    position_id: Optional[int] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PositionRecord:
    """Position lifecycle tracking."""
    id: Optional[int] = None
    token_mint: str = None
    token_symbol: str = None
    open_time: datetime = None
    close_time: Optional[datetime] = None
    
    # Entry
    entry_price: float = None
    entry_amount_tokens: float = None
    entry_amount_sol: float = None
    entry_signature: str = None
    
    # Exit
    exit_price: Optional[float] = None
    exit_amount_tokens: Optional[float] = None
    exit_amount_sol: Optional[float] = None
    exit_signature: Optional[str] = None
    exit_reason: Optional[str] = None
    
    # Performance
    pnl_sol: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_time_minutes: Optional[int] = None
    peak_price: float = None
    max_drawdown_pct: Optional[float] = None
    
    # Metadata
    strategy: str = "default"
    notes: str = ""
    
    def __post_init__(self):
        if self.open_time is None:
            self.open_time = datetime.now()
        if self.peak_price is None:
            self.peak_price = self.entry_price


@dataclass
class DailyStats:
    """Daily performance metrics."""
    date: str  # YYYY-MM-DD
    trades_total: int = 0
    trades_buy: int = 0
    trades_sell: int = 0
    
    positions_opened: int = 0
    positions_closed: int = 0
    positions_profitable: int = 0
    
    pnl_sol: float = 0.0
    pnl_pct: float = 0.0
    
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    total_volume_sol: float = 0.0
    total_gas_fees_sol: float = 0.0
    
    largest_win_pct: float = 0.0
    largest_loss_pct: float = 0.0


class TradeDatabase:
    """
    SQLite database manager for trading data.
    Provides CRUD operations and analytics queries.
    """
    
    def __init__(self, db_path: str = "fenrir_trades.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Create database and tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        
        cursor = self.conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                token_mint TEXT NOT NULL,
                token_symbol TEXT,
                amount_sol REAL NOT NULL,
                amount_tokens REAL NOT NULL,
                price_per_token REAL NOT NULL,
                slippage_pct REAL,
                gas_fee_sol REAL,
                signature TEXT,
                position_id INTEGER,
                notes TEXT,
                FOREIGN KEY (position_id) REFERENCES positions (id)
            )
        """)
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_mint TEXT NOT NULL,
                token_symbol TEXT,
                open_time TEXT NOT NULL,
                close_time TEXT,
                
                entry_price REAL NOT NULL,
                entry_amount_tokens REAL NOT NULL,
                entry_amount_sol REAL NOT NULL,
                entry_signature TEXT,
                
                exit_price REAL,
                exit_amount_tokens REAL,
                exit_amount_sol REAL,
                exit_signature TEXT,
                exit_reason TEXT,
                
                pnl_sol REAL,
                pnl_pct REAL,
                hold_time_minutes INTEGER,
                peak_price REAL,
                max_drawdown_pct REAL,
                
                strategy TEXT DEFAULT 'default',
                notes TEXT
            )
        """)
        
        # Daily stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                trades_total INTEGER DEFAULT 0,
                trades_buy INTEGER DEFAULT 0,
                trades_sell INTEGER DEFAULT 0,
                
                positions_opened INTEGER DEFAULT 0,
                positions_closed INTEGER DEFAULT 0,
                positions_profitable INTEGER DEFAULT 0,
                
                pnl_sol REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                
                win_rate REAL DEFAULT 0,
                avg_win_pct REAL DEFAULT 0,
                avg_loss_pct REAL DEFAULT 0,
                
                total_volume_sol REAL DEFAULT 0,
                total_gas_fees_sol REAL DEFAULT 0,
                
                largest_win_pct REAL DEFAULT 0,
                largest_loss_pct REAL DEFAULT 0
            )
        """)
        
        # Configuration snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                config_json TEXT NOT NULL
            )
        """)
        
        # Indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
            ON trades (timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_positions_token 
            ON positions (token_mint)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_positions_close_time 
            ON positions (close_time)
        """)
        
        self.conn.commit()
    
    # ═══════════════════════════════════════════════════════════════════════
    #                           TRADE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    def record_trade(self, trade: Trade) -> int:
        """
        Record a trade in the database.
        Returns the trade ID.
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                timestamp, trade_type, token_mint, token_symbol,
                amount_sol, amount_tokens, price_per_token,
                slippage_pct, gas_fee_sol, signature,
                position_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.timestamp.isoformat(),
            trade.trade_type,
            trade.token_mint,
            trade.token_symbol,
            trade.amount_sol,
            trade.amount_tokens,
            trade.price_per_token,
            trade.slippage_pct,
            trade.gas_fee_sol,
            trade.signature,
            trade.position_id,
            trade.notes
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent_trades(self, limit: int = 50) -> List[Trade]:
        """Get most recent trades."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        return [self._row_to_trade(row) for row in rows]
    
    # ═══════════════════════════════════════════════════════════════════════
    #                         POSITION OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    def open_position(self, position: PositionRecord) -> int:
        """
        Open a new position.
        Returns the position ID.
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO positions (
                token_mint, token_symbol, open_time,
                entry_price, entry_amount_tokens, entry_amount_sol,
                entry_signature, peak_price, strategy, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.token_mint,
            position.token_symbol,
            position.open_time.isoformat(),
            position.entry_price,
            position.entry_amount_tokens,
            position.entry_amount_sol,
            position.entry_signature,
            position.peak_price,
            position.strategy,
            position.notes
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def close_position(
        self,
        position_id: int,
        exit_price: float,
        exit_amount_tokens: float,
        exit_amount_sol: float,
        exit_signature: str,
        exit_reason: str
    ):
        """Close an existing position and calculate P&L."""
        cursor = self.conn.cursor()
        
        # Get position details
        cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
        row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"Position {position_id} not found")
        
        # Calculate metrics
        open_time = datetime.fromisoformat(row["open_time"])
        close_time = datetime.now()
        hold_time_minutes = int((close_time - open_time).total_seconds() / 60)
        
        pnl_sol = exit_amount_sol - row["entry_amount_sol"]
        pnl_pct = (pnl_sol / row["entry_amount_sol"]) * 100
        
        # Calculate max drawdown
        peak_price = row["peak_price"]
        max_drawdown_pct = ((peak_price - exit_price) / peak_price) * 100
        
        # Update position
        cursor.execute("""
            UPDATE positions SET
                close_time = ?,
                exit_price = ?,
                exit_amount_tokens = ?,
                exit_amount_sol = ?,
                exit_signature = ?,
                exit_reason = ?,
                pnl_sol = ?,
                pnl_pct = ?,
                hold_time_minutes = ?,
                max_drawdown_pct = ?
            WHERE id = ?
        """, (
            close_time.isoformat(),
            exit_price,
            exit_amount_tokens,
            exit_amount_sol,
            exit_signature,
            exit_reason,
            pnl_sol,
            pnl_pct,
            hold_time_minutes,
            max_drawdown_pct,
            position_id
        ))
        
        self.conn.commit()
    
    def update_position_peak_price(self, position_id: int, new_peak: float):
        """Update peak price for trailing stop calculations."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE positions SET peak_price = ?
            WHERE id = ? AND peak_price < ?
        """, (new_peak, position_id, new_peak))
        self.conn.commit()
    
    def get_open_positions(self) -> List[PositionRecord]:
        """Get all currently open positions."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM positions
            WHERE close_time IS NULL
            ORDER BY open_time DESC
        """)
        
        rows = cursor.fetchall()
        return [self._row_to_position(row) for row in rows]
    
    def get_closed_positions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PositionRecord]:
        """Get closed positions within date range."""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM positions WHERE close_time IS NOT NULL"
        params = []
        
        if start_date:
            query += " AND close_time >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND close_time <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY close_time DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [self._row_to_position(row) for row in rows]
    
    # ═══════════════════════════════════════════════════════════════════════
    #                         ANALYTICS & REPORTING
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_performance_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get comprehensive performance summary.
        Returns key trading metrics.
        """
        positions = self.get_closed_positions(start_date, end_date)
        
        if not positions:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl_sol": 0,
                "avg_pnl_pct": 0,
                "avg_win_pct": 0,
                "avg_loss_pct": 0,
                "largest_win_pct": 0,
                "largest_loss_pct": 0,
                "avg_hold_time_minutes": 0,
                "sharpe_ratio": 0
            }
        
        # Calculate metrics
        total_trades = len(positions)
        winning_trades = [p for p in positions if p.pnl_sol > 0]
        losing_trades = [p for p in positions if p.pnl_sol <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl_sol = sum(p.pnl_sol for p in positions)
        avg_pnl_pct = sum(p.pnl_pct for p in positions) / total_trades
        
        avg_win_pct = (sum(p.pnl_pct for p in winning_trades) / len(winning_trades)
                      if winning_trades else 0)
        avg_loss_pct = (sum(p.pnl_pct for p in losing_trades) / len(losing_trades)
                       if losing_trades else 0)
        
        largest_win_pct = max((p.pnl_pct for p in positions), default=0)
        largest_loss_pct = min((p.pnl_pct for p in positions), default=0)
        
        avg_hold_time = sum(p.hold_time_minutes for p in positions) / total_trades
        
        # Sharpe ratio (simplified: return / std_dev)
        returns = [p.pnl_pct for p in positions]
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl_sol": total_pnl_sol,
            "avg_pnl_pct": avg_pnl_pct,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "largest_win_pct": largest_win_pct,
            "largest_loss_pct": largest_loss_pct,
            "avg_hold_time_minutes": avg_hold_time,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0
        }
    
    def export_for_taxes(self, year: int):
        """
        Export trades in format suitable for tax reporting.
        Returns DataFrame with all trades for the specified year.
        Requires pandas to be installed.
        """
        import pandas as pd

        cursor = self.conn.cursor()
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        cursor.execute("""
            SELECT 
                t.timestamp,
                t.trade_type,
                t.token_mint,
                t.token_symbol,
                t.amount_sol,
                t.amount_tokens,
                t.price_per_token,
                t.signature,
                p.pnl_sol,
                p.pnl_pct
            FROM trades t
            LEFT JOIN positions p ON t.position_id = p.id
            WHERE t.timestamp >= ? AND t.timestamp <= ?
            ORDER BY t.timestamp
        """, (start_date, end_date))
        
        rows = cursor.fetchall()
        
        # Convert to DataFrame
        data = []
        for row in rows:
            data.append({
                "Date": row["timestamp"],
                "Type": row["trade_type"],
                "Token": row["token_symbol"] or row["token_mint"][:8],
                "SOL Amount": row["amount_sol"],
                "Token Amount": row["amount_tokens"],
                "Price": row["price_per_token"],
                "P&L SOL": row["pnl_sol"],
                "P&L %": row["pnl_pct"],
                "Signature": row["signature"]
            })
        
        return pd.DataFrame(data)
    
    def get_best_performing_tokens(self, limit: int = 10) -> List[Dict]:
        """Get tokens with highest average returns."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                token_mint,
                token_symbol,
                COUNT(*) as trades,
                AVG(pnl_pct) as avg_return,
                SUM(pnl_sol) as total_pnl
            FROM positions
            WHERE close_time IS NOT NULL
            GROUP BY token_mint
            ORDER BY avg_return DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def _row_to_trade(self, row) -> Trade:
        """Convert database row to Trade object."""
        return Trade(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            trade_type=row["trade_type"],
            token_mint=row["token_mint"],
            token_symbol=row["token_symbol"],
            amount_sol=row["amount_sol"],
            amount_tokens=row["amount_tokens"],
            price_per_token=row["price_per_token"],
            slippage_pct=row["slippage_pct"],
            gas_fee_sol=row["gas_fee_sol"],
            signature=row["signature"],
            position_id=row["position_id"],
            notes=row["notes"]
        )
    
    def _row_to_position(self, row) -> PositionRecord:
        """Convert database row to PositionRecord object."""
        return PositionRecord(
            id=row["id"],
            token_mint=row["token_mint"],
            token_symbol=row["token_symbol"],
            open_time=datetime.fromisoformat(row["open_time"]),
            close_time=datetime.fromisoformat(row["close_time"]) if row["close_time"] else None,
            entry_price=row["entry_price"],
            entry_amount_tokens=row["entry_amount_tokens"],
            entry_amount_sol=row["entry_amount_sol"],
            entry_signature=row["entry_signature"],
            exit_price=row["exit_price"],
            exit_amount_tokens=row["exit_amount_tokens"],
            exit_amount_sol=row["exit_amount_sol"],
            exit_signature=row["exit_signature"],
            exit_reason=row["exit_reason"],
            pnl_sol=row["pnl_sol"],
            pnl_pct=row["pnl_pct"],
            hold_time_minutes=row["hold_time_minutes"],
            peak_price=row["peak_price"],
            max_drawdown_pct=row["max_drawdown_pct"],
            strategy=row["strategy"],
            notes=row["notes"]
        )
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# ═══════════════════════════════════════════════════════════════════════════
#                              EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 FENRIR - Trade Database")
    print("=" * 70)
    
    # Initialize database
    db = TradeDatabase("fenrir_trades_example.db")
    
    # Example: Record a position
    print("\n📊 Recording example position...")
    
    position = PositionRecord(
        token_mint="TOKEN123ABC",
        token_symbol="WOLF",
        entry_price=0.000001,
        entry_amount_tokens=1000000,
        entry_amount_sol=1.0,
        entry_signature="sig123",
        strategy="sniper"
    )
    
    position_id = db.open_position(position)
    print(f"✅ Position opened with ID: {position_id}")
    
    # Simulate closing the position
    db.close_position(
        position_id=position_id,
        exit_price=0.000002,
        exit_amount_tokens=1000000,
        exit_amount_sol=2.0,
        exit_signature="sig456",
        exit_reason="Take profit at +100%"
    )
    print(f"✅ Position closed")
    
    # Get performance summary
    summary = db.get_performance_summary()
    print(f"\n📈 Performance Summary:")
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Win Rate: {summary['win_rate']:.1f}%")
    print(f"   Total P&L: {summary['total_pnl_sol']:+.4f} SOL")
    print(f"   Avg Return: {summary['avg_pnl_pct']:+.1f}%")
    print(f"   Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    
    db.close()
    print("\n✅ Database closed")
