#!/usr/bin/env python3
"""
FENRIR - Historical Memory (Cross-Session Persistence)

Persists trade outcomes across bot restarts so Claude can learn from
historical patterns. Unlike AISessionMemory (in-memory, resets on restart),
this module stores aggregated stats in SQLite and injects them into
AI prompts as concise statistical context.

What it tracks:
- Per-creator stats: launch count, buy count, avg PnL, rug count
- Per-token-profile stats: win rate by liquidity range, market cap range
- Time-of-day patterns: win rate by hour of day
- AI decision accuracy: how well did confidence predict outcomes

Usage:
    memory = HistoricalMemory(db_path="fenrir_trades.db")

    # After every closed trade:
    memory.record_outcome(
        token_address="ABC123",
        token_symbol="WOLF",
        creator_address="CREATOR456",
        initial_liquidity_sol=5.0,
        market_cap_sol=30.0,
        ai_decision="BUY",
        ai_confidence=0.85,
        ai_risk_score=4.2,
        was_bought=True,
        pnl_pct=42.0,
        pnl_sol=0.042,
        hold_time_minutes=12,
        exit_reason="Take Profit",
        strategy_id="sniper",
    )

    # Before AI evaluates a new token:
    context = memory.build_historical_context(
        creator_address="CREATOR456",
        initial_liquidity_sol=5.0,
        market_cap_sol=30.0,
    )
    # -> Injects concise stats into AI prompt
"""

import sqlite3
from datetime import datetime


class HistoricalMemory:
    """
    SQLite-backed cross-session memory for trading intelligence.

    Stores individual outcomes and maintains aggregate tables
    for fast context generation.
    """

    def __init__(self, db_path: str = "fenrir_trades.db"):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create tables for historical tracking."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Individual outcome records
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                token_address TEXT NOT NULL,
                token_symbol TEXT,
                creator_address TEXT,
                initial_liquidity_sol REAL,
                market_cap_sol REAL,
                ai_decision TEXT,
                ai_confidence REAL,
                ai_risk_score REAL,
                was_bought INTEGER NOT NULL DEFAULT 0,
                pnl_pct REAL,
                pnl_sol REAL,
                hold_time_minutes INTEGER,
                exit_reason TEXT,
                strategy_id TEXT,
                hour_of_day INTEGER
            )
        """)

        # Aggregated creator statistics
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS creator_stats (
                creator_address TEXT PRIMARY KEY,
                total_launches INTEGER DEFAULT 0,
                tokens_bought INTEGER DEFAULT 0,
                tokens_profitable INTEGER DEFAULT 0,
                avg_pnl_pct REAL DEFAULT 0.0,
                total_pnl_sol REAL DEFAULT 0.0,
                rug_count INTEGER DEFAULT 0,
                best_pnl_pct REAL DEFAULT 0.0,
                worst_pnl_pct REAL DEFAULT 0.0,
                avg_hold_minutes REAL DEFAULT 0.0,
                first_seen TEXT,
                last_seen TEXT
            )
        """)

        # Indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hist_creator
            ON historical_outcomes (creator_address)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hist_token
            ON historical_outcomes (token_address)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hist_timestamp
            ON historical_outcomes (timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hist_bought
            ON historical_outcomes (was_bought)
        """)

        self.conn.commit()

    def record_outcome(
        self,
        token_address: str,
        token_symbol: str = "???",  # noqa: S107
        creator_address: str | None = None,
        initial_liquidity_sol: float = 0.0,
        market_cap_sol: float = 0.0,
        ai_decision: str = "",
        ai_confidence: float = 0.0,
        ai_risk_score: float = 0.0,
        was_bought: bool = False,
        pnl_pct: float | None = None,
        pnl_sol: float | None = None,
        hold_time_minutes: int | None = None,
        exit_reason: str | None = None,
        strategy_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Record a trade outcome for cross-session learning.

        Call this when a position closes (or when AI skips a token
        and you want to track what would have happened).
        """
        now = datetime.now()
        hour = now.hour

        self.conn.execute(
            """
            INSERT INTO historical_outcomes
                (session_id, timestamp, token_address, token_symbol,
                 creator_address, initial_liquidity_sol, market_cap_sol,
                 ai_decision, ai_confidence, ai_risk_score,
                 was_bought, pnl_pct, pnl_sol, hold_time_minutes,
                 exit_reason, strategy_id, hour_of_day)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                now.isoformat(),
                token_address,
                token_symbol,
                creator_address,
                initial_liquidity_sol,
                market_cap_sol,
                ai_decision,
                ai_confidence,
                ai_risk_score,
                int(was_bought),
                pnl_pct,
                pnl_sol,
                hold_time_minutes,
                exit_reason,
                strategy_id,
                hour,
            ),
        )

        # Update creator stats
        if creator_address:
            self._update_creator_stats(
                creator_address, was_bought, pnl_pct, pnl_sol, hold_time_minutes, now
            )

        self.conn.commit()

    def _update_creator_stats(
        self,
        creator_address: str,
        was_bought: bool,
        pnl_pct: float | None,
        pnl_sol: float | None,
        hold_time_minutes: int | None,
        now: datetime,
    ) -> None:
        """Update aggregate stats for a creator."""
        existing = self.conn.execute(
            "SELECT * FROM creator_stats WHERE creator_address = ?",
            (creator_address,),
        ).fetchone()

        now_str = now.isoformat()

        if not existing:
            self.conn.execute(
                """
                INSERT INTO creator_stats
                    (creator_address, total_launches, tokens_bought,
                     tokens_profitable, avg_pnl_pct, total_pnl_sol,
                     rug_count, best_pnl_pct, worst_pnl_pct,
                     avg_hold_minutes, first_seen, last_seen)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    creator_address,
                    int(was_bought),
                    int(was_bought and pnl_pct is not None and pnl_pct > 0),
                    pnl_pct or 0.0,
                    pnl_sol or 0.0,
                    int(was_bought and pnl_pct is not None and pnl_pct < -80),
                    pnl_pct or 0.0,
                    pnl_pct or 0.0,
                    float(hold_time_minutes) if hold_time_minutes else 0.0,
                    now_str,
                    now_str,
                ),
            )
        else:
            total = existing["total_launches"] + 1
            bought = existing["tokens_bought"] + (1 if was_bought else 0)
            profitable = existing["tokens_profitable"]
            if was_bought and pnl_pct is not None and pnl_pct > 0:
                profitable += 1

            # Incremental average PnL
            old_avg = existing["avg_pnl_pct"]
            old_bought = existing["tokens_bought"]
            if was_bought and pnl_pct is not None and old_bought > 0:
                new_avg = (old_avg * old_bought + pnl_pct) / bought
            elif was_bought and pnl_pct is not None:
                new_avg = pnl_pct
            else:
                new_avg = old_avg

            total_sol = existing["total_pnl_sol"] + (pnl_sol or 0.0)
            rugs = existing["rug_count"]
            if was_bought and pnl_pct is not None and pnl_pct < -80:
                rugs += 1

            best = max(existing["best_pnl_pct"], pnl_pct or 0.0)
            worst = min(existing["worst_pnl_pct"], pnl_pct or 0.0)

            # Avg hold time
            old_hold = existing["avg_hold_minutes"]
            if was_bought and hold_time_minutes is not None and old_bought > 0:
                new_hold = (old_hold * old_bought + hold_time_minutes) / bought
            elif was_bought and hold_time_minutes is not None:
                new_hold = float(hold_time_minutes)
            else:
                new_hold = old_hold

            self.conn.execute(
                """
                UPDATE creator_stats SET
                    total_launches = ?,
                    tokens_bought = ?,
                    tokens_profitable = ?,
                    avg_pnl_pct = ?,
                    total_pnl_sol = ?,
                    rug_count = ?,
                    best_pnl_pct = ?,
                    worst_pnl_pct = ?,
                    avg_hold_minutes = ?,
                    last_seen = ?
                WHERE creator_address = ?
                """,
                (
                    total, bought, profitable, new_avg, total_sol,
                    rugs, best, worst, new_hold, now_str, creator_address,
                ),
            )

    # ──────────────────────────────────────────────────────────────
    #  CONTEXT GENERATION (for AI prompt injection)
    # ──────────────────────────────────────────────────────────────

    def build_historical_context(
        self,
        creator_address: str | None = None,
        initial_liquidity_sol: float = 0.0,
        market_cap_sol: float = 0.0,
        max_tokens: int = 250,
    ) -> str:
        """
        Build a concise text block of historical patterns for AI injection.

        Keeps output under ~250 tokens to avoid bloating the prompt.
        Returns empty string if no relevant history exists.
        """
        lines = []

        # Creator profile
        if creator_address:
            creator_ctx = self._get_creator_context(creator_address)
            if creator_ctx:
                lines.append(creator_ctx)

        # Liquidity-range stats
        liq_ctx = self._get_liquidity_range_context(initial_liquidity_sol)
        if liq_ctx:
            lines.append(liq_ctx)

        # Time-of-day stats
        hour_ctx = self._get_hour_context()
        if hour_ctx:
            lines.append(hour_ctx)

        # Overall session-independent stats
        overall_ctx = self._get_overall_context()
        if overall_ctx:
            lines.append(overall_ctx)

        if not lines:
            return ""

        return "# HISTORICAL PATTERNS (from past sessions)\n" + "\n".join(lines)

    def _get_creator_context(self, creator_address: str) -> str:
        """Get creator's historical track record."""
        row = self.conn.execute(
            "SELECT * FROM creator_stats WHERE creator_address = ?",
            (creator_address,),
        ).fetchone()

        if not row or row["total_launches"] < 1:
            return ""

        total = row["total_launches"]
        bought = row["tokens_bought"]
        profitable = row["tokens_profitable"]
        rugs = row["rug_count"]
        avg_pnl = row["avg_pnl_pct"]

        parts = [f"- Creator {creator_address[:8]}...: {total} launches seen"]

        if bought > 0:
            win_rate = (profitable / bought * 100) if bought > 0 else 0
            parts.append(
                f"  {bought} bought, {win_rate:.0f}% win rate, "
                f"avg PnL {avg_pnl:+.1f}%"
            )
        if rugs > 0:
            parts.append(f"  ⚠️ {rugs} suspected rug(s) (PnL < -80%)")

        return "\n".join(parts)

    def _get_liquidity_range_context(self, liquidity_sol: float) -> str:
        """Get win rate for tokens with similar initial liquidity."""
        # Define bucket: ±50% of the target liquidity
        low = liquidity_sol * 0.5
        high = liquidity_sol * 1.5

        row = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl
            FROM historical_outcomes
            WHERE was_bought = 1
              AND pnl_pct IS NOT NULL
              AND initial_liquidity_sol BETWEEN ? AND ?
            """,
            (low, high),
        ).fetchone()

        if not row or row["total"] < 3:
            return ""

        total = row["total"]
        wins = row["wins"] or 0
        avg_pnl = row["avg_pnl"] or 0.0
        win_rate = (wins / total * 100) if total > 0 else 0

        return (
            f"- Tokens with {low:.1f}-{high:.1f} SOL liquidity: "
            f"{win_rate:.0f}% win rate across {total} trades "
            f"(avg {avg_pnl:+.1f}%)"
        )

    def _get_hour_context(self) -> str:
        """Get performance for current hour of day."""
        hour = datetime.now().hour

        row = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl
            FROM historical_outcomes
            WHERE was_bought = 1
              AND pnl_pct IS NOT NULL
              AND hour_of_day = ?
            """,
            (hour,),
        ).fetchone()

        if not row or row["total"] < 5:
            return ""

        total = row["total"]
        wins = row["wins"] or 0
        avg_pnl = row["avg_pnl"] or 0.0
        win_rate = (wins / total * 100) if total > 0 else 0

        return (
            f"- Trades at {hour:02d}:00 UTC: {win_rate:.0f}% win rate "
            f"across {total} trades (avg {avg_pnl:+.1f}%)"
        )

    def _get_overall_context(self) -> str:
        """Get overall historical performance summary."""
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl,
                SUM(pnl_sol) as total_sol
            FROM historical_outcomes
            WHERE was_bought = 1 AND pnl_pct IS NOT NULL
            """
        ).fetchone()

        if not row or row["total"] < 5:
            return ""

        total = row["total"]
        wins = row["wins"] or 0
        avg_pnl = row["avg_pnl"] or 0.0
        total_sol = row["total_sol"] or 0.0
        win_rate = (wins / total * 100) if total > 0 else 0

        return (
            f"- All-time: {total} trades, {win_rate:.0f}% win rate, "
            f"avg {avg_pnl:+.1f}%, total {total_sol:+.4f} SOL"
        )

    # ──────────────────────────────────────────────────────────────
    #  QUERY METHODS
    # ──────────────────────────────────────────────────────────────

    def get_creator_profile(self, creator_address: str) -> dict | None:
        """Get full creator stats as a dict."""
        row = self.conn.execute(
            "SELECT * FROM creator_stats WHERE creator_address = ?",
            (creator_address,),
        ).fetchone()
        return dict(row) if row else None

    def get_strategy_performance(self, strategy_id: str) -> dict:
        """Get performance breakdown for a specific strategy."""
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl,
                SUM(pnl_sol) as total_sol,
                AVG(hold_time_minutes) as avg_hold
            FROM historical_outcomes
            WHERE strategy_id = ? AND was_bought = 1 AND pnl_pct IS NOT NULL
            """,
            (strategy_id,),
        ).fetchone()

        if not row or not row["total"]:
            return {"total": 0}

        return {
            "total": row["total"],
            "wins": row["wins"] or 0,
            "win_rate": (row["wins"] or 0) / row["total"] * 100,
            "avg_pnl_pct": row["avg_pnl"] or 0.0,
            "total_pnl_sol": row["total_sol"] or 0.0,
            "avg_hold_minutes": row["avg_hold"] or 0.0,
        }

    def get_total_outcomes(self) -> int:
        """Total number of recorded outcomes."""
        row = self.conn.execute(
            "SELECT COUNT(*) as c FROM historical_outcomes"
        ).fetchone()
        return row["c"] if row else 0

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
