#!/usr/bin/env python3
"""
FENRIR - AI Session Memory

Rolling memory of AI trading decisions and outcomes.
Gives Claude context across decisions so it can learn from
recent patterns within a session (e.g., "last 3 buys were rugs").
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class DecisionRecord:
    """A single AI decision and its eventual outcome."""
    timestamp: datetime
    token_mint: str
    token_symbol: str
    token_name: str
    decision: str  # "STRONG_BUY", "BUY", "SKIP", "AVOID"
    confidence: float
    risk_score: float
    reasoning_summary: str  # Truncated to ~100 chars
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)

    # Outcome tracking (filled in after trade closes)
    was_bought: bool = False
    outcome_pnl_pct: Optional[float] = None
    outcome_exit_reason: Optional[str] = None
    outcome_hold_time_minutes: Optional[int] = None


class AISessionMemory:
    """
    Rolling memory of AI decisions and outcomes for context injection.

    Maintains a FIFO buffer of recent decisions so Claude can see
    what it previously decided and how those decisions played out.
    Memory resets on bot restart (intentional — stale context misleads).
    """

    def __init__(self, max_size: int = 15):
        self.decisions: List[DecisionRecord] = []
        self.max_size = max_size
        self.session_start: datetime = datetime.now()

        # Running tallies
        self._total_buys = 0
        self._total_skips = 0
        self._profitable_trades = 0
        self._closed_trades = 0
        self._total_pnl_sol = 0.0

    def record_decision(self, record: DecisionRecord) -> None:
        """Add a decision to the rolling buffer. Evicts oldest if full."""
        self.decisions.append(record)
        if len(self.decisions) > self.max_size:
            self.decisions.pop(0)

        if record.was_bought:
            self._total_buys += 1
        else:
            self._total_skips += 1

    def update_outcome(
        self,
        token_mint: str,
        pnl_pct: float,
        exit_reason: str,
        hold_time_minutes: int,
        pnl_sol: float = 0.0,
    ) -> None:
        """Backfill outcome for a previously recorded decision."""
        for record in reversed(self.decisions):
            if record.token_mint == token_mint and record.was_bought:
                record.outcome_pnl_pct = pnl_pct
                record.outcome_exit_reason = exit_reason
                record.outcome_hold_time_minutes = hold_time_minutes

                self._closed_trades += 1
                if pnl_pct > 0:
                    self._profitable_trades += 1
                self._total_pnl_sol += pnl_sol
                break

    def build_context_block(self) -> str:
        """
        Build a formatted text block summarizing recent decisions and outcomes
        for injection into Claude's system prompt.
        """
        if not self.decisions:
            return ""

        now = datetime.now()
        lines = [f"# YOUR RECENT DECISIONS (last {len(self.decisions)})"]

        for i, d in enumerate(reversed(self.decisions), 1):
            age = now - d.timestamp
            age_str = self._format_age(age.total_seconds())

            outcome_str = ""
            if d.was_bought and d.outcome_pnl_pct is not None:
                emoji = "+" if d.outcome_pnl_pct > 0 else ""
                hold = f" in {d.outcome_hold_time_minutes}min" if d.outcome_hold_time_minutes else ""
                outcome_str = f" -> RESULT: {emoji}{d.outcome_pnl_pct:.1f}%{hold}"
            elif d.was_bought:
                outcome_str = " -> OPEN (no result yet)"

            lines.append(
                f"{i}. [{age_str}] {d.decision} ${d.token_symbol} "
                f"(conf: {d.confidence:.1f}, risk: {d.risk_score:.1f}) "
                f'- "{d.reasoning_summary}"{outcome_str}'
            )

        # Session stats
        stats = self.get_session_stats()
        lines.append("")
        lines.append("# SESSION PERFORMANCE")
        lines.append(
            f"- Decisions: {stats['total_decisions']} "
            f"({stats['total_buys']} BUY, {stats['total_skips']} SKIP)"
        )
        if stats["closed_trades"] > 0:
            lines.append(
                f"- Closed: {stats['closed_trades']}, "
                f"Profitable: {stats['profitable_trades']} "
                f"({stats['win_rate']:.0%} win rate)"
            )
            lines.append(f"- Session P&L: {stats['total_pnl_sol']:+.4f} SOL")

        return "\n".join(lines)

    def build_portfolio_context(self, positions: Dict) -> str:
        """
        Build a text block describing current open positions.

        Args:
            positions: Dict of token_address -> Position objects
                       (from PositionManager.positions)
        """
        if not positions:
            return "# CURRENT PORTFOLIO\nNo open positions."

        lines = [f"# CURRENT PORTFOLIO ({len(positions)} open positions)"]
        total_invested = 0.0
        total_current = 0.0

        for i, (addr, pos) in enumerate(positions.items(), 1):
            pnl_pct = pos.get_pnl_percent()
            hold_secs = (datetime.now() - pos.entry_time).total_seconds()
            hold_min = int(hold_secs / 60)

            invested = pos.amount_sol_invested
            current_val = invested * (1 + pnl_pct / 100) if invested else 0.0
            total_invested += invested
            total_current += current_val

            symbol = addr[:8] + "..."
            emoji = "+" if pnl_pct >= 0 else ""
            lines.append(
                f"{i}. ${symbol}: {emoji}{pnl_pct:.1f}% "
                f"({hold_min}min hold, entry {invested:.3f} SOL)"
            )

        lines.append(
            f"Total invested: {total_invested:.3f} SOL | "
            f"Current value: {total_current:.3f} SOL | "
            f"P&L: {total_current - total_invested:+.4f} SOL"
        )
        return "\n".join(lines)

    def get_risk_appetite_adjustment(self) -> str:
        """
        Based on recent outcomes, suggest risk adjustment context.
        Gives Claude situational awareness about session momentum.
        """
        if not self.decisions:
            return ""

        # Count recent consecutive outcomes
        recent_bought = [d for d in reversed(self.decisions) if d.was_bought]
        if not recent_bought:
            return "No trades executed yet this session. First trade — be selective."

        # Check for losing streaks
        consecutive_losses = 0
        for d in reversed(recent_bought):
            if d.outcome_pnl_pct is not None and d.outcome_pnl_pct < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            return (
                f"WARNING: {consecutive_losses} consecutive losing trades. "
                "Be extra cautious. Only recommend BUY for high-conviction setups."
            )

        # Check for winning streaks
        consecutive_wins = 0
        for d in reversed(recent_bought):
            if d.outcome_pnl_pct is not None and d.outcome_pnl_pct > 0:
                consecutive_wins += 1
            else:
                break

        if consecutive_wins >= 3:
            return (
                f"Session going well ({consecutive_wins} wins in a row). "
                "Don't get overconfident — maintain risk discipline."
            )

        # Check for recent rug
        for d in reversed(self.decisions[:5]):
            if d.was_bought and d.outcome_pnl_pct is not None and d.outcome_pnl_pct < -50:
                age = (datetime.now() - d.timestamp).total_seconds()
                if age < 600:  # Within last 10 minutes
                    return (
                        f"Recent large loss on ${d.token_symbol} "
                        f"({d.outcome_pnl_pct:+.0f}%) {int(age/60)} min ago. "
                        "Watch for similar patterns."
                    )

        return ""

    def get_session_stats(self) -> Dict:
        """Return aggregate stats for the session."""
        return {
            "session_age_minutes": int(
                (datetime.now() - self.session_start).total_seconds() / 60
            ),
            "total_decisions": len(self.decisions),
            "total_buys": self._total_buys,
            "total_skips": self._total_skips,
            "closed_trades": self._closed_trades,
            "profitable_trades": self._profitable_trades,
            "win_rate": (
                self._profitable_trades / self._closed_trades
                if self._closed_trades > 0
                else 0.0
            ),
            "total_pnl_sol": self._total_pnl_sol,
        }

    @staticmethod
    def _format_age(seconds: float) -> str:
        """Format age in human-readable form."""
        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds / 60)}min ago"
        else:
            return f"{seconds / 3600:.1f}h ago"
