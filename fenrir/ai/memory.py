#!/usr/bin/env python3
"""
FENRIR - AI Session Memory

Rolling memory of AI trading decisions and outcomes.
Gives Claude context across decisions so it can learn from
recent patterns within a session (e.g., "last 3 buys were rugs").
"""

from dataclasses import dataclass, field
from datetime import datetime


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
    red_flags: list[str] = field(default_factory=list)
    green_flags: list[str] = field(default_factory=list)

    # Outcome tracking (filled in after trade closes)
    was_bought: bool = False
    outcome_pnl_pct: float | None = None
    outcome_exit_reason: str | None = None
    outcome_hold_time_minutes: int | None = None


class AISessionMemory:
    """
    Rolling memory of AI decisions and outcomes for context injection.

    Maintains a FIFO buffer of recent decisions so Claude can see
    what it previously decided and how those decisions played out.
    Memory resets on bot restart (intentional — stale context misleads).
    """

    def __init__(self, max_size: int = 15):
        self.decisions: list[DecisionRecord] = []
        self.max_size = max_size
        self.session_start: datetime = datetime.now()

        # Running tallies
        self._total_buys = 0
        self._total_skips = 0
        self._profitable_trades = 0
        self._closed_trades = 0
        self._total_pnl_sol = 0.0

    # ──────────────────────────────────────────────────────────────
    #  PROJECTION FROM AUDIT CHAIN (§1 harness kernel)
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_audit_chain(
        cls,
        audit,
        session_id: str | None = None,
        max_size: int = 15,
    ) -> "AISessionMemory":
        """
        Reconstruct session memory by projecting the audit chain's event log.

        This makes AISessionMemory a *pure projection* of the append-only audit
        trail rather than a parallel in-memory buffer: the same rendered context
        can be rebuilt deterministically at any time (crash recovery, replay,
        offline analysis), so the live buffer and the durable log can never
        silently drift.

        Folds three event types in chronological order, mirroring the live path
        (record_decision → buy → update_outcome):
          - AI_DECISION   → a DecisionRecord (decision / confidence / risk /
                            reasoning)
          - BUY_EXECUTED  → marks the matching decision was_bought (or synthesizes
                            a bought record when AI was disabled and the bot
                            auto-bought, so trade outcomes still surface)
          - SELL_EXECUTED → backfills the outcome (pnl_pct / reason / hold time)

        Only fields the rendered context actually uses are reconstructed: the
        audit events don't carry red/green flags or the token name, and neither
        does build_context_block(), so nothing visible in the prompt is lost.

        Args:
            audit:      An AuditChain (or anything exposing get_session_log()).
            session_id: Which session to project. Defaults to the chain's current
                        session. NOTE: AuditChain assigns a fresh random
                        session_id per process, so cross-restart resume requires
                        the operator to reuse a session_id; otherwise this
                        projects an empty history (a safe no-op).
            max_size:   Rolling buffer size (keeps the most recent N decisions).

        Returns:
            A populated AISessionMemory. Returns an empty instance on any read
            error — projection must never block AI startup.
        """
        mem = cls(max_size=max_size)
        try:
            records = audit.get_session_log(session_id=session_id, limit=100_000)
        except Exception:
            return mem

        decisions: list[DecisionRecord] = []
        closed = 0
        profitable = 0
        total_pnl_sol = 0.0
        earliest: datetime | None = None

        for rec in records:
            payload = rec.payload or {}
            addr = rec.token_address
            ts = cls._parse_ts(rec.timestamp)
            if earliest is None or ts < earliest:
                earliest = ts

            if rec.event_type == "AI_DECISION":
                decisions.append(
                    DecisionRecord(
                        timestamp=ts,
                        token_mint=addr or "",
                        token_symbol=payload.get("symbol") or "???",
                        token_name=payload.get("name") or payload.get("symbol") or "Unknown",
                        decision=str(payload.get("decision", "SKIP")).upper(),
                        confidence=float(payload.get("confidence", 0.0) or 0.0),
                        risk_score=float(payload.get("risk_score", 0.0) or 0.0),
                        reasoning_summary=str(payload.get("reasoning", ""))[:100],
                    )
                )

            elif rec.event_type == "BUY_EXECUTED" and addr:
                target = cls._latest_unbought(decisions, addr)
                if target is not None:
                    target.was_bought = True
                else:
                    # AI disabled / auto-buy path: no AI_DECISION precedes the
                    # buy, so synthesize a minimal bought record. Keeps trade
                    # outcomes (and the losing-streak signal) reconstructable.
                    decisions.append(
                        DecisionRecord(
                            timestamp=ts,
                            token_mint=addr,
                            token_symbol=payload.get("symbol") or "???",
                            token_name=payload.get("symbol") or "Unknown",
                            decision="BUY",
                            confidence=0.0,
                            risk_score=0.0,
                            reasoning_summary="auto-buy (AI disabled)",
                            was_bought=True,
                        )
                    )

            elif rec.event_type == "SELL_EXECUTED" and addr:
                target = cls._latest_open_bought(decisions, addr)
                if target is not None:
                    target.outcome_pnl_pct = float(payload.get("pnl_pct", 0.0) or 0.0)
                    target.outcome_exit_reason = payload.get("reason", "")
                    target.outcome_hold_time_minutes = int(payload.get("hold_minutes", 0) or 0)
                    closed += 1
                    if target.outcome_pnl_pct > 0:
                        profitable += 1
                    total_pnl_sol += float(payload.get("pnl_sol", 0.0) or 0.0)

        # Tallies count every decision ever seen (matching the live path, where
        # _total_* are not decremented on eviction); the deque keeps the tail.
        mem._total_buys = sum(1 for d in decisions if d.was_bought)
        mem._total_skips = sum(1 for d in decisions if not d.was_bought)
        mem._closed_trades = closed
        mem._profitable_trades = profitable
        mem._total_pnl_sol = total_pnl_sol
        mem.decisions = decisions[-max_size:]
        if earliest is not None:
            mem.session_start = earliest
        return mem

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        """Parse an audit-chain ISO timestamp; fall back to now() on garbage."""
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return datetime.now()

    @staticmethod
    def _latest_unbought(decisions: list[DecisionRecord], token_mint: str) -> DecisionRecord | None:
        """Most recent decision for this token not yet marked bought."""
        for d in reversed(decisions):
            if d.token_mint == token_mint and not d.was_bought:
                return d
        return None

    @staticmethod
    def _latest_open_bought(
        decisions: list[DecisionRecord], token_mint: str
    ) -> DecisionRecord | None:
        """Most recent bought decision for this token still awaiting an outcome."""
        for d in reversed(decisions):
            if d.token_mint == token_mint and d.was_bought and d.outcome_pnl_pct is None:
                return d
        return None

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
                hold = (
                    f" in {d.outcome_hold_time_minutes}min" if d.outcome_hold_time_minutes else ""
                )
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

    def build_portfolio_context(self, positions: dict) -> str:
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

    def get_session_stats(self) -> dict:
        """Return aggregate stats for the session."""
        return {
            "session_age_minutes": int((datetime.now() - self.session_start).total_seconds() / 60),
            "total_decisions": len(self.decisions),
            "total_buys": self._total_buys,
            "total_skips": self._total_skips,
            "closed_trades": self._closed_trades,
            "profitable_trades": self._profitable_trades,
            "win_rate": (
                self._profitable_trades / self._closed_trades if self._closed_trades > 0 else 0.0
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
