#!/usr/bin/env python3
"""
FENRIR - Event Types

Strongly-typed event definitions for the event bus.
Every significant action in the bot lifecycle emits an event.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventSeverity(Enum):
    """How important is this event?"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EventCategory(Enum):
    """Broad event categories for filtering."""

    DETECTION = "detection"
    AI = "ai"
    TRADING = "trading"
    POSITION = "position"
    SYSTEM = "system"
    STRATEGY = "strategy"


@dataclass
class TradeEvent:
    """
    A single event in the FENRIR lifecycle.

    Every adapter on the event bus receives these and decides
    what to do with them (log, alert, persist, display, etc.).
    """

    event_type: str
    category: EventCategory
    severity: EventSeverity = EventSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.now)

    # Context (all optional — not every event involves a token or strategy)
    token_address: str | None = None
    token_symbol: str | None = None
    strategy_id: str | None = None

    # Flexible payload for event-specific data
    data: dict = field(default_factory=dict)

    # Human-readable summary (adapters can use this directly)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "token_address": self.token_address,
            "token_symbol": self.token_symbol,
            "strategy_id": self.strategy_id,
            "data": self.data,
            "message": self.message,
        }


# ═══════════════════════════════════════════════════════════════════════════
#                        EVENT FACTORY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def token_detected_event(
    token_address: str,
    symbol: str,
    name: str,
    liquidity_sol: float,
    market_cap_sol: float,
    creator: str | None = None,
) -> TradeEvent:
    return TradeEvent(
        event_type="TOKEN_DETECTED",
        category=EventCategory.DETECTION,
        severity=EventSeverity.INFO,
        token_address=token_address,
        token_symbol=symbol,
        data={
            "name": name,
            "liquidity_sol": liquidity_sol,
            "market_cap_sol": market_cap_sol,
            "creator": creator,
        },
        message=f"New launch: ${symbol} ({name}) | Liq: {liquidity_sol:.2f} SOL",
    )


def ai_decision_event(
    token_address: str,
    symbol: str,
    decision: str,
    confidence: float,
    risk_score: float,
    reasoning: str,
    strategy_id: str | None = None,
    elapsed_ms: float = 0.0,
) -> TradeEvent:
    return TradeEvent(
        event_type="AI_DECISION",
        category=EventCategory.AI,
        severity=EventSeverity.INFO,
        token_address=token_address,
        token_symbol=symbol,
        strategy_id=strategy_id,
        data={
            "decision": decision,
            "confidence": confidence,
            "risk_score": risk_score,
            "reasoning": reasoning[:200],
            "elapsed_ms": elapsed_ms,
        },
        message=(
            f"AI: {decision} ${symbol} "
            f"(conf={confidence:.0%}, risk={risk_score:.1f}/10, {elapsed_ms:.0f}ms)"
        ),
    )


def ai_override_event(
    token_address: str,
    symbol: str,
    mechanical_trigger: str,
    reasoning: str,
    strategy_id: str | None = None,
) -> TradeEvent:
    return TradeEvent(
        event_type="AI_OVERRIDE",
        category=EventCategory.AI,
        severity=EventSeverity.WARNING,
        token_address=token_address,
        token_symbol=symbol,
        strategy_id=strategy_id,
        data={
            "mechanical_trigger": mechanical_trigger,
            "reasoning": reasoning[:200],
        },
        message=f"AI OVERRIDE: Holding ${symbol} despite '{mechanical_trigger}'",
    )


def buy_executed_event(
    token_address: str,
    symbol: str,
    amount_sol: float,
    entry_price: float,
    signature: str | None = None,
    simulation: bool = False,
    strategy_id: str | None = None,
) -> TradeEvent:
    mode = "SIM " if simulation else ""
    return TradeEvent(
        event_type="BUY_EXECUTED",
        category=EventCategory.TRADING,
        severity=EventSeverity.CRITICAL if not simulation else EventSeverity.INFO,
        token_address=token_address,
        token_symbol=symbol,
        strategy_id=strategy_id,
        data={
            "amount_sol": amount_sol,
            "entry_price": entry_price,
            "signature": signature,
            "simulation": simulation,
        },
        message=f"{mode}BUY: {amount_sol:.4f} SOL -> ${symbol} @ {entry_price:.10f}",
    )


def sell_executed_event(
    token_address: str,
    symbol: str,
    pnl_pct: float,
    pnl_sol: float,
    reason: str,
    hold_minutes: int = 0,
    signature: str | None = None,
    simulation: bool = False,
    strategy_id: str | None = None,
) -> TradeEvent:
    emoji = "+" if pnl_pct > 0 else ""
    mode = "SIM " if simulation else ""
    return TradeEvent(
        event_type="SELL_EXECUTED",
        category=EventCategory.TRADING,
        severity=EventSeverity.CRITICAL if not simulation else EventSeverity.INFO,
        token_address=token_address,
        token_symbol=symbol,
        strategy_id=strategy_id,
        data={
            "pnl_pct": pnl_pct,
            "pnl_sol": pnl_sol,
            "reason": reason,
            "hold_minutes": hold_minutes,
            "signature": signature,
            "simulation": simulation,
        },
        message=(
            f"{mode}SELL: ${symbol} | {emoji}{pnl_pct:.2f}% "
            f"({emoji}{pnl_sol:.4f} SOL) | {reason}"
        ),
    )


def trade_failed_event(
    token_address: str,
    symbol: str,
    trade_type: str,
    error: str,
    strategy_id: str | None = None,
) -> TradeEvent:
    return TradeEvent(
        event_type="TRADE_FAILED",
        category=EventCategory.TRADING,
        severity=EventSeverity.WARNING,
        token_address=token_address,
        token_symbol=symbol,
        strategy_id=strategy_id,
        data={"trade_type": trade_type, "error": error},
        message=f"{trade_type} FAILED for ${symbol}: {error}",
    )


def position_update_event(
    token_address: str,
    symbol: str,
    pnl_pct: float,
    current_price: float,
    hold_minutes: int,
    strategy_id: str | None = None,
) -> TradeEvent:
    emoji = "+" if pnl_pct >= 0 else ""
    return TradeEvent(
        event_type="POSITION_UPDATE",
        category=EventCategory.POSITION,
        severity=EventSeverity.DEBUG,
        token_address=token_address,
        token_symbol=symbol,
        strategy_id=strategy_id,
        data={
            "pnl_pct": pnl_pct,
            "current_price": current_price,
            "hold_minutes": hold_minutes,
        },
        message=f"${symbol}: {emoji}{pnl_pct:.1f}% ({hold_minutes}min)",
    )


def budget_exhausted_event(
    strategy_id: str,
    budget_sol: float,
    spent_sol: float,
) -> TradeEvent:
    return TradeEvent(
        event_type="BUDGET_EXHAUSTED",
        category=EventCategory.STRATEGY,
        severity=EventSeverity.WARNING,
        strategy_id=strategy_id,
        data={"budget_sol": budget_sol, "spent_sol": spent_sol},
        message=f"Strategy '{strategy_id}' budget exhausted: {spent_sol:.4f}/{budget_sol:.4f} SOL",
    )


def bot_lifecycle_event(
    action: str,
    details: dict | None = None,
) -> TradeEvent:
    return TradeEvent(
        event_type=f"BOT_{action.upper()}",
        category=EventCategory.SYSTEM,
        severity=EventSeverity.INFO,
        data=details or {},
        message=f"FENRIR {action.lower()}",
    )


def error_event(
    context: str,
    error: str,
    token_address: str | None = None,
    strategy_id: str | None = None,
) -> TradeEvent:
    return TradeEvent(
        event_type="ERROR",
        category=EventCategory.SYSTEM,
        severity=EventSeverity.WARNING,
        token_address=token_address,
        strategy_id=strategy_id,
        data={"context": context, "error": error},
        message=f"ERROR in {context}: {error}",
    )
