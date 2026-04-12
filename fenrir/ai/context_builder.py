#!/usr/bin/env python3
"""
FENRIR - Batched Position Context Builder

Builds structured JSON context payloads for multi-position LLM exit evaluation,
following Nocturne's batched-asset context pattern.

Key features
────────────
• Serializes all open positions: prices, PnL, hold time, peak/drawdown
• Injects AI-written exit plans from prior cycles (Nocturne's stateful replay)
• Parses "cooldown_until: <ISO>" markers and skips positions still in cooldown
• Produces a single context JSON passed as the LLM user message

Cooldown pattern (from Nocturne)
─────────────────────────────────
The AI writes its own continuation contract into exit_plan, e.g.:

    "Hold while RSI14 > 40 and drawdown < 35%. cooldown_until: 2025-10-19T16:00Z"

On the next evaluation cycle, apply_exit_plan_to_position() persists this
onto the Position object. build_batched_exit_context() then:
  - Skips positions where now() < cooldown_until
  - Re-injects prior_ai_exit_plan for positions not in cooldown so Claude
    can decide whether its own conditions have been met or invalidated

Usage
─────
    # After an exit decision cycle:
    for decision in exit_decisions:
        pos = positions[decision["token_address"]]
        apply_exit_plan_to_position(pos, decision["exit_plan"])

    # On the next cycle:
    context_json, active_addrs = build_batched_exit_context(
        positions=position_manager.positions,
        portfolio_summary=position_manager.get_portfolio_summary(),
        wallet_balance_sol=wallet.balance,
        triggered_exits={"ABC...": "Trailing Stop: -35%"},
        session_memory_block=brain.memory.build_context_block(),
    )
    # Pass context_json to ClaudeBrain.evaluate_exits_batched()
"""

import json
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  Cooldown helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def parse_cooldown_from_exit_plan(exit_plan: str | None) -> datetime | None:
    """
    Extract a cooldown timestamp from an AI-written exit_plan string.

    Recognises the Nocturne-style marker:
        "cooldown_until: 2025-10-19T15:55Z"
        "cooldown_until: 2025-10-19T15:55:00+00:00"

    Returns:
        datetime (UTC-aware) if found, else None.
    """
    if not exit_plan:
        return None
    lower = exit_plan.lower()
    marker = "cooldown_until:"
    idx = lower.find(marker)
    if idx == -1:
        return None
    # Grab the first token after the colon
    raw_ts = exit_plan[idx + len(marker):].strip().split()[0].rstrip(",;.")
    try:
        # Normalise trailing Z → +00:00
        if raw_ts.endswith("Z"):
            raw_ts = raw_ts[:-1] + "+00:00"
        return datetime.fromisoformat(raw_ts)
    except ValueError:
        return None


def is_in_cooldown(position: Any) -> bool:
    """
    Return True if the position has an active AI-imposed cooldown that has
    not yet expired.

    Looks for `ai_cooldown_until` attribute on the Position object.
    """
    cooldown_until = getattr(position, "ai_cooldown_until", None)
    if cooldown_until is None:
        return False
    return _now_utc() < _ensure_utc(cooldown_until)


def apply_exit_plan_to_position(position: Any, exit_plan: str) -> None:
    """
    Persist an AI-written exit_plan onto a Position and parse any cooldown marker.

    Mutates:
        position.ai_exit_plan       (str | None)
        position.ai_cooldown_until  (datetime | None)

    Call this after receiving exit decisions from ClaudeBrain.evaluate_exits_batched().
    """
    position.ai_exit_plan = exit_plan
    position.ai_cooldown_until = parse_cooldown_from_exit_plan(exit_plan)


# ─────────────────────────────────────────────────────────────────────────────
#  Context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_batched_exit_context(
    positions: dict[str, Any],
    portfolio_summary: dict,
    wallet_balance_sol: float = 0.0,
    recent_diary: list | None = None,
    session_memory_block: str = "",
    triggered_exits: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """
    Build a single JSON context string for batched exit evaluation.

    Args:
        positions:            Dict of token_address → Position objects
                              (from PositionManager.positions).
        portfolio_summary:    From PositionManager.get_portfolio_summary().
        wallet_balance_sol:   Available wallet balance in SOL.
        recent_diary:         Optional list of recent diary/audit entries.
        session_memory_block: AISessionMemory.build_context_block() output.
        triggered_exits:      Dict of token_address → mechanical trigger string
                              for positions where a rule-based exit has fired.
                              The AI can choose to OVERRIDE_HOLD these.

    Returns:
        (context_json_str, active_addresses)

        context_json_str:  JSON string to use as the LLM user message.
        active_addresses:  Ordered list of token addresses included in this
                           context (excludes positions still in AI cooldown).
                           Use this list to map exit_decisions back to positions.
    """
    triggered_exits = triggered_exits or {}
    now_iso = _now_utc().isoformat()

    position_sections: list[dict] = []
    active_addresses: list[str] = []

    for addr, pos in positions.items():
        # Skip positions whose AI-imposed cooldown has not expired
        if is_in_cooldown(pos):
            continue

        # Hold time since entry
        entry_time = getattr(pos, "entry_time", _now_utc())
        hold_minutes = int(
            (_now_utc() - _ensure_utc(entry_time)).total_seconds() / 60
        )

        # Peak and drawdown
        peak_price = getattr(pos, "peak_price", pos.entry_price) or pos.entry_price
        drawdown_pct = (
            (peak_price - pos.current_price) / peak_price * 100
            if peak_price > 0
            else 0.0
        )

        section: dict = {
            "token_address": addr,
            "symbol": getattr(pos, "token_symbol", "???"),
            "strategy_id": getattr(pos, "strategy_id", "default"),
            # Prices
            "entry_price": round(pos.entry_price, 10),
            "current_price": round(pos.current_price, 10),
            "peak_price": round(peak_price, 10),
            # Performance
            "pnl_pct": round(pos.get_pnl_percent(), 2),
            "pnl_sol": round(pos.get_pnl_sol(), 4),
            "drawdown_from_peak_pct": round(drawdown_pct, 2),
            "hold_minutes": hold_minutes,
            # Size
            "amount_sol_invested": round(pos.amount_sol_invested, 4),
        }

        # ── Nocturne pattern: replay prior AI exit plan ──────────────────
        # If Claude wrote an exit_plan last cycle, inject it so it can
        # check whether its own conditions have been met or invalidated.
        prior_exit_plan = getattr(pos, "ai_exit_plan", None)
        if prior_exit_plan:
            section["prior_ai_exit_plan"] = prior_exit_plan

        # ── Mechanical trigger: give AI override opportunity ─────────────
        trigger = triggered_exits.get(addr)
        if trigger:
            section["mechanical_trigger_fired"] = trigger

        position_sections.append(section)
        active_addresses.append(addr)

    payload = OrderedDict([
        ("current_time", now_iso),
        ("account", {
            "wallet_balance_sol": round(wallet_balance_sol, 4),
            "num_positions": portfolio_summary.get("num_positions", len(positions)),
            "total_invested_sol": round(
                portfolio_summary.get("total_invested_sol", 0.0), 4
            ),
            "current_value_sol": round(
                portfolio_summary.get("current_value_sol", 0.0), 4
            ),
            "total_pnl_sol": round(portfolio_summary.get("total_pnl_sol", 0.0), 4),
            "total_pnl_pct": round(portfolio_summary.get("total_pnl_pct", 0.0), 2),
        }),
        ("positions", position_sections),
        ("session_context", session_memory_block or ""),
        ("recent_diary", (recent_diary or [])[-10:]),
        ("instructions", {
            "task": "Evaluate exit actions for all listed positions.",
            "token_addresses": active_addresses,
            "requirement": (
                "Return one exit_decision per token_address in the order listed. "
                "For each, write an exit_plan that encodes your hold conditions and "
                "any self-imposed cooldown using the format: "
                "'cooldown_until: <ISO timestamp UTC>'. "
                "If a mechanical_trigger_fired is present, you may OVERRIDE_HOLD to "
                "keep the position open — explain clearly why the trigger is premature. "
                "Respect prior_ai_exit_plan unless its conditions have been invalidated."
            ),
        }),
    ])

    return json.dumps(payload, default=str), active_addresses
