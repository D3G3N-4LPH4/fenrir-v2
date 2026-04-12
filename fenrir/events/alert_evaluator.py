#!/usr/bin/env python3
"""
FENRIR - Alert Evaluator

Port of Crucix's alert evaluation logic.

Implements a three-tier alert system (FLASH / PRIORITY / ROUTINE) with:
  - Rule-based evaluator (fast path, no LLM required)
  - LLM evaluation prompt builder (optional LLM enhancement)
  - JSON parsing helpers for LLM responses

Tier definitions:
  FLASH    — Immediate action required (cooldown: 5min, max 6/hr)
  PRIORITY — Important but non-urgent (cooldown: 30min, max 4/hr)
  ROUTINE  — Informational context (cooldown: 60min, max 2/hr)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─── Tier configuration ────────────────────────────────────────────────────────

TIER_CONFIG: dict[str, dict] = {
    "FLASH": {
        "cooldown_s": 300,       # 5 minutes
        "max_per_hour": 6,
        "emoji": "🚨",
        "label": "FLASH ALERT",
    },
    "PRIORITY": {
        "cooldown_s": 1800,      # 30 minutes
        "max_per_hour": 4,
        "emoji": "⚡",
        "label": "PRIORITY ALERT",
    },
    "ROUTINE": {
        "cooldown_s": 3600,      # 60 minutes
        "max_per_hour": 2,
        "emoji": "📊",
        "label": "ROUTINE UPDATE",
    },
}

# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """A single trading signal for alert evaluation."""
    key: str
    label: str
    severity: str = "moderate"    # "critical" | "high" | "moderate" | "low"
    direction: str = "new"        # "new" | "escalating" | "improving" | "resolving"
    value: Any = None             # Current value
    prev_value: Any = None        # Previous value
    pct_change: float | None = None
    text: str | None = None       # Free-text description (for social/news signals)
    token_address: str | None = None
    source: str | None = None     # "on_chain" | "social" | "smc" | "system"
    extra: dict = field(default_factory=dict)


@dataclass
class AlertEvaluation:
    """Result of alert evaluation — whether to alert and how."""
    should_alert: bool
    tier: str                     # "FLASH" | "PRIORITY" | "ROUTINE"
    headline: str
    reason: str
    confidence: str = "MEDIUM"   # "HIGH" | "MEDIUM" | "LOW"
    actionable: str = "Monitor"
    cross_correlation: str | None = None
    suppress_reason: str | None = None
    signals: list[str] = field(default_factory=list)
    token_address: str | None = None
    source: str = "rules"         # "rules" | "llm"


# ─── Rule-based evaluator ──────────────────────────────────────────────────────

class RuleBasedEvaluator:
    """
    Five-rule decision tree for alert tier classification.

    Rule priority (first match wins):
      1. Safety signal (rug/scam/dump keywords)  → FLASH
      2. Cross-domain critical (multiple critical signals) → FLASH
      3. Same-direction escalation               → PRIORITY
      4. Social surge (high engagement bearish)  → PRIORITY
      5. Routine metrics update                  → ROUTINE
    """

    # Safety keywords that warrant immediate FLASH alert
    SAFETY_KEYWORDS = frozenset([
        "rug", "rugpull", "scam", "honeypot", "dump", "dumping",
        "dev sold", "dev dumped", "exit liquidity", "bundled", "bundle",
    ])

    def evaluate(
        self,
        signals: list[Signal],
        delta_direction: str = "mixed",
    ) -> AlertEvaluation:
        """
        Evaluate a batch of signals and return an AlertEvaluation.

        Args:
            signals:         List of Signal objects to evaluate.
            delta_direction: "risk-on" | "risk-off" | "mixed"
        """
        if not signals:
            return AlertEvaluation(
                should_alert=False,
                tier="ROUTINE",
                headline="No signals",
                reason="Empty signal batch",
                suppress_reason="no_signals",
            )

        critical_sigs = [s for s in signals if s.severity == "critical"]
        high_sigs = [s for s in signals if s.severity == "high"]
        escalating = [s for s in signals if s.direction in ("escalating", "new")]

        # Rule 1: Safety signal
        safety = self._find_safety_signal(signals)
        if safety:
            return AlertEvaluation(
                should_alert=True,
                tier="FLASH",
                headline=f"Safety signal: {safety.label}",
                reason=safety.text or f"Safety keyword detected in {safety.source or 'signal'}",
                confidence="HIGH",
                actionable="Consider immediate exit",
                signals=[s.label for s in signals[:5]],
                source="rules",
            )

        # Rule 2: Cross-domain critical (2+ critical signals from different sources)
        if len(critical_sigs) >= 2:
            sources = {s.source for s in critical_sigs if s.source}
            if len(sources) >= 2:
                return AlertEvaluation(
                    should_alert=True,
                    tier="FLASH",
                    headline=f"Cross-domain critical: {len(critical_sigs)} signals",
                    reason=f"Critical signals from: {', '.join(sources)}",
                    confidence="HIGH",
                    actionable="Review all positions",
                    cross_correlation=f"{len(sources)} domains affected",
                    signals=[s.label for s in critical_sigs[:5]],
                    source="rules",
                )

        # Rule 3: Same-direction escalation (multiple escalating signals)
        if len(escalating) >= 3 and delta_direction in ("risk-off", "mixed"):
            return AlertEvaluation(
                should_alert=True,
                tier="PRIORITY",
                headline=f"Escalation: {len(escalating)} signals ({delta_direction})",
                reason=f"{len(escalating)} escalating/new signals in {delta_direction} regime",
                confidence="MEDIUM",
                actionable="Tighten stops",
                signals=[s.label for s in escalating[:5]],
                source="rules",
            )

        # Rule 4: High-severity signal
        if high_sigs:
            return AlertEvaluation(
                should_alert=True,
                tier="PRIORITY",
                headline=f"High severity: {high_sigs[0].label}",
                reason=high_sigs[0].text or f"High severity signal: {high_sigs[0].key}",
                confidence="MEDIUM",
                actionable="Monitor closely",
                signals=[s.label for s in high_sigs[:3]],
                source="rules",
            )

        # Rule 5: Routine update
        if signals:
            return AlertEvaluation(
                should_alert=True,
                tier="ROUTINE",
                headline=f"Update: {signals[0].label}",
                reason=f"{len(signals)} signal(s) — routine context update",
                confidence="LOW",
                actionable="Monitor",
                signals=[s.label for s in signals[:3]],
                source="rules",
            )

        return AlertEvaluation(
            should_alert=False,
            tier="ROUTINE",
            headline="No actionable signals",
            reason="Signals present but below alert threshold",
            suppress_reason="below_threshold",
        )

    def _find_safety_signal(self, signals: list[Signal]) -> Signal | None:
        """Check if any signal contains safety/rug keywords."""
        for sig in signals:
            text = (sig.text or sig.label or "").lower()
            if any(kw in text for kw in self.SAFETY_KEYWORDS):
                return sig
            # Also check extra dict for rug-related keys
            if sig.extra.get("bear_warnings") or sig.extra.get("rug_detected"):
                return sig
        return None


# ─── LLM prompt builder ───────────────────────────────────────────────────────

class LLMEvaluationPrompt:
    """
    Builds system + user prompts for LLM-based alert evaluation.
    The LLM response should be valid JSON matching the schema below.
    """

    SYSTEM = (
        "You are FENRIR's alert evaluation engine. Your job is to decide whether "
        "a batch of trading signals warrants sending a Telegram alert, and if so, "
        "what tier and framing to use.\n\n"
        "Respond ONLY with valid JSON matching this schema:\n"
        "{\n"
        '  "shouldAlert": boolean,\n'
        '  "tier": "FLASH" | "PRIORITY" | "ROUTINE",\n'
        '  "headline": "short headline (max 80 chars)",\n'
        '  "reason": "1-2 sentence explanation",\n'
        '  "confidence": "HIGH" | "MEDIUM" | "LOW",\n'
        '  "actionable": "specific action or Monitor",\n'
        '  "crossCorrelation": "optional cross-signal pattern or null",\n'
        '  "signals": ["list", "of", "key", "signals"]\n'
        "}"
    )

    @staticmethod
    def build_user_message(
        signals: list[Signal],
        delta_direction: str,
        token_address: str | None = None,
    ) -> str:
        """Build the user message for LLM evaluation."""
        lines = [
            f"Delta direction: {delta_direction}",
            f"Token: {token_address or 'portfolio-wide'}",
            f"Signal count: {len(signals)}",
            "",
            "Signals:",
        ]
        for sig in signals[:10]:  # cap at 10 to keep prompt compact
            parts = [f"  [{sig.severity.upper()}] {sig.label}"]
            if sig.direction != "new":
                parts.append(f"({sig.direction})")
            if sig.text:
                parts.append(f"— {sig.text[:100]}")
            elif sig.value is not None:
                val_str = f"{sig.prev_value} → {sig.value}"
                if sig.pct_change is not None:
                    val_str += f" ({sig.pct_change:+.1f}%)"
                parts.append(f"— {val_str}")
            lines.append(" ".join(parts))

        if len(signals) > 10:
            lines.append(f"  ... and {len(signals) - 10} more signals")

        lines += [
            "",
            "Evaluate: should FENRIR send a Telegram alert for this batch?",
            "Consider: urgency, actionability, and alert fatigue.",
        ]
        return "\n".join(lines)


# ─── LLM response parsing ─────────────────────────────────────────────────────

def parse_llm_json(text: str | None) -> dict | None:
    """
    Triple-fallback JSON parser for LLM responses.

    Attempt 1: Direct json.loads()
    Attempt 2: Strip markdown fences then parse
    Attempt 3: Extract outermost {} and parse
    """
    if not text:
        return None

    clean = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown fences
    for fence in ("```json", "```"):
        if fence in clean:
            clean = clean.replace(fence, "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Attempt 3: outermost braces
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("parse_llm_json: all attempts failed. Preview: %.120s", text)
    return None


def llm_response_to_evaluation(
    parsed: dict,
    source: str = "llm",
) -> AlertEvaluation:
    """
    Convert a parsed LLM JSON response to an AlertEvaluation.

    Args:
        parsed: Dict from parse_llm_json()
        source: "llm" or "rules"
    """
    tier = parsed.get("tier", "ROUTINE")
    if tier not in TIER_CONFIG:
        tier = "ROUTINE"

    return AlertEvaluation(
        should_alert=bool(parsed.get("shouldAlert", False)),
        tier=tier,
        headline=str(parsed.get("headline", ""))[:120],
        reason=str(parsed.get("reason", "")),
        confidence=parsed.get("confidence", "MEDIUM"),
        actionable=parsed.get("actionable", "Monitor"),
        cross_correlation=parsed.get("crossCorrelation"),
        signals=parsed.get("signals", []),
        source=source,
    )
