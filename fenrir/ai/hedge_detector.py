#!/usr/bin/env python3
"""
FENRIR - HedgeDetector: STM Output Normalization for AIHealthMonitor

G0DM0D3 lineage: G0DM0D3 observed that Claude occasionally returns hedged
scoring language ("I think this might be a good buy", "perhaps the confidence
is approximately 0.7") that causes two distinct problems:

  1. PARSING LIABILITY — Hedges inside or around JSON blocks cause json.loads()
     to fail or produce garbage values, silently degrading to conservative
     defaults and reducing the AI's effective trade rate.

  2. DRIFT SIGNAL — Increasing hedge frequency correlates with model
     distribution shift. When Claude is uncertain about its own outputs
     (epistemic self-censorship), hedge rate rises measurably before other
     drift metrics trip. It is the canary in the coal mine.

This module provides:
  - HedgeDetector: strips hedges from raw LLM responses before JSON extraction
  - Rolling hedge rate tracking (window=10) for linguistic drift detection
  - AI_DRIFT event emission when rolling rate > 0.6

Integration:
    detector = HedgeDetector()

    # In AITradingAnalyst._call_llm / _parse_llm_response:
    cleaned, hedge_count = detector.process(raw_response)
    analysis = _parse_json(cleaned)

    # hedge_count is passed to AI_DECISION events so AIHealthMonitor
    # can drive the 7th drift detector (linguistic_hedge_drift).
    if detector.is_drifting():
        await bus.emit(ai_drift_event(...))
"""

from __future__ import annotations

import re
from collections import deque

# ─────────────────────────────────────────────────────────────
#  Pattern library
# ─────────────────────────────────────────────────────────────

# Epistemic hedge phrases — stripped from response text.
# Ordered from most to least specific to avoid partial-match side-effects.
_HEDGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bI\s+think\b",          re.I),
    re.compile(r"\bI\s+believe\b",        re.I),
    re.compile(r"\bI\s+feel\b",           re.I),
    re.compile(r"\bI\s+suspect\b",        re.I),
    re.compile(r"\bperhaps\b",            re.I),
    re.compile(r"\bmaybe\b",              re.I),
    re.compile(r"\bit\s+seems?\b",        re.I),
    re.compile(r"\bseems?\s+to\b",        re.I),
    re.compile(r"\bpotentially\b",        re.I),
    re.compile(r"\bmight\b",              re.I),
    re.compile(r"\bcould\s+be\b",         re.I),
    re.compile(r"\bapproximately\b",      re.I),
    re.compile(r"\bsomewhat\b",           re.I),
    re.compile(r"\brather\b",             re.I),
    re.compile(r"\bfairly\b",             re.I),
    re.compile(r"\bsort\s+of\b",          re.I),
    re.compile(r"\bkind\s+of\b",          re.I),
    re.compile(r"\baround\b",             re.I),  # "around 0.7"
    re.compile(r"\babout\b",              re.I),  # "about 75 score"
]

# Preamble phrases that precede the actual analysis.
# These are typically sentence-initial and safe to remove entirely.
_PREAMBLE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^Based\s+on\b[^,\n]*,?\s*",        re.I | re.M),
    re.compile(r"^Looking\s+at\b[^,\n]*,?\s*",       re.I | re.M),
    re.compile(r"^Given\s+the\b[^,\n]*,?\s*",        re.I | re.M),
    re.compile(r"^Considering\s+that\b[^,\n]*,?\s*", re.I | re.M),
    re.compile(r"^Analyzing\s+the\b[^,\n]*,?\s*",    re.I | re.M),
    re.compile(r"^From\s+the\b[^,\n]*,?\s*",         re.I | re.M),
    re.compile(r"^Taking\s+into\s+account\b[^,\n]*,?\s*", re.I | re.M),
    re.compile(r"^After\s+(reviewing|analyzing|considering)\b[^,\n]*,?\s*", re.I | re.M),
]


# ─────────────────────────────────────────────────────────────
#  HedgeDetector
# ─────────────────────────────────────────────────────────────


class HedgeDetector:
    """
    Post-processor for raw LLM responses.

    Strips epistemic hedges and preamble phrases before JSON extraction.
    Tracks rolling hedge rate (last 10 responses) as a linguistic drift
    signal for AIHealthMonitor's 7th detector.

    G0DM0D3 lineage: Hedged output degrades both JSON parsing reliability
    and the quality of confidence signals fed to position sizing. Rising
    hedge rate precedes other drift signals, making it an early-warning
    indicator for model quality degradation.

    This class is NOT async and is safe to call from any context.
    It is intentionally stateful (rolling window) — instantiate once per
    AI session and share across calls.
    """

    WINDOW_SIZE: int = 10
    DRIFT_THRESHOLD: float = 0.6  # > 60% of recent responses hedged → AI_DRIFT

    def __init__(self, window_size: int = WINDOW_SIZE, drift_threshold: float = DRIFT_THRESHOLD):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        # Hedge count per response (> 0 means "hedged")
        self._hedge_counts: deque[int] = deque(maxlen=window_size)

    # ── Public API ────────────────────────────────────────────

    def process(self, response: str) -> tuple[str, int]:
        """
        Strip hedges and preambles from a raw LLM response.

        Args:
            response: Raw text returned by the LLM API.

        Returns:
            (cleaned_text, hedge_count)
            - cleaned_text: Safe to pass to json.loads / _parse_llm_response.
            - hedge_count: Number of hedge phrases removed. Pass this to
              AI_DECISION events so AIHealthMonitor can track linguistic drift.
        """
        cleaned = response
        hedge_count = 0

        # Strip epistemic hedges
        for pattern in _HEDGE_PATTERNS:
            matches = pattern.findall(cleaned)
            if matches:
                hedge_count += len(matches)
                cleaned = pattern.sub("", cleaned)

        # Strip sentence-initial preamble phrases
        for pattern in _PREAMBLE_PATTERNS:
            matches = pattern.findall(cleaned)
            if matches:
                hedge_count += len(matches)
                cleaned = pattern.sub("", cleaned)

        # Normalize whitespace introduced by removals
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

        self._hedge_counts.append(hedge_count)
        return cleaned, hedge_count

    def get_rolling_hedge_rate(self) -> float:
        """
        Fraction of recent responses that contained at least one hedge.

        Used by AIHealthMonitor._detect_linguistic_hedge_drift().
        Returns 0.0 if no responses have been processed yet.
        """
        if not self._hedge_counts:
            return 0.0
        hedged = sum(1 for c in self._hedge_counts if c > 0)
        return hedged / len(self._hedge_counts)

    def get_rolling_avg_hedge_count(self) -> float:
        """Average number of hedge phrases per response in the current window."""
        if not self._hedge_counts:
            return 0.0
        return sum(self._hedge_counts) / len(self._hedge_counts)

    def is_drifting(self) -> bool:
        """
        True when the rolling hedge rate exceeds the drift threshold and
        the window is fully populated.

        Callers should emit an AI_DRIFT event when this returns True.
        """
        return (
            len(self._hedge_counts) >= self.window_size
            and self.get_rolling_hedge_rate() > self.drift_threshold
        )

    def get_stats(self) -> dict:
        """Current detector state — useful for health reports."""
        return {
            "window_size": self.window_size,
            "samples": len(self._hedge_counts),
            "rolling_hedge_rate": round(self.get_rolling_hedge_rate(), 3),
            "avg_hedge_count": round(self.get_rolling_avg_hedge_count(), 2),
            "is_drifting": self.is_drifting(),
            "drift_threshold": self.drift_threshold,
        }

    def reset(self) -> None:
        """Clear the rolling window (e.g., after a strategy restart)."""
        self._hedge_counts.clear()
