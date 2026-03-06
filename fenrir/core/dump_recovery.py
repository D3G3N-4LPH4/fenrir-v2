#!/usr/bin/env python3
"""
FENRIR - Post-Dump Recovery Detector (Ouroboros Pattern)

Detects the "dumps → fake recovery → second dump" pattern that destroys
degen positions. Inspired by OBLITERATUS's Ouroboros effect detection —
the observation that removed guardrails sometimes self-repair through
adjacent pathways. In trading terms: price action that looks like a
recovery is often the second phase of a distribution scheme.

The pattern:
    Entry price: 1.0
    Phase 1 — Dump:      drops to 0.55  (-45%)
    Phase 2 — Recovery:  bounces to 0.70  (+27% from bottom)
    Phase 3 — Second dump: crashes to 0.20 (-71% from entry)

Without detection: trailing stop is set from peak (1.0), recovery bounce
looks healthy, bot holds through the second dump.

With detection: after Phase 1 drop + Phase 2 bounce pattern is recognized,
trailing stop is tightened from 15% → 8% so Phase 3 kicks the exit early.

Integration:
    # In bot.py __init__:
    self.dump_detector = PostDumpRecoveryDetector()

    # In _position_management_loop, after price update:
    for addr, pos in self.positions.positions.items():
        alert = self.dump_detector.update(addr, pos.current_price, pos.entry_price)
        if alert.ouroboros_detected:
            pos.trailing_stop_pct = alert.tightened_trailing_stop_pct
            self.logger.warning(
                f"🐍 OUROBOROS DETECTED {addr[:8]}... "
                f"Trailing stop tightened to {alert.tightened_trailing_stop_pct}%"
            )
            await self.event_bus.emit(ouroboros_detected_event(addr, alert))
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#                           CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OuroborosConfig:
    """
    Tunable thresholds for Ouroboros pattern detection.

    Defaults are calibrated for pump.fun tokens, which tend to have
    sharper dumps and shallower recoveries than established tokens.
    """

    # Phase 1: Minimum drawdown from peak to qualify as a "dump"
    dump_threshold_pct: float = 30.0

    # Phase 2: Minimum recovery from the dump bottom to flag as "fake recovery"
    # (a real recovery would be deeper; a dead-cat bounce is typically 10-30%)
    recovery_threshold_pct: float = 10.0

    # Phase 2: Maximum recovery that still counts as "fake"
    # (if it fully recovers, it's probably legitimate)
    max_recovery_to_flag_pct: float = 60.0

    # How many price ticks to look back for the pattern
    lookback_ticks: int = 30

    # Trailing stop to switch to when Ouroboros detected (percentage)
    tightened_trailing_stop_pct: float = 8.0

    # Cooldown: once Ouroboros fires for a position, don't re-trigger for N ticks
    cooldown_ticks: int = 5


# ═══════════════════════════════════════════════════════════════════════════
#                           DETECTION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OuroborosAlert:
    """Result of an Ouroboros check for a single price update."""

    ouroboros_detected: bool = False

    # Price geometry when detected
    entry_price: float = 0.0
    dump_low: float = 0.0
    recovery_high: float = 0.0
    dump_pct: float = 0.0        # How far it dropped from peak
    recovery_pct: float = 0.0   # How much it bounced from the low

    # Recommended response
    tightened_trailing_stop_pct: float = 8.0

    # Human-readable summary
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "ouroboros_detected": self.ouroboros_detected,
            "entry_price": self.entry_price,
            "dump_low": self.dump_low,
            "recovery_high": self.recovery_high,
            "dump_pct": round(self.dump_pct, 2),
            "recovery_pct": round(self.recovery_pct, 2),
            "tightened_trailing_stop_pct": self.tightened_trailing_stop_pct,
            "message": self.message,
        }


# ═══════════════════════════════════════════════════════════════════════════
#                           DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _PositionState:
    """Internal state tracked per position."""
    price_history: deque = field(default_factory=lambda: deque(maxlen=50))
    peak_price: float = 0.0
    dump_low: float = float("inf")
    in_dump_phase: bool = False
    triggered: bool = False  # Has Ouroboros already fired?
    cooldown_remaining: int = 0
    last_detected_at: datetime | None = None


class PostDumpRecoveryDetector:
    """
    Per-position Ouroboros pattern detector.

    Maintains a rolling price history for each open position and
    checks for the dump → recovery → second-dump structure on
    every price update.

    Thread-safe: uses per-position state, no shared mutable state.
    """

    def __init__(self, config: OuroborosConfig | None = None):
        self.config = config or OuroborosConfig()
        self._states: dict[str, _PositionState] = {}
        self._detections_total: int = 0

    def update(
        self,
        token_address: str,
        current_price: float,
        entry_price: float,
    ) -> OuroborosAlert:
        """
        Feed a new price tick for a position and check for Ouroboros.

        Args:
            token_address: Position mint address (used as key)
            current_price: Latest price from price feed
            entry_price: Original entry price (for context in alert)

        Returns:
            OuroborosAlert — check .ouroboros_detected to act
        """
        if token_address not in self._states:
            self._states[token_address] = _PositionState(peak_price=entry_price)

        state = self._states[token_address]

        # Guard: skip if cooldown active
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
            return OuroborosAlert(ouroboros_detected=False)

        # Guard: skip if already triggered (one alert per position)
        if state.triggered:
            return OuroborosAlert(ouroboros_detected=False)

        # Record price
        state.price_history.append(current_price)

        # Update peak
        if current_price > state.peak_price:
            state.peak_price = current_price
            # If price returns to peak, the dump phase is over (legit recovery)
            state.in_dump_phase = False
            state.dump_low = float("inf")

        # Need at least a few ticks to have a pattern
        if len(state.price_history) < 5:
            return OuroborosAlert(ouroboros_detected=False)

        cfg = self.config

        # ── Phase 1 Detection: Is there a significant dump from peak? ──
        drawdown_from_peak = (
            (state.peak_price - current_price) / state.peak_price
        ) * 100

        if drawdown_from_peak >= cfg.dump_threshold_pct:
            # We're in a dump phase — track the low
            state.in_dump_phase = True
            if current_price < state.dump_low:
                state.dump_low = current_price

        # ── Phase 2 Detection: Is there a recovery from the dump low? ──
        if state.in_dump_phase and state.dump_low < float("inf"):
            recovery_from_low = (
                (current_price - state.dump_low) / state.dump_low
            ) * 100

            if (
                recovery_from_low >= cfg.recovery_threshold_pct
                and recovery_from_low <= cfg.max_recovery_to_flag_pct
            ):
                # Pattern confirmed: significant dump + partial recovery
                # This is the Ouroboros setup — second dump likely incoming

                dump_pct = (
                    (state.peak_price - state.dump_low) / state.peak_price
                ) * 100

                state.triggered = True
                state.cooldown_remaining = cfg.cooldown_ticks
                state.last_detected_at = datetime.now()
                self._detections_total += 1

                alert = OuroborosAlert(
                    ouroboros_detected=True,
                    entry_price=entry_price,
                    dump_low=state.dump_low,
                    recovery_high=current_price,
                    dump_pct=dump_pct,
                    recovery_pct=recovery_from_low,
                    tightened_trailing_stop_pct=cfg.tightened_trailing_stop_pct,
                    message=(
                        f"Ouroboros pattern: -{dump_pct:.1f}% dump from peak, "
                        f"+{recovery_from_low:.1f}% fake recovery. "
                        f"Tightening trailing stop to {cfg.tightened_trailing_stop_pct}%."
                    ),
                )

                logger.warning(
                    f"🐍 Ouroboros on {token_address[:8]}...: "
                    f"dump={dump_pct:.1f}% recovery={recovery_from_low:.1f}% "
                    f"→ trailing stop → {cfg.tightened_trailing_stop_pct}%"
                )
                return alert

        return OuroborosAlert(ouroboros_detected=False)

    def remove_position(self, token_address: str) -> None:
        """Clean up state when a position is closed."""
        self._states.pop(token_address, None)

    def get_stats(self) -> dict:
        """Summary of detector activity."""
        return {
            "positions_tracked": len(self._states),
            "total_detections": self._detections_total,
            "currently_triggered": sum(
                1 for s in self._states.values() if s.triggered
            ),
        }

    def reset_for_position(self, token_address: str) -> None:
        """
        Reset detection state for a position (e.g. after AI override-hold).
        Allows re-detection if the pattern repeats.
        """
        if token_address in self._states:
            state = self._states[token_address]
            state.triggered = False
            state.in_dump_phase = False
            state.dump_low = float("inf")
            state.cooldown_remaining = 0


# ═══════════════════════════════════════════════════════════════════════════
#                           EVENT FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def ouroboros_detected_event(
    token_address: str,
    alert: OuroborosAlert,
    symbol: str = "???",
    strategy_id: str | None = None,
):
    """
    Create a TradeEvent for emission on the EventBus.

    Usage:
        await self.event_bus.emit(
            ouroboros_detected_event(addr, alert, symbol=pos.token_symbol)
        )
    """
    from fenrir.events.types import EventCategory, EventSeverity, TradeEvent

    return TradeEvent(
        event_type="OUROBOROS_DETECTED",
        category=EventCategory.POSITION,
        severity=EventSeverity.WARNING,
        token_address=token_address,
        token_symbol=symbol,
        strategy_id=strategy_id,
        data=alert.to_dict(),
        message=alert.message,
    )
