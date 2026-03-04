#!/usr/bin/env python3
"""
FENRIR - AI Health Monitor

Observational adapter that sits on the event bus and watches for
AI decision-making drift. Detects degradation patterns that a human
would notice after staring at logs for an hour — but in real time.

Monitored failure modes:
─────────────────────────────────────────────────────────────────
  CONFIDENCE CLUSTERING    Confidence scores cluster near the
                           buy/skip threshold with low variance.
                           The AI is coin-flipping, not analyzing.

  REASONING COLLAPSE       Consecutive evaluations produce nearly
                           identical reasoning text across different
                           tokens. The AI is templating, not thinking.

  SKIP STREAK              Long unbroken run of SKIP decisions during
                           active market conditions (tokens are arriving
                           but nothing gets bought). Overly cautious
                           drift, possibly from a string of losses in
                           historical memory.

  RESPONSE TIME DRIFT      Mean response time rising steadily, which
                           on pump.fun means missed entries. Could
                           indicate prompt bloat or upstream throttling.

  LOSS CASCADE             Consecutive losing trades without any wins.
                           Not an AI bug per se, but a regime signal
                           that should trigger strategy-level caution.

  WIN RATE DECAY           Rolling win rate drops significantly below
                           the session's baseline. Gradual degradation
                           that's hard to spot without aggregation.

Each detector runs independently. When a condition trips, the monitor
emits an AI_HEALTH_WARNING event on the bus (so Telegram, logs, and
the audit chain all see it) and records the alert internally for the
health report API.

Usage:
    monitor = AIHealthMonitor(config)
    bus.register(monitor)

    # Later:
    report = monitor.get_health_report()
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from fenrir.events.bus import EventListener
from fenrir.events.types import EventCategory, EventSeverity, TradeEvent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════


@dataclass
class HealthMonitorConfig:
    """
    Tunable thresholds for drift detection.

    Defaults are calibrated for pump.fun sniping: high-frequency
    token arrivals, fast AI evaluations, short position lifetimes.
    Adjust for slower markets or longer-horizon strategies.
    """

    # ── Rolling window sizes ────────────────────────────────────
    confidence_window: int = 30        # Last N confidence scores
    reasoning_window: int = 15         # Last N reasoning hashes
    response_time_window: int = 30     # Last N response times (ms)
    trade_outcome_window: int = 20     # Last N trade outcomes

    # ── Confidence clustering ───────────────────────────────────
    confidence_stddev_floor: float = 0.06
    # If stddev of recent confidence scores drops below this,
    # the AI isn't discriminating between tokens.
    confidence_cluster_min_samples: int = 10
    # Need at least this many samples before flagging.

    # ── Reasoning collapse ──────────────────────────────────────
    reasoning_similarity_threshold: float = 0.80
    # Fraction of recent reasoning hashes that are duplicates.
    reasoning_collapse_min_samples: int = 8

    # ── Skip streak ─────────────────────────────────────────────
    skip_streak_threshold: int = 15
    # N consecutive SKIPs before flagging. On pump.fun, 15 skips
    # in a row during an active market is suspicious.

    # ── Response time drift ─────────────────────────────────────
    response_time_drift_factor: float = 2.0
    # Flag when recent mean exceeds baseline mean by this factor.
    response_time_min_samples: int = 10

    # ── Loss cascade ────────────────────────────────────────────
    loss_cascade_threshold: int = 5
    # N consecutive losses before warning.

    # ── Win rate decay ──────────────────────────────────────────
    win_rate_decay_threshold: float = 0.15
    # Flag when rolling win rate drops this far below session baseline.
    # e.g., if session baseline is 45% and rolling is 28%, delta = 0.17 > 0.15
    win_rate_min_trades: int = 8
    # Need at least this many trades for meaningful win rate comparison.

    # ── Cooldowns ───────────────────────────────────────────────
    alert_cooldown_seconds: float = 300.0
    # Don't repeat the same alert type within this window.
    # 5 minutes prevents alert fatigue while still catching
    # persistent drift.


# ═══════════════════════════════════════════════════════════════════
#  DATA TYPES
# ═══════════════════════════════════════════════════════════════════


class DriftType(Enum):
    """Categories of AI drift."""

    CONFIDENCE_CLUSTERING = "confidence_clustering"
    REASONING_COLLAPSE = "reasoning_collapse"
    SKIP_STREAK = "skip_streak"
    RESPONSE_TIME_DRIFT = "response_time_drift"
    LOSS_CASCADE = "loss_cascade"
    WIN_RATE_DECAY = "win_rate_decay"


@dataclass
class DriftAlert:
    """A single detected drift event."""

    drift_type: DriftType
    severity: str          # "warning" or "critical"
    message: str
    details: dict
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_id: str | None = None


@dataclass
class _StrategyHealthState:
    """Per-strategy rolling state for drift detection."""

    # Confidence tracking
    confidences: deque  # deque[float]
    decisions: deque    # deque[str]  ("BUY", "STRONG_BUY", "SKIP", etc.)
    reasoning_hashes: deque  # deque[str]  (truncated SHA256)

    # Response time tracking
    response_times_ms: deque  # deque[float]

    # Trade outcome tracking
    trade_outcomes: deque  # deque[bool]  (True = win, False = loss)

    # Fields with defaults must come after all non-default fields
    baseline_response_time_ms: float = 0.0
    baseline_response_time_count: int = 0
    total_wins: int = 0
    total_losses: int = 0

    # Streak counters
    consecutive_skips: int = 0
    consecutive_losses: int = 0

    # Token flow (to distinguish "no tokens arriving" from "skipping everything")
    tokens_seen_since_last_buy: int = 0

    @property
    def total_trades(self) -> int:
        return self.total_wins + self.total_losses

    @property
    def session_win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades

    @property
    def rolling_win_rate(self) -> float:
        if not self.trade_outcomes:
            return 0.0
        wins = sum(1 for w in self.trade_outcomes if w)
        return wins / len(self.trade_outcomes)


def _new_strategy_state(config: HealthMonitorConfig) -> _StrategyHealthState:
    return _StrategyHealthState(
        confidences=deque(maxlen=config.confidence_window),
        decisions=deque(maxlen=config.confidence_window),
        reasoning_hashes=deque(maxlen=config.reasoning_window),
        response_times_ms=deque(maxlen=config.response_time_window),
        trade_outcomes=deque(maxlen=config.trade_outcome_window),
    )


# ═══════════════════════════════════════════════════════════════════
#  HEALTH MONITOR ADAPTER
# ═══════════════════════════════════════════════════════════════════


class AIHealthMonitor(EventListener):
    """
    Event bus adapter that monitors AI decision health.

    Purely observational — never modifies trading behavior.
    When drift is detected, it emits AI_HEALTH_WARNING events
    back onto the bus for other adapters to handle.
    """

    # Subscribe to AI and TRADING events
    categories = {EventCategory.AI, EventCategory.TRADING, EventCategory.DETECTION}
    min_severity = EventSeverity.DEBUG

    def __init__(
        self,
        config: HealthMonitorConfig | None = None,
        event_bus=None,
    ):
        self.config = config or HealthMonitorConfig()
        self._bus = event_bus  # For emitting warnings back onto the bus

        # Per-strategy state
        self._states: dict[str, _StrategyHealthState] = {}

        # Global state (for non-strategy-specific events)
        self._global = _new_strategy_state(self.config)

        # Alert history and cooldowns
        self._alerts: list[DriftAlert] = []
        self._last_alert_time: dict[str, float] = {}  # drift_type:strategy -> timestamp

        # Token arrival tracking (global)
        self._tokens_detected_total: int = 0
        self._monitor_start = time.monotonic()

    def set_event_bus(self, bus) -> None:
        """Set the event bus reference (for emitting warnings)."""
        self._bus = bus

    def _get_state(self, strategy_id: str | None) -> _StrategyHealthState:
        """Get or create state for a strategy."""
        if strategy_id is None:
            return self._global
        if strategy_id not in self._states:
            self._states[strategy_id] = _new_strategy_state(self.config)
        return self._states[strategy_id]

    # ───────────────────────────────────────────────────────────
    #  EVENT HANDLER
    # ───────────────────────────────────────────────────────────

    async def on_event(self, event: TradeEvent) -> None:
        """Route incoming events to the appropriate tracker."""
        if event.event_type == "TOKEN_DETECTED":
            self._on_token_detected(event)

        elif event.event_type == "AI_DECISION":
            self._on_ai_decision(event)
            await self._run_detectors(event.strategy_id)

        elif event.event_type == "BUY_EXECUTED":
            self._on_buy(event)

        elif event.event_type == "SELL_EXECUTED":
            self._on_sell(event)
            await self._run_detectors(event.strategy_id)

    def _on_token_detected(self, event: TradeEvent) -> None:
        """Track token arrival rate."""
        self._tokens_detected_total += 1
        # Increment tokens-seen counter for all active strategies
        for state in self._states.values():
            state.tokens_seen_since_last_buy += 1
        self._global.tokens_seen_since_last_buy += 1

    def _on_ai_decision(self, event: TradeEvent) -> None:
        """Ingest an AI evaluation result."""
        state = self._get_state(event.strategy_id)
        data = event.data

        confidence = data.get("confidence", 0.0)
        decision = data.get("decision", "SKIP")
        reasoning = data.get("reasoning", "")
        elapsed_ms = data.get("elapsed_ms", 0.0)

        # Record confidence
        state.confidences.append(confidence)
        state.decisions.append(decision)

        # Record reasoning hash (first 16 chars of SHA256)
        # Normalize: lowercase, strip whitespace runs, truncate to 150 chars
        normalized = " ".join(reasoning.lower().split())[:150]
        reason_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        state.reasoning_hashes.append(reason_hash)

        # Record response time
        if elapsed_ms > 0:
            state.response_times_ms.append(elapsed_ms)
            # Update baseline (exponential moving average of first N samples)
            if state.baseline_response_time_count < self.config.response_time_min_samples:
                state.baseline_response_time_count += 1
                alpha = 1.0 / state.baseline_response_time_count
                state.baseline_response_time_ms = (
                    state.baseline_response_time_ms * (1 - alpha) + elapsed_ms * alpha
                )

        # Track skip streak
        if decision in ("SKIP", "PASS", "NO_BUY"):
            state.consecutive_skips += 1
        else:
            state.consecutive_skips = 0

    def _on_buy(self, event: TradeEvent) -> None:
        """Reset skip/token counters on buy."""
        state = self._get_state(event.strategy_id)
        state.consecutive_skips = 0
        state.tokens_seen_since_last_buy = 0

    def _on_sell(self, event: TradeEvent) -> None:
        """Ingest a trade outcome."""
        state = self._get_state(event.strategy_id)
        pnl_pct = event.data.get("pnl_pct", 0.0)
        is_win = pnl_pct > 0

        state.trade_outcomes.append(is_win)
        if is_win:
            state.total_wins += 1
            state.consecutive_losses = 0
        else:
            state.total_losses += 1
            state.consecutive_losses += 1

    # ───────────────────────────────────────────────────────────
    #  DETECTORS
    # ───────────────────────────────────────────────────────────

    async def _run_detectors(self, strategy_id: str | None) -> None:
        """Run all drift detectors and emit warnings for any that trip."""
        state = self._get_state(strategy_id)
        alerts = []

        alert = self._detect_confidence_clustering(state, strategy_id)
        if alert:
            alerts.append(alert)

        alert = self._detect_reasoning_collapse(state, strategy_id)
        if alert:
            alerts.append(alert)

        alert = self._detect_skip_streak(state, strategy_id)
        if alert:
            alerts.append(alert)

        alert = self._detect_response_time_drift(state, strategy_id)
        if alert:
            alerts.append(alert)

        alert = self._detect_loss_cascade(state, strategy_id)
        if alert:
            alerts.append(alert)

        alert = self._detect_win_rate_decay(state, strategy_id)
        if alert:
            alerts.append(alert)

        for alert in alerts:
            await self._emit_alert(alert)

    def _detect_confidence_clustering(
        self,
        state: _StrategyHealthState,
        strategy_id: str | None,
    ) -> DriftAlert | None:
        """
        Detect when confidence scores cluster near the threshold.

        The AI should produce a spread of confidence values across
        different tokens. If it's always returning ~0.60, it's not
        actually discriminating.
        """
        if len(state.confidences) < self.config.confidence_cluster_min_samples:
            return None

        values = list(state.confidences)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = math.sqrt(variance)

        if stddev >= self.config.confidence_stddev_floor:
            return None

        return DriftAlert(
            drift_type=DriftType.CONFIDENCE_CLUSTERING,
            severity="warning",
            message=(
                f"Confidence clustering: stddev={stddev:.3f} "
                f"(floor={self.config.confidence_stddev_floor:.3f}) "
                f"over last {len(values)} evaluations. "
                f"Mean={mean:.2f}. AI may be coin-flipping near threshold."
            ),
            details={
                "stddev": round(stddev, 4),
                "mean": round(mean, 3),
                "floor": self.config.confidence_stddev_floor,
                "sample_size": len(values),
                "recent_values": [round(v, 3) for v in values[-5:]],
            },
            strategy_id=strategy_id,
        )

    def _detect_reasoning_collapse(
        self,
        state: _StrategyHealthState,
        strategy_id: str | None,
    ) -> DriftAlert | None:
        """
        Detect when the AI produces identical reasoning across
        different tokens.

        We hash normalized reasoning strings and check what
        fraction of recent hashes are duplicates. High duplicate
        rate means the AI is templating, not analyzing.
        """
        if len(state.reasoning_hashes) < self.config.reasoning_collapse_min_samples:
            return None

        hashes = list(state.reasoning_hashes)
        unique = len(set(hashes))
        total = len(hashes)
        duplicate_ratio = 1.0 - (unique / total)

        if duplicate_ratio < self.config.reasoning_similarity_threshold:
            return None

        return DriftAlert(
            drift_type=DriftType.REASONING_COLLAPSE,
            severity="warning",
            message=(
                f"Reasoning collapse: {duplicate_ratio:.0%} duplicate reasoning "
                f"across last {total} evaluations ({unique} unique). "
                f"AI may be using template responses."
            ),
            details={
                "duplicate_ratio": round(duplicate_ratio, 3),
                "unique_hashes": unique,
                "total_hashes": total,
                "threshold": self.config.reasoning_similarity_threshold,
            },
            strategy_id=strategy_id,
        )

    def _detect_skip_streak(
        self,
        state: _StrategyHealthState,
        strategy_id: str | None,
    ) -> DriftAlert | None:
        """
        Detect long runs of SKIP decisions during active markets.

        Important distinction: we only flag this when tokens ARE
        arriving. If pump.fun is quiet, skipping is correct.
        """
        if state.consecutive_skips < self.config.skip_streak_threshold:
            return None

        # Only flag if tokens have been arriving
        if state.tokens_seen_since_last_buy < self.config.skip_streak_threshold:
            return None

        return DriftAlert(
            drift_type=DriftType.SKIP_STREAK,
            severity="warning",
            message=(
                f"Skip streak: {state.consecutive_skips} consecutive SKIPs "
                f"with {state.tokens_seen_since_last_buy} tokens seen. "
                f"AI may be overly cautious (loss-shy from historical memory?)."
            ),
            details={
                "consecutive_skips": state.consecutive_skips,
                "tokens_seen": state.tokens_seen_since_last_buy,
                "threshold": self.config.skip_streak_threshold,
                "recent_decisions": list(state.decisions)[-10:],
            },
            strategy_id=strategy_id,
        )

    def _detect_response_time_drift(
        self,
        state: _StrategyHealthState,
        strategy_id: str | None,
    ) -> DriftAlert | None:
        """
        Detect response time creep.

        On pump.fun, every second matters. If the AI is taking 2x
        longer than baseline, you're entering positions late.
        """
        if len(state.response_times_ms) < self.config.response_time_min_samples:
            return None
        if state.baseline_response_time_ms <= 0:
            return None

        recent = list(state.response_times_ms)
        # Use second half of the window as "recent" to compare against baseline
        half = len(recent) // 2
        if half < 3:
            return None
        recent_mean = sum(recent[half:]) / (len(recent) - half)
        baseline = state.baseline_response_time_ms

        ratio = recent_mean / baseline
        if ratio < self.config.response_time_drift_factor:
            return None

        return DriftAlert(
            drift_type=DriftType.RESPONSE_TIME_DRIFT,
            severity="warning",
            message=(
                f"Response time drift: recent mean {recent_mean:.0f}ms "
                f"vs baseline {baseline:.0f}ms ({ratio:.1f}x). "
                f"Entries may be arriving late."
            ),
            details={
                "recent_mean_ms": round(recent_mean, 1),
                "baseline_ms": round(baseline, 1),
                "ratio": round(ratio, 2),
                "drift_factor_threshold": self.config.response_time_drift_factor,
            },
            strategy_id=strategy_id,
        )

    def _detect_loss_cascade(
        self,
        state: _StrategyHealthState,
        strategy_id: str | None,
    ) -> DriftAlert | None:
        """
        Detect consecutive losing trades.

        Not necessarily an AI bug — could be a market regime shift.
        But it's a signal that the strategy should be cautious.
        """
        if state.consecutive_losses < self.config.loss_cascade_threshold:
            return None

        return DriftAlert(
            drift_type=DriftType.LOSS_CASCADE,
            severity="critical",
            message=(
                f"Loss cascade: {state.consecutive_losses} consecutive losses. "
                f"Session win rate: {state.session_win_rate:.0%}. "
                f"Consider pausing strategy or adjusting parameters."
            ),
            details={
                "consecutive_losses": state.consecutive_losses,
                "session_win_rate": round(state.session_win_rate, 3),
                "total_trades": state.total_trades,
                "threshold": self.config.loss_cascade_threshold,
            },
            strategy_id=strategy_id,
        )

    def _detect_win_rate_decay(
        self,
        state: _StrategyHealthState,
        strategy_id: str | None,
    ) -> DriftAlert | None:
        """
        Detect gradual win rate degradation.

        Compares the rolling window win rate against the full
        session baseline. A significant drop means something
        changed — market conditions, AI quality, or prompt staleness.
        """
        if state.total_trades < self.config.win_rate_min_trades:
            return None
        if len(state.trade_outcomes) < self.config.win_rate_min_trades:
            return None

        session_wr = state.session_win_rate
        rolling_wr = state.rolling_win_rate
        decay = session_wr - rolling_wr

        if decay < self.config.win_rate_decay_threshold:
            return None

        return DriftAlert(
            drift_type=DriftType.WIN_RATE_DECAY,
            severity="warning",
            message=(
                f"Win rate decay: rolling {rolling_wr:.0%} vs "
                f"session baseline {session_wr:.0%} "
                f"(delta={decay:.0%}). Performance degrading."
            ),
            details={
                "rolling_win_rate": round(rolling_wr, 3),
                "session_win_rate": round(session_wr, 3),
                "decay": round(decay, 3),
                "threshold": self.config.win_rate_decay_threshold,
                "rolling_window_size": len(state.trade_outcomes),
                "total_trades": state.total_trades,
            },
            strategy_id=strategy_id,
        )

    # ───────────────────────────────────────────────────────────
    #  ALERT EMISSION
    # ───────────────────────────────────────────────────────────

    async def _emit_alert(self, alert: DriftAlert) -> None:
        """Emit a drift alert if not in cooldown."""
        cooldown_key = f"{alert.drift_type.value}:{alert.strategy_id or 'global'}"
        now = time.monotonic()

        if cooldown_key in self._last_alert_time:
            last_time = self._last_alert_time[cooldown_key]
            if now - last_time < self.config.alert_cooldown_seconds:
                return

        self._last_alert_time[cooldown_key] = now
        self._alerts.append(alert)

        # Log it directly (always)
        log_fn = logger.warning if alert.severity == "warning" else logger.critical
        log_fn("🩺 AI HEALTH: %s", alert.message)

        # Emit onto the event bus (if available)
        if self._bus is not None:
            event = TradeEvent(
                event_type="AI_HEALTH_WARNING",
                category=EventCategory.AI,
                severity=(
                    EventSeverity.CRITICAL
                    if alert.severity == "critical"
                    else EventSeverity.WARNING
                ),
                strategy_id=alert.strategy_id,
                data={
                    "drift_type": alert.drift_type.value,
                    "details": alert.details,
                },
                message=f"🩺 {alert.message}",
            )
            # Avoid infinite recursion: emit but we won't re-process
            # AI_HEALTH_WARNING events (they're category=AI but type
            # is not in our handler's dispatch)
            await self._bus.emit(event)

    # ───────────────────────────────────────────────────────────
    #  PUBLIC API
    # ───────────────────────────────────────────────────────────

    def get_health_report(self) -> dict:
        """
        Full health report across all strategies.

        Returns a dict suitable for JSON serialization, dashboards,
        or inclusion in the bot's --status output.
        """
        uptime = time.monotonic() - self._monitor_start
        strategies = {}

        for sid, state in self._states.items():
            strategies[sid] = self._build_strategy_report(sid, state)

        # Global report
        global_report = self._build_strategy_report("global", self._global)

        return {
            "status": self._overall_status(),
            "uptime_seconds": round(uptime, 1),
            "tokens_detected": self._tokens_detected_total,
            "total_alerts": len(self._alerts),
            "active_alerts": self._get_active_alerts(),
            "strategies": strategies,
            "global": global_report,
            "recent_alerts": [
                {
                    "type": a.drift_type.value,
                    "severity": a.severity,
                    "message": a.message,
                    "strategy": a.strategy_id,
                    "time": a.timestamp.isoformat(),
                }
                for a in self._alerts[-10:]
            ],
        }

    def _build_strategy_report(self, sid: str, state: _StrategyHealthState) -> dict:
        """Build health metrics for a single strategy."""
        conf_values = list(state.confidences)
        conf_stddev = 0.0
        conf_mean = 0.0
        if len(conf_values) >= 2:
            conf_mean = sum(conf_values) / len(conf_values)
            variance = sum((v - conf_mean) ** 2 for v in conf_values) / len(conf_values)
            conf_stddev = math.sqrt(variance)

        # Reasoning uniqueness
        hashes = list(state.reasoning_hashes)
        reasoning_uniqueness = (
            len(set(hashes)) / len(hashes) if hashes else 1.0
        )

        # Response time stats
        rt = list(state.response_times_ms)
        rt_mean = sum(rt) / len(rt) if rt else 0.0

        return {
            "confidence": {
                "mean": round(conf_mean, 3),
                "stddev": round(conf_stddev, 4),
                "samples": len(conf_values),
                "healthy": conf_stddev >= self.config.confidence_stddev_floor or len(conf_values) < self.config.confidence_cluster_min_samples,
            },
            "reasoning": {
                "uniqueness": round(reasoning_uniqueness, 3),
                "samples": len(hashes),
                "healthy": reasoning_uniqueness > (1 - self.config.reasoning_similarity_threshold) or len(hashes) < self.config.reasoning_collapse_min_samples,
            },
            "response_time": {
                "current_mean_ms": round(rt_mean, 1),
                "baseline_ms": round(state.baseline_response_time_ms, 1),
                "samples": len(rt),
            },
            "decisions": {
                "consecutive_skips": state.consecutive_skips,
                "tokens_since_last_buy": state.tokens_seen_since_last_buy,
            },
            "outcomes": {
                "session_win_rate": round(state.session_win_rate, 3),
                "rolling_win_rate": round(state.rolling_win_rate, 3),
                "consecutive_losses": state.consecutive_losses,
                "total_trades": state.total_trades,
            },
        }

    def _overall_status(self) -> str:
        """Summarize health as a single status string."""
        if not self._alerts:
            return "healthy"

        # Check for any active (non-cooldown-expired) critical alerts
        now = time.monotonic()
        recent_criticals = [
            a for a in self._alerts[-20:]
            if a.severity == "critical"
            and (now - self._monitor_start) - (a.timestamp - datetime.fromtimestamp(self._monitor_start + (a.timestamp.timestamp() - datetime.now().timestamp()))).total_seconds() < self.config.alert_cooldown_seconds * 2
        ]
        # Simpler check: any critical alerts in last N minutes
        cutoff = datetime.now().timestamp() - self.config.alert_cooldown_seconds * 2
        recent_criticals = [
            a for a in self._alerts
            if a.severity == "critical" and a.timestamp.timestamp() > cutoff
        ]
        if recent_criticals:
            return "critical"

        recent_warnings = [
            a for a in self._alerts
            if a.severity == "warning" and a.timestamp.timestamp() > cutoff
        ]
        if recent_warnings:
            return "degraded"

        return "healthy"

    def _get_active_alerts(self) -> list[str]:
        """Get currently active (non-expired) alert types."""
        now = time.monotonic()
        active = []
        for key, last_time in self._last_alert_time.items():
            if now - last_time < self.config.alert_cooldown_seconds * 2:
                active.append(key)
        return active

    def reset(self, strategy_id: str | None = None) -> None:
        """
        Reset health state for a strategy (or all strategies).

        Useful after parameter changes or strategy restarts.
        """
        if strategy_id is None:
            self._states.clear()
            self._global = _new_strategy_state(self.config)
            self._alerts.clear()
            self._last_alert_time.clear()
            logger.info("🩺 AI Health Monitor: all state reset")
        elif strategy_id in self._states:
            self._states[strategy_id] = _new_strategy_state(self.config)
            # Clear alerts for this strategy
            self._alerts = [a for a in self._alerts if a.strategy_id != strategy_id]
            keys_to_clear = [k for k in self._last_alert_time if k.endswith(f":{strategy_id}")]
            for k in keys_to_clear:
                del self._last_alert_time[k]
            logger.info("🩺 AI Health Monitor: reset state for %s", strategy_id)
