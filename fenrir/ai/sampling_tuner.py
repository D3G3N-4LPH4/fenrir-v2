#!/usr/bin/env python3
"""
FENRIR - SamplingTuner: EMA-based Claude Parameter Adaptation

G0DM0D3 lineage: G0DM0D3 observed that static sampling parameters across
all market regimes is suboptimal. The optimal temperature for sniping
fresh launches (fast, decisive) differs from graduation plays (conservative,
deliberate) and whale-copy scenarios (high-conviction verification).

This module EMA-adapts temperature, top_p, and frequency_penalty per
market regime based on trade outcomes, then persists state to SQLite so
parameters survive bot restarts.

Design:
  - EMA alpha = 0.1 (slow, stable adaptation — avoids chasing noise)
  - Reward signal: pnl_pct normalized to [-1, 1]
  - Positive reward → current params worked → drift toward params_used
  - Negative reward → current params underperformed → drift toward defaults
  - Effective alpha = alpha * |reward| (scales update by confidence)

ART export integration:
  Include params_used in trade records alongside pnl_pct for OpenPipe
  GRPO fine-tuning. The SamplingTuner.to_art_record() method formats
  this for the duality/replay_and_art.py exporter.

Usage:
    tuner = SamplingTuner(db_path="fenrir_trades.db")

    # Before each Claude call:
    params = tuner.get_params(MarketRegime.SNIPE)
    response = await analyst._call_llm(prompt, temperature=params.temperature)

    # After each trade close:
    tuner.record_outcome(MarketRegime.SNIPE, params, pnl_pct=+42.0)

    # For ART export:
    art_record = tuner.to_art_record(regime, params, pnl_pct)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# ─────────────────────────────────────────────────────────────
#  Market regimes
# ─────────────────────────────────────────────────────────────


class MarketRegime(Enum):
    """
    Market conditions that should drive different sampling strategies.

    SNIPE         — Fresh pump.fun launch; speed over precision.
    GRADUATION    — Token approaching bonding curve completion; deliberate.
    WHALE_COPY    — Mirroring a known profitable wallet; verification mode.
    HIGH_VOLATILITY — Rapid price swings; higher exploration needed.
    LOW_LIQUIDITY — Thin order book; conservative to avoid slippage traps.
    """

    SNIPE = "snipe"
    GRADUATION = "graduation"
    WHALE_COPY = "whale_copy"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"


# ─────────────────────────────────────────────────────────────
#  Sampling parameters
# ─────────────────────────────────────────────────────────────


@dataclass
class SamplingParams:
    """
    Claude/LLM sampling parameters for one inference call.

    All fields are clamped to valid ranges by .clamp().
    Pass as kwargs to the LLM API payload.
    """

    temperature: float = 0.3       # [0.1, 0.9]
    top_p: float = 0.9             # [0.1, 1.0]
    frequency_penalty: float = 0.0  # [0.0, 2.0]

    def clamp(self) -> SamplingParams:
        """Return a new SamplingParams with all values clamped to valid ranges."""
        return SamplingParams(
            temperature=max(0.1, min(0.9, self.temperature)),
            top_p=max(0.1, min(1.0, self.top_p)),
            frequency_penalty=max(0.0, min(2.0, self.frequency_penalty)),
        )

    def to_dict(self) -> dict:
        return {
            "temperature": round(self.temperature, 4),
            "top_p": round(self.top_p, 4),
            "frequency_penalty": round(self.frequency_penalty, 4),
        }

    def to_api_kwargs(self) -> dict:
        """Return only params that OpenRouter/Anthropic accepts."""
        return {
            "temperature": round(self.temperature, 4),
            "top_p": round(self.top_p, 4),
        }


# ─────────────────────────────────────────────────────────────
#  Per-regime defaults
# ─────────────────────────────────────────────────────────────

_DEFAULTS: dict[MarketRegime, SamplingParams] = {
    MarketRegime.SNIPE: SamplingParams(
        temperature=0.3,   # Decisive, fast
        top_p=0.90,
        frequency_penalty=0.0,
    ),
    MarketRegime.GRADUATION: SamplingParams(
        temperature=0.2,   # Conservative, calculated
        top_p=0.85,
        frequency_penalty=0.1,  # Penalise repetitive hedging
    ),
    MarketRegime.WHALE_COPY: SamplingParams(
        temperature=0.25,  # Verification mode
        top_p=0.90,
        frequency_penalty=0.0,
    ),
    MarketRegime.HIGH_VOLATILITY: SamplingParams(
        temperature=0.5,   # More exploratory under uncertainty
        top_p=0.95,
        frequency_penalty=0.0,
    ),
    MarketRegime.LOW_LIQUIDITY: SamplingParams(
        temperature=0.4,   # Higher caution
        top_p=0.90,
        frequency_penalty=0.2,  # Penalise token repetition in thin markets
    ),
}

EMA_ALPHA: float = 0.1  # Slow, stable adaptation


# ─────────────────────────────────────────────────────────────
#  SamplingTuner
# ─────────────────────────────────────────────────────────────


class SamplingTuner:
    """
    EMA-based Claude sampling parameter adaptation per market regime.

    G0DM0D3 lineage: Static temperature is suboptimal across regimes.
    G0DM0D3 identified that the optimal temperature for snipe decisions
    differs by ~0.2 from graduation plays, and that adaptive parameters
    improve win rate in backtests by reducing over-confident BUY decisions
    in high-volatility sessions.

    State is persisted to SQLite (sampling_tuner table) and loaded on
    restart, so the EMA accumulates across sessions rather than resetting
    on each bot launch.
    """

    _TABLE = "sampling_tuner"

    def __init__(self, db_path: str = "fenrir_trades.db"):
        self.db_path = db_path
        self._params: dict[MarketRegime, SamplingParams] = {}
        self._trade_counts: dict[MarketRegime, int] = {}
        self._init_db()
        self._load_state()

    # ── Setup ─────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    regime            TEXT PRIMARY KEY,
                    temperature       REAL NOT NULL,
                    top_p             REAL NOT NULL,
                    frequency_penalty REAL NOT NULL,
                    trade_count       INTEGER NOT NULL DEFAULT 0,
                    updated_at        TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_state(self) -> None:
        """Initialize with defaults, then override with any persisted values."""
        for regime in MarketRegime:
            self._params[regime] = _DEFAULTS[regime]
            self._trade_counts[regime] = 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    f"SELECT regime, temperature, top_p, frequency_penalty, trade_count "
                    f"FROM {self._TABLE}"
                ).fetchall()
        except sqlite3.Error:
            return

        for row in rows:
            try:
                regime = MarketRegime(row[0])
                self._params[regime] = SamplingParams(
                    temperature=float(row[1]),
                    top_p=float(row[2]),
                    frequency_penalty=float(row[3]),
                ).clamp()
                self._trade_counts[regime] = int(row[4])
            except (ValueError, TypeError):
                pass  # Unknown regime or bad data — skip

    # ── Public API ────────────────────────────────────────────

    def get_params(self, regime: MarketRegime) -> SamplingParams:
        """
        Return current EMA-tuned sampling params for the given regime.

        Always returns a valid SamplingParams (falls back to defaults if
        the regime has never been seen).
        """
        return self._params.get(regime, _DEFAULTS[regime])

    def record_outcome(
        self,
        regime: MarketRegime,
        params_used: SamplingParams,
        pnl_pct: float,
    ) -> None:
        """
        Update EMA params based on a completed trade outcome.

        Args:
            regime: Market regime active when the trade was taken.
            params_used: The SamplingParams that were passed to the LLM call.
            pnl_pct: Realised P&L in percent (+42.0 = +42%).

        EMA update rule:
            reward = clamp(pnl_pct / 100, -1, 1)
            Positive reward → drift toward params_used (they worked)
            Negative reward → drift toward regime defaults (revert to safe)
            effective_alpha = EMA_ALPHA * |reward|  (scale by confidence)
        """
        reward = max(-1.0, min(1.0, pnl_pct / 100.0))
        current = self._params[regime]
        default = _DEFAULTS[regime]
        effective_alpha = EMA_ALPHA * abs(reward)

        if reward >= 0:
            # Params worked — nudge toward what was used
            target = params_used
        else:
            # Params underperformed — nudge back toward regime defaults
            target = default

        new_params = SamplingParams(
            temperature=current.temperature
            + effective_alpha * (target.temperature - current.temperature),
            top_p=current.top_p
            + effective_alpha * (target.top_p - current.top_p),
            frequency_penalty=current.frequency_penalty
            + effective_alpha * (target.frequency_penalty - current.frequency_penalty),
        ).clamp()

        self._params[regime] = new_params
        self._trade_counts[regime] = self._trade_counts.get(regime, 0) + 1
        self._persist(regime, new_params)

    def get_all_params(self) -> dict:
        """Full status dict — suitable for health reports and dashboards."""
        return {
            regime.value: {
                **params.to_dict(),
                "default": _DEFAULTS[regime].to_dict(),
                "trade_count": self._trade_counts.get(regime, 0),
                "delta_temperature": round(
                    params.temperature - _DEFAULTS[regime].temperature, 4
                ),
            }
            for regime, params in self._params.items()
        }

    def to_art_record(
        self,
        regime: MarketRegime,
        params_used: SamplingParams,
        pnl_pct: float,
    ) -> dict:
        """
        Serialize a (regime, params, outcome) tuple for OpenPipe ART export.

        Plug into duality/replay_and_art.py to include sampling params
        in GRPO fine-tuning datasets alongside trade outcomes.
        """
        return {
            "regime": regime.value,
            "params_used": params_used.to_dict(),
            "current_ema": self._params[regime].to_dict(),
            "default": _DEFAULTS[regime].to_dict(),
            "pnl_pct": round(pnl_pct, 4),
            "reward": round(max(-1.0, min(1.0, pnl_pct / 100.0)), 4),
            "trade_count": self._trade_counts.get(regime, 0),
        }

    def reset(self, regime: MarketRegime | None = None) -> None:
        """Reset params to defaults for one regime (or all if None)."""
        targets = [regime] if regime else list(MarketRegime)
        for r in targets:
            self._params[r] = _DEFAULTS[r]
            self._trade_counts[r] = 0
            self._persist(r, _DEFAULTS[r])

    # ── Persistence ───────────────────────────────────────────

    def _persist(self, regime: MarketRegime, params: SamplingParams) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"INSERT OR REPLACE INTO {self._TABLE} "
                    f"(regime, temperature, top_p, frequency_penalty, trade_count, updated_at) "
                    f"VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        regime.value,
                        params.temperature,
                        params.top_p,
                        params.frequency_penalty,
                        self._trade_counts.get(regime, 0),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
        except sqlite3.Error:
            pass  # Non-critical — in-memory state is still correct
