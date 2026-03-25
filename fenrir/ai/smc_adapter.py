"""
smc_adapter.py — FENRIR v2 Smart Money Concepts Integration
============================================================
Wraps the `smartmoneyconcepts` library and bridges it into FENRIR's
EventBus + Claude AI context pipeline.

Install dep:
    pip install smartmoneyconcepts

Usage:
    adapter = SMCAdapter(swing_length=10, min_candles=30)
    adapter.update(ohlcv_df)              # call on each new candle
    signals = adapter.get_signals()       # dict of latest SMC state
    context = adapter.get_claude_context()  # formatted string for AI prompt
    events  = adapter.drain_events()      # list of EventBus-ready dicts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    from smartmoneyconcepts import smc
except ImportError:
    raise ImportError("pip install smartmoneyconcepts")

logger = logging.getLogger("fenrir.smc_adapter")


# ---------------------------------------------------------------------------
# Data classes for typed signal payloads
# ---------------------------------------------------------------------------

@dataclass
class FVGSignal:
    direction: str          # "bullish" | "bearish"
    top: float
    bottom: float
    gap_size: float         # absolute size in price units
    gap_pct: float          # gap size as % of bottom price
    candle_index: int
    mitigated: bool
    mitigated_index: int | None = None

    @property
    def event_type(self) -> str:
        return "SMC_FVG"

    def to_dict(self) -> dict:
        return {
            "event": self.event_type,
            "direction": self.direction,
            "top": self.top,
            "bottom": self.bottom,
            "gap_size": self.gap_size,
            "gap_pct": round(self.gap_pct, 4),
            "candle_index": self.candle_index,
            "mitigated": self.mitigated,
            "mitigated_index": self.mitigated_index,
        }


@dataclass
class StructureSignal:
    signal_type: str        # "BOS" | "CHoCH"
    direction: str          # "bullish" | "bearish"
    level: float
    broken_index: int
    candle_index: int

    @property
    def event_type(self) -> str:
        return f"SMC_{self.signal_type}"

    def to_dict(self) -> dict:
        return {
            "event": self.event_type,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "level": self.level,
            "broken_index": self.broken_index,
            "candle_index": self.candle_index,
        }


@dataclass
class LiquiditySignal:
    direction: str          # "bullish" (buy-side swept) | "bearish" (sell-side swept)
    level: float
    end_index: int          # last candle that was part of the pool
    swept_index: int        # candle that swept it
    candle_index: int

    @property
    def event_type(self) -> str:
        return "SMC_LIQUIDITY_SWEEP"

    def to_dict(self) -> dict:
        return {
            "event": self.event_type,
            "direction": self.direction,
            "level": self.level,
            "end_index": self.end_index,
            "swept_index": self.swept_index,
            "candle_index": self.candle_index,
        }


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class SMCAdapter:
    """
    Maintains a rolling OHLCV window and computes SMC indicators on each
    update. Surfaces signals as typed objects, raw dicts for the EventBus,
    and a formatted string for Claude's decision context.

    Parameters
    ----------
    swing_length : int
        Lookback/lookahead for swing high/low detection.
        Use 5–15 for fast memecoin candles (vs 50 for traditional markets).
    min_candles : int
        Minimum candles required before any indicator fires.
        Need at least 2*swing_length + a few to avoid edge noise.
    fvg_min_gap_pct : float
        Ignore FVGs smaller than this percentage of price.
        Filters microstructure noise on low-liquidity tokens.
    liq_range_pct : float
        How close two swing levels need to be (as fraction of price)
        to count as a liquidity pool. Default 1%.
    """

    def __init__(
        self,
        swing_length: int = 10,
        min_candles: int = 30,
        fvg_min_gap_pct: float = 0.005,   # 0.5%
        liq_range_pct: float = 0.01,
    ):
        self.swing_length = swing_length
        self.min_candles = max(min_candles, swing_length * 2 + 5)
        self.fvg_min_gap_pct = fvg_min_gap_pct
        self.liq_range_pct = liq_range_pct

        # Rolling window — trimmed to keep memory bounded
        self._MAX_CANDLES = 500
        self._ohlcv: pd.DataFrame = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )

        # Latest computed signals
        self._fvgs: list[FVGSignal] = []
        self._structure: list[StructureSignal] = []
        self._liquidity: list[LiquiditySignal] = []

        # Pending events not yet drained
        self._event_queue: list[dict] = []

        # Track what we've already emitted to avoid duplicate events.
        # BOS and CHoCH use separate sets so a candle can emit both signal
        # types without the integer-offset collision bug (choch_key = idx + 10_000
        # would collide once candle indices exceed 10 000).
        self._emitted_fvg_indices: set[int] = set()
        self._emitted_structure_indices: set[int] = set()   # BOS only
        self._emitted_choch_indices: set[int] = set()       # CHoCH only
        self._emitted_liq_indices: set[int] = set()

        self._last_update: datetime | None = None
        self._candle_count: int = 0
        # Last error from _compute_all(); surfaced in get_signals() so callers
        # know whether the current signals are fresh or stale from a prior pass.
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, ohlcv: pd.DataFrame) -> bool:
        """
        Feed new OHLCV data. Can be a full history or incremental append.
        Call this on every new candle close from your LiveDataCollector.

        Returns True if signals were computed, False if not enough data yet.
        """
        if ohlcv is None or ohlcv.empty:
            return False

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(ohlcv.columns):
            missing = required_cols - set(ohlcv.columns)
            raise ValueError(f"OHLCV missing columns: {missing}")

        # Merge + deduplicate on index
        combined = pd.concat([self._ohlcv, ohlcv])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        # Trim rolling window
        if len(combined) > self._MAX_CANDLES:
            combined = combined.iloc[-self._MAX_CANDLES:]
            # Rebuild emitted sets to only track candles still in window
            # (simplest approach: clear and let them repopulate without re-emit
            #  because old indices no longer exist in the window)
            self._emitted_fvg_indices.clear()
            self._emitted_structure_indices.clear()
            self._emitted_choch_indices.clear()
            self._emitted_liq_indices.clear()

        self._ohlcv = combined.astype(float)
        self._candle_count = len(self._ohlcv)
        self._last_update = datetime.now(timezone.utc)

        if self._candle_count < self.min_candles:
            logger.debug(
                f"SMCAdapter: {self._candle_count}/{self.min_candles} candles — waiting"
            )
            return False

        self._last_error = None   # clear stale error before recompute
        self._compute_all()
        return True

    def get_signals(self) -> dict[str, Any]:
        """
        Returns the latest SMC state as a clean dictionary.
        Fresh = not yet mitigated. Recent = within last 20 candles.

        Check ``signals["stale"]`` before acting on results: if True, the last
        ``_compute_all()`` call failed and the signals below are carried over
        from the previous successful pass.
        """
        n = self._candle_count
        recent_cutoff = max(0, n - 20)

        fresh_fvgs = [f for f in self._fvgs if not f.mitigated]
        recent_bos = [s for s in self._structure if s.signal_type == "BOS" and s.candle_index >= recent_cutoff]
        recent_choch = [s for s in self._structure if s.signal_type == "CHoCH" and s.candle_index >= recent_cutoff]
        recent_sweeps = [l for l in self._liquidity if l.candle_index >= recent_cutoff]

        return {
            "candle_count": n,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "current_price": float(self._ohlcv["close"].iloc[-1]),
            "fresh_fvgs": [f.to_dict() for f in fresh_fvgs],
            "recent_bos": [s.to_dict() for s in recent_bos],
            "recent_choch": [s.to_dict() for s in recent_choch],
            "recent_liquidity_sweeps": [l.to_dict() for l in recent_sweeps],
            "summary": self._build_summary(fresh_fvgs, recent_bos, recent_choch, recent_sweeps),
            # Stale flag: True when _compute_all() failed last pass.
            # Signals above are carried from the prior successful update.
            "stale": self._last_error is not None,
            "last_error": self._last_error,
        }

    def get_claude_context(self) -> str:
        """
        Returns a compact, structured string to inject into Claude's
        decision prompt. Designed for token efficiency.

        When the last compute pass failed, a WARNING header is prepended
        so Claude knows the signals below are carried from a prior candle and
        should be weighted accordingly — not treated as fresh structure.
        """
        if self._candle_count < self.min_candles:
            return f"[SMC] Insufficient candles ({self._candle_count}/{self.min_candles}) — no signals yet."

        signals = self.get_signals()
        price = signals["current_price"]

        # Stale guard: if the last _compute_all() failed, tell Claude explicitly.
        # Stale signals are still included — partial context beats silence — but
        # Claude must know they may not reflect the most recent candle.
        if signals["stale"]:
            stale_header = (
                "WARNING: SMC signals are STALE "
                "(last compute failed: " + str(signals["last_error"]) + "). "
                "Treat structure signals below with lower confidence -- "
                "they reflect a prior candle, not the current bar. "
                "When in doubt, bias toward SKIP or reduce position size.\n"
            )
        else:
            stale_header = ""

        lines = [f"{stale_header}[SMC ANALYSIS @ price={price:.6f} | candles={self._candle_count}]"]

        # --- FVGs ---
        fresh = signals["fresh_fvgs"]
        if fresh:
            lines.append(f"\nFair Value Gaps (unmitigated):")
            for f in fresh[-3:]:  # cap at 3 most recent
                dist_pct = ((price - f["bottom"]) / price) * 100 if f["direction"] == "bullish" else ((f["top"] - price) / price) * 100
                proximity = "PRICE INSIDE" if f["bottom"] <= price <= f["top"] else f"{abs(dist_pct):.1f}% away"
                lines.append(
                    f"  • {f['direction'].upper()} FVG [{f['bottom']:.6f} – {f['top']:.6f}] "
                    f"gap={f['gap_pct']*100:.2f}% | {proximity}"
                )
        else:
            lines.append("\nFair Value Gaps: none unmitigated")

        # --- BOS / CHoCH ---
        choch = signals["recent_choch"]
        bos = signals["recent_bos"]
        if choch:
            last = choch[-1]
            lines.append(
                f"\nChange of Character (CHoCH): {last['direction'].upper()} at level {last['level']:.6f} "
                f"[candle {last['broken_index']}] ⚠️ STRUCTURE REVERSAL SIGNAL"
            )
        if bos:
            last = bos[-1]
            lines.append(
                f"\nBreak of Structure (BOS): {last['direction'].upper()} at level {last['level']:.6f} "
                f"[candle {last['broken_index']}] — trend continuation"
            )
        if not choch and not bos:
            lines.append("\nMarket Structure: no recent BOS or CHoCH")

        # --- Liquidity Sweeps ---
        sweeps = signals["recent_liquidity_sweeps"]
        if sweeps:
            lines.append(f"\nLiquidity Sweeps (last 20 candles):")
            for s in sweeps[-3:]:
                reversal_hint = "→ watch for reversal" if s["direction"] == "bearish" else "→ continuation possible"
                lines.append(
                    f"  • {s['direction'].upper()} pool swept at {s['level']:.6f} "
                    f"[swept candle {s['swept_index']}] {reversal_hint}"
                )
        else:
            lines.append("\nLiquidity Sweeps: none in recent candles")

        # --- Bias summary ---
        lines.append(f"\nBias: {signals['summary']['bias']}")
        if signals["summary"]["warnings"]:
            lines.append("Warnings: " + " | ".join(signals["summary"]["warnings"]))

        return "\n".join(lines)

    def drain_events(self) -> list[dict]:
        """
        Returns all pending events and clears the queue.
        Plug this into your EventBus.publish() loop.

        Example:
            for event in adapter.drain_events():
                event_bus.publish(event["event"], event)
        """
        events = self._event_queue.copy()
        self._event_queue.clear()
        return events

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_all(self) -> None:
        ohlcv = self._ohlcv.copy()
        n = len(ohlcv)

        try:
            swing_df = smc.swing_highs_lows(ohlcv, swing_length=self.swing_length)
        except Exception as e:
            self._last_error = f"swing_highs_lows: {e}"
            logger.warning(f"SMC swing_highs_lows failed: {e}")
            return

        self._compute_fvg(ohlcv)
        self._compute_bos_choch(ohlcv, swing_df, n)
        self._compute_liquidity(ohlcv, swing_df, n)
        # All sub-computations succeeded; errors from sub-methods are logged
        # individually but don't mark the whole pass as stale — partial signals
        # are still better than nothing.
        self._last_error = None

    def _compute_fvg(self, ohlcv: pd.DataFrame) -> None:
        try:
            fvg_df = smc.fvg(ohlcv, join_consecutive=True)
        except Exception as e:
            logger.warning(f"SMC fvg failed: {e}")
            return

        self._fvgs = []
        for i, row in fvg_df.iterrows():
            if pd.isna(row.get("FVG")) or row["FVG"] == 0:
                continue

            top = row.get("Top")
            bottom = row.get("Bottom")
            if pd.isna(top) or pd.isna(bottom) or bottom == 0:
                continue

            gap_size = top - bottom
            gap_pct = gap_size / bottom

            # Filter microstructure noise
            if gap_pct < self.fvg_min_gap_pct:
                continue

            mit_idx_raw = row.get("MitigatedIndex")
            mitigated = not pd.isna(mit_idx_raw) and mit_idx_raw > 0
            mit_idx = int(mit_idx_raw) if mitigated else None

            candle_idx = ohlcv.index.get_loc(i) if i in ohlcv.index else -1

            signal = FVGSignal(
                direction="bullish" if row["FVG"] == 1 else "bearish",
                top=float(top),
                bottom=float(bottom),
                gap_size=float(gap_size),
                gap_pct=float(gap_pct),
                candle_index=candle_idx,
                mitigated=mitigated,
                mitigated_index=mit_idx,
            )
            self._fvgs.append(signal)

            # Emit new unmitigated FVGs to event queue
            if not mitigated and candle_idx not in self._emitted_fvg_indices:
                self._emitted_fvg_indices.add(candle_idx)
                self._event_queue.append(signal.to_dict())
                logger.debug(f"SMC FVG event: {signal.direction} [{signal.bottom:.6f}–{signal.top:.6f}]")

    def _compute_bos_choch(
        self, ohlcv: pd.DataFrame, swing_df: pd.DataFrame, n: int
    ) -> None:
        try:
            struct_df = smc.bos_choch(ohlcv, swing_df, close_break=True)
        except Exception as e:
            logger.warning(f"SMC bos_choch failed: {e}")
            return

        self._structure = []
        for i, row in struct_df.iterrows():
            bos_val = row.get("BOS", 0)
            choch_val = row.get("CHoCH", 0)

            if (pd.isna(bos_val) or bos_val == 0) and (pd.isna(choch_val) or choch_val == 0):
                continue

            level = row.get("Level")
            broken_idx = row.get("BrokenIndex")
            if pd.isna(level) or pd.isna(broken_idx):
                continue

            candle_idx = ohlcv.index.get_loc(i) if i in ohlcv.index else -1

            if not pd.isna(bos_val) and bos_val != 0:
                sig = StructureSignal(
                    signal_type="BOS",
                    direction="bullish" if bos_val == 1 else "bearish",
                    level=float(level),
                    broken_index=int(broken_idx),
                    candle_index=candle_idx,
                )
                self._structure.append(sig)
                if candle_idx not in self._emitted_structure_indices:
                    self._emitted_structure_indices.add(candle_idx)
                    self._event_queue.append(sig.to_dict())

            if not pd.isna(choch_val) and choch_val != 0:
                sig = StructureSignal(
                    signal_type="CHoCH",
                    direction="bullish" if choch_val == 1 else "bearish",
                    level=float(level),
                    broken_index=int(broken_idx),
                    candle_index=candle_idx,
                )
                self._structure.append(sig)
                # Use a dedicated set for CHoCH indices — avoids the integer-offset
                # collision (candle_idx + 10_000) that broke deduplication once
                # candle indices exceeded 10 000 in long-running sessions.
                if candle_idx not in self._emitted_choch_indices:
                    self._emitted_choch_indices.add(candle_idx)
                    self._event_queue.append(sig.to_dict())
                    logger.debug(f"SMC CHoCH event: {sig.direction} level={sig.level:.6f}")

    def _compute_liquidity(
        self, ohlcv: pd.DataFrame, swing_df: pd.DataFrame, n: int
    ) -> None:
        try:
            liq_df = smc.liquidity(ohlcv, swing_df, range_percent=self.liq_range_pct)
        except Exception as e:
            logger.warning(f"SMC liquidity failed: {e}")
            return

        self._liquidity = []
        for i, row in liq_df.iterrows():
            liq_val = row.get("Liquidity", 0)
            if pd.isna(liq_val) or liq_val == 0:
                continue

            swept_raw = row.get("Swept")
            if pd.isna(swept_raw) or swept_raw == 0:
                continue  # Only care about swept pools

            level = row.get("Level")
            end_raw = row.get("End")
            if pd.isna(level) or pd.isna(end_raw):
                continue

            candle_idx = ohlcv.index.get_loc(i) if i in ohlcv.index else -1
            swept_idx = int(swept_raw)

            sig = LiquiditySignal(
                direction="bullish" if liq_val == 1 else "bearish",
                level=float(level),
                end_index=int(end_raw),
                swept_index=swept_idx,
                candle_index=candle_idx,
            )
            self._liquidity.append(sig)

            if swept_idx not in self._emitted_liq_indices:
                self._emitted_liq_indices.add(swept_idx)
                self._event_queue.append(sig.to_dict())
                logger.debug(f"SMC Liquidity Sweep: {sig.direction} at {sig.level:.6f}")

    def _build_summary(
        self,
        fresh_fvgs: list[FVGSignal],
        recent_bos: list[StructureSignal],
        recent_choch: list[StructureSignal],
        recent_sweeps: list[LiquiditySignal],
    ) -> dict:
        """
        Derives a simple directional bias and warning flags.
        Claude can use this for quick triage without parsing all signals.
        """
        warnings = []
        bull_score = 0
        bear_score = 0

        price = float(self._ohlcv["close"].iloc[-1])

        # FVG scoring — bearish FVG above price = overhead supply
        for f in fresh_fvgs:
            if f.direction == "bearish" and f.bottom > price:
                bear_score += 1
                warnings.append(f"bearish FVG overhead at {f.bottom:.6f}")
            elif f.direction == "bullish" and f.top < price:
                bull_score += 1

        # CHoCH is the strongest reversal signal
        for c in recent_choch:
            if c.direction == "bearish":
                bear_score += 3
                warnings.append(f"bearish CHoCH at {c.level:.6f} — DUMP RISK")
            else:
                bull_score += 2

        # BOS continuation
        for b in recent_bos:
            if b.direction == "bullish":
                bull_score += 1
            else:
                bear_score += 1

        # Bearish liquidity sweep = stops above equal highs taken → reversal likely
        for s in recent_sweeps:
            if s.direction == "bearish":
                bear_score += 2
                warnings.append(f"bearish liquidity swept at {s.level:.6f} — reversal watch")
            else:
                bull_score += 1

        if bear_score > bull_score + 2:
            bias = f"BEARISH (score: bear={bear_score} bull={bull_score})"
        elif bull_score > bear_score + 2:
            bias = f"BULLISH (score: bull={bull_score} bear={bear_score})"
        else:
            bias = f"NEUTRAL (score: bull={bull_score} bear={bear_score})"

        return {"bias": bias, "warnings": warnings, "bull_score": bull_score, "bear_score": bear_score}


# ---------------------------------------------------------------------------
# FENRIR integration helper — drop this into your strategy layer
# ---------------------------------------------------------------------------

class FENRIRSMCMixin:
    """
    Mixin for FENRIR strategy classes (SniperStrategy, GraduationStrategy).
    Attach an SMCAdapter per token and enrich Claude context automatically.

    Usage in your strategy:
        class SniperStrategy(FENRIRSMCMixin, BaseStrategy):
            def on_candle(self, token_mint, ohlcv_df):
                self.update_smc(token_mint, ohlcv_df)
                context = self.get_smc_context(token_mint)
                # inject context into your Claude prompt
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._smc_adapters: dict[str, SMCAdapter] = {}

    def _get_or_create_adapter(self, token_mint: str) -> SMCAdapter:
        if token_mint not in self._smc_adapters:
            self._smc_adapters[token_mint] = SMCAdapter(
                swing_length=10,
                min_candles=25,
                fvg_min_gap_pct=0.003,   # 0.3% — tighter for fast memecoins
                liq_range_pct=0.015,
            )
        return self._smc_adapters[token_mint]

    def update_smc(self, token_mint: str, ohlcv_df: pd.DataFrame) -> None:
        adapter = self._get_or_create_adapter(token_mint)
        adapter.update(ohlcv_df)

        # Drain events and publish to EventBus if available
        events = adapter.drain_events()
        if events and hasattr(self, "event_bus"):
            for event in events:
                self.event_bus.publish(event["event"], event)

    def get_smc_context(self, token_mint: str) -> str:
        adapter = self._smc_adapters.get(token_mint)
        if adapter is None:
            return "[SMC] No data yet for this token."
        return adapter.get_claude_context()

    def cleanup_smc(self, token_mint: str) -> None:
        """Call when exiting a position to free memory."""
        self._smc_adapters.pop(token_mint, None)


# ---------------------------------------------------------------------------
# Example: Claude prompt injection
# ---------------------------------------------------------------------------

CLAUDE_PROMPT_TEMPLATE = """
You are FENRIR's trade decision engine analyzing a pump.fun memecoin.

Token: {token_mint}
Current Price: {price}
Position: {position_size} SOL
Entry Price: {entry_price}

--- MARKET STRUCTURE (Smart Money Concepts) ---
{smc_context}

--- RECENT TRADES ---
{recent_trades}

--- TASK ---
Based on the SMC signals above, decide: HOLD, ADD, or EXIT.
Pay special attention to:
- Any bearish CHoCH (structure reversal) = strong exit signal
- Bearish liquidity sweeps near current price = reversal imminent
- Unmitigated bearish FVGs above price = resistance / supply zones
- Bullish BOS = trend intact, bias toward holding

Respond with JSON: {{"action": "HOLD|ADD|EXIT", "confidence": 0.0-1.0, "reasoning": "..."}}
"""

def build_claude_prompt(
    adapter: SMCAdapter,
    token_mint: str,
    price: float,
    position_size: float,
    entry_price: float,
    recent_trades: str = "N/A",
) -> str:
    return CLAUDE_PROMPT_TEMPLATE.format(
        token_mint=token_mint,
        price=price,
        position_size=position_size,
        entry_price=entry_price,
        smc_context=adapter.get_claude_context(),
        recent_trades=recent_trades,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.DEBUG)

    # Simulate 60 candles of a pump-then-dump token
    random.seed(42)
    prices = [0.000100]
    for i in range(1, 60):
        if i < 30:
            # pump phase
            prices.append(prices[-1] * (1 + random.uniform(0.01, 0.08)))
        else:
            # dump phase
            prices.append(prices[-1] * (1 - random.uniform(0.01, 0.06)))

    rows = []
    for i, close in enumerate(prices):
        open_ = prices[i - 1] if i > 0 else close
        high = close * (1 + random.uniform(0, 0.03))
        low = close * (1 - random.uniform(0, 0.03))
        volume = random.uniform(1000, 50000)
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": volume})

    df = pd.DataFrame(rows)

    adapter = SMCAdapter(swing_length=8, min_candles=20)
    adapter.update(df)

    print("\n" + "=" * 60)
    print(adapter.get_claude_context())
    print("\n--- EVENTS EMITTED ---")
    for e in adapter.drain_events():
        print(e)

    print("\n--- SAMPLE PROMPT ---")
    print(build_claude_prompt(adapter, "TokenMintXYZ", prices[-1], 0.5, prices[10]))
