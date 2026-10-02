#!/usr/bin/env python3
"""Chart-pattern recognition for FENRIR's playbook layer.

Classical chart patterns (double tops/bottoms, head & shoulders, bull/bear
flags) are reversal/continuation signals — a different axis from FENRIR's
momentum/flow playbooks. They assist coin selection two ways:

- bullish patterns (double bottom, inverse H&S, bull flag) join the playbook
  tags as confluence for entries;
- bearish patterns (double top, H&S, bear flag) surface as caution tags on a
  candidate that otherwise passed the gates (they do NOT auto-fail anything —
  hit rates get quantified in the user_cases loop before they ever gate).

Pure detection (no scipy — hand-rolled ZigZag keeps the dependency set
unchanged). Fail-open: short/noisy series simply yield no patterns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ── Tunables (memecoin timeframes are noisier than stocks; tolerances are
#    wider than textbook values) ──────────────────────────────────────────
ZIGZAG_DEVIATION = 0.08  # pivot confirmed on an 8% reversal
PEAK_TOLERANCE = 0.10  # double-top peaks within 10% of each other
HEAD_PROMINENCE = 0.12  # head must exceed shoulders by 12%
SHOULDER_TOLERANCE = 0.15  # shoulders within 15% of each other
FLAG_IMPULSE_PCT = 0.40  # flagpole: >=40% move ...
FLAG_IMPULSE_MAX_BARS = 12  # ... in at most 12 hourly bars
FLAG_BARS_MIN = 8  # consolidation lasts at least 8 bars
FLAG_BARS_MAX = 36
FLAG_RANGE_PCT = 0.30  # consolidation range <= 30% of the impulse
MIN_CANDLES = 30

# Candle tuple: (ts, open, high, low, close, volume_usd) — matches
# second_life.parse_ohlcv output.


@dataclass
class PatternMatch:
    """One detected chart pattern."""

    pattern_id: str  # e.g. "double_top"
    display_name: str  # e.g. "Double Top"
    direction: str  # "bullish" | "bearish"
    strength: float  # 0-1 conviction
    rationale: str
    key_level: float | None = None  # neckline / breakout level
    bar_index: int = -1  # bar where the pattern completed


def zigzag(
    closes: list[float], deviation: float = ZIGZAG_DEVIATION
) -> list[tuple[int, float, str]]:
    """Pivots as (index, price, "high"|"low").

    A pivot is confirmed when price reverses ``deviation`` from the running
    extreme; the final extreme is appended as a provisional pivot.
    """
    pivots: list[tuple[int, float, str]] = []
    n = len(closes)
    if n < 2 or deviation <= 0:
        return pivots
    if any(c <= 0 for c in closes):
        return pivots
    up = closes[1] >= closes[0]
    ext_idx, ext = 0, closes[0]
    for i in range(1, n):
        p = closes[i]
        if up:
            if p >= ext:
                ext_idx, ext = i, p
            elif (ext - p) / ext >= deviation:
                pivots.append((ext_idx, ext, "high"))
                up = False
                ext_idx, ext = i, p
        else:
            if p <= ext:
                ext_idx, ext = i, p
            elif (p - ext) / ext >= deviation:
                pivots.append((ext_idx, ext, "low"))
                up = True
                ext_idx, ext = i, p
    pivots.append((ext_idx, ext, "high" if up else "low"))
    return pivots


def _rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(a, b)


def _fit_slope(points: list[tuple[int, float]]) -> float:
    """Least-squares slope of price on bar index (relative, per-bar)."""
    n = len(points)
    if n < 2:
        return 0.0
    xs = [float(x) for x, _ in points]
    ys = [p for _, p in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / den
    return slope / my if my else 0.0


# Trendline slope bands (per-bar relative): |slope| <= FLAT -> horizontal.
TRI_FLAT_SLOPE = 0.004
TRI_TREND_SLOPE = 0.008
TRI_PIVOTS = 6  # pivots considered for triangle/wedge fits


def _detect_triangle_wedge(pivots: list[tuple[int, float, str]]) -> PatternMatch | None:
    """Ascending/descending/symmetrical triangles + rising/falling wedges.

    Fits trendlines through recent swing highs and swing lows:
    - highs flat + lows rising -> ascending triangle (bullish)
    - lows flat + highs falling -> descending triangle (bearish)
    - highs falling + lows rising -> symmetrical triangle (neutral)
    - both rising + converging -> rising wedge (bearish)
    - both falling + converging -> falling wedge (bullish)
    """
    if len(pivots) < TRI_PIVOTS:
        return None
    seq = pivots[-TRI_PIVOTS:]
    highs = [(i, p) for i, p, k in seq if k == "high"]
    lows = [(i, p) for i, p, k in seq if k == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    sh, sl = _fit_slope(highs), _fit_slope(lows)
    highs_flat = abs(sh) <= TRI_FLAT_SLOPE
    lows_flat = abs(sl) <= TRI_FLAT_SLOPE
    converging = (highs[-1][1] - lows[-1][1]) < (highs[0][1] - lows[0][1])
    pid: str | None = None
    name: str | None = None
    direction: str | None = None
    if highs_flat and sl > TRI_TREND_SLOPE:
        pid, name, direction = "ascending_triangle", "Ascending Triangle", "bullish"
    elif lows_flat and sh < -TRI_TREND_SLOPE:
        pid, name, direction = "descending_triangle", "Descending Triangle", "bearish"
    elif sh < -TRI_TREND_SLOPE and sl > TRI_TREND_SLOPE:
        pid, name, direction = "symmetrical_triangle", "Symmetrical Triangle", "neutral"
    elif sh > TRI_TREND_SLOPE and sl > TRI_TREND_SLOPE and converging:
        pid, name, direction = "rising_wedge", "Rising Wedge", "bearish"
    elif sh < -TRI_TREND_SLOPE and sl < -TRI_TREND_SLOPE and converging:
        pid, name, direction = "falling_wedge", "Falling Wedge", "bullish"
    if pid is None:
        return None
    strength = round(
        min(
            1.0,
            0.4
            + min(abs(sh) / (TRI_TREND_SLOPE * 4), 1.0) * 0.3
            + min(abs(sl) / (TRI_TREND_SLOPE * 4), 1.0) * 0.3,
        ),
        3,
    )
    return PatternMatch(
        pattern_id=pid,
        display_name=name or pid,
        direction=direction or "neutral",
        strength=strength,
        rationale=f"upper trend {sh:+.1%}/bar, lower trend {sl:+.1%}/bar over {len(seq)} swings",
        key_level=highs[-1][1] if direction == "bullish" else lows[-1][1],
        bar_index=seq[-1][0],
    )


def _detect_triple_top(pivots: list[tuple[int, float, str]]) -> PatternMatch | None:
    if len(pivots) < 5:
        return None
    seq = pivots[-5:]
    if [k for _, _, k in seq] != ["high", "low", "high", "low", "high"]:
        return None
    tops = [seq[0][1], seq[2][1], seq[4][1]]
    base = min(tops)
    if max(tops) / base - 1 > PEAK_TOLERANCE:
        return None
    valleys = [seq[1][1], seq[3][1]]
    if min(valleys) >= base:
        return None
    depth = (base - min(valleys)) / base
    strength = round(min(1.0, 0.5 + min(depth * 2, 1.0) * 0.5), 3)
    return PatternMatch(
        pattern_id="triple_top",
        display_name="Triple Top",
        direction="bearish",
        strength=strength,
        rationale=f"three peaks within {max(tops) / base - 1:.0%}, {depth:.0%} deep valleys",
        key_level=min(valleys),
        bar_index=seq[-1][0],
    )


def _detect_triple_bottom(pivots: list[tuple[int, float, str]]) -> PatternMatch | None:
    if len(pivots) < 5:
        return None
    seq = pivots[-5:]
    if [k for _, _, k in seq] != ["low", "high", "low", "high", "low"]:
        return None
    bots = [seq[0][1], seq[2][1], seq[4][1]]
    top = max(bots)
    if 1 - min(bots) / top > PEAK_TOLERANCE:
        return None
    peaks = [seq[1][1], seq[3][1]]
    if max(peaks) <= top:
        return None
    height = (max(peaks) - top) / top
    strength = round(min(1.0, 0.5 + min(height * 2, 1.0) * 0.5), 3)
    return PatternMatch(
        pattern_id="triple_bottom",
        display_name="Triple Bottom",
        direction="bullish",
        strength=strength,
        rationale=f"three troughs within {1 - min(bots) / top:.0%}, {height:.0%} rebound",
        key_level=max(peaks),
        bar_index=seq[-1][0],
    )


def _detect_double_top(pivots: list[tuple[int, float, str]]) -> PatternMatch | None:
    # ... high, low, high with the second high at/near the end.
    if len(pivots) < 3:
        return None
    (i1, h1, k1), (iv, v, kv), (i2, h2, k2) = pivots[-3:]
    if not (k1 == "high" and kv == "low" and k2 == "high"):
        return None
    if v >= min(h1, h2):
        return None
    diff = _rel_diff(h1, h2)
    if diff > PEAK_TOLERANCE:
        return None
    depth = (min(h1, h2) - v) / min(h1, h2)
    strength = round(min(1.0, (1 - diff / PEAK_TOLERANCE) * 0.6 + min(depth * 2, 1.0) * 0.4), 3)
    return PatternMatch(
        pattern_id="double_top",
        display_name="Double Top",
        direction="bearish",
        strength=strength,
        rationale=f"twin peaks within {diff:.0%} with a {depth:.0%} valley between",
        key_level=v,
        bar_index=i2,
    )


def _detect_double_bottom(pivots: list[tuple[int, float, str]]) -> PatternMatch | None:
    if len(pivots) < 3:
        return None
    (i1, l1, k1), (iv, v, kv), (i2, l2, k2) = pivots[-3:]
    if not (k1 == "low" and kv == "high" and k2 == "low"):
        return None
    if v <= max(l1, l2):
        return None
    diff = _rel_diff(l1, l2)
    if diff > PEAK_TOLERANCE:
        return None
    height = (v - max(l1, l2)) / max(l1, l2)
    strength = round(min(1.0, (1 - diff / PEAK_TOLERANCE) * 0.6 + min(height * 2, 1.0) * 0.4), 3)
    return PatternMatch(
        pattern_id="double_bottom",
        display_name="Double Bottom",
        direction="bullish",
        strength=strength,
        rationale=f"twin troughs within {diff:.0%} with a {height:.0%} rebound between",
        key_level=v,
        bar_index=i2,
    )


def _detect_head_shoulders(pivots: list[tuple[int, float, str]]) -> PatternMatch | None:
    # high, low, HIGHER high, low, high — head prominent, shoulders symmetric.
    if len(pivots) < 5:
        return None
    seq = pivots[-5:]
    kinds = [k for _, _, k in seq]
    if kinds != ["high", "low", "high", "low", "high"]:
        return None
    (_, ls, _), _, (_, head, _), _, (_, rs, _) = seq
    if head <= max(ls, rs) * (1 + HEAD_PROMINENCE):
        return None
    if _rel_diff(ls, rs) > SHOULDER_TOLERANCE:
        return None
    neckline = (seq[1][1] + seq[3][1]) / 2
    sym = 1 - _rel_diff(ls, rs) / SHOULDER_TOLERANCE
    prom = min(1.0, (head / max(ls, rs) - 1) / (HEAD_PROMINENCE * 2))
    strength = round(min(1.0, sym * 0.5 + prom * 0.5), 3)
    return PatternMatch(
        pattern_id="head_shoulders",
        display_name="Head & Shoulders",
        direction="bearish",
        strength=strength,
        rationale=f"head {head / max(ls, rs) - 1:.0%} above shoulders, symmetry {sym:.0%}",
        key_level=neckline,
        bar_index=seq[-1][0],
    )


def _detect_inverse_head_shoulders(pivots: list[tuple[int, float, str]]) -> PatternMatch | None:
    if len(pivots) < 5:
        return None
    seq = pivots[-5:]
    kinds = [k for _, _, k in seq]
    if kinds != ["low", "high", "low", "high", "low"]:
        return None
    (_, ls, _), _, (_, head, _), _, (_, rs, _) = seq
    if head >= min(ls, rs) * (1 - HEAD_PROMINENCE):
        return None
    if _rel_diff(ls, rs) > SHOULDER_TOLERANCE:
        return None
    neckline = (seq[1][1] + seq[3][1]) / 2
    sym = 1 - _rel_diff(ls, rs) / SHOULDER_TOLERANCE
    prom = min(1.0, (1 - head / min(ls, rs)) / (HEAD_PROMINENCE * 2))
    strength = round(min(1.0, sym * 0.5 + prom * 0.5), 3)
    return PatternMatch(
        pattern_id="inverse_head_shoulders",
        display_name="Inverse Head & Shoulders",
        direction="bullish",
        strength=strength,
        rationale=f"head {1 - head / min(ls, rs):.0%} below shoulders, symmetry {sym:.0%}",
        key_level=neckline,
        bar_index=seq[-1][0],
    )


def _detect_flag(closes: list[float], bullish: bool) -> PatternMatch | None:
    """Impulse + tight consolidation. Operates on closes, not pivots."""
    n = len(closes)
    if n < FLAG_IMPULSE_MAX_BARS + FLAG_BARS_MIN:
        return None
    # Find the strongest impulse ending at least FLAG_BARS_MIN bars ago.
    best: tuple[float, int, int] | None = None  # (move, start, end)
    for end in range(FLAG_IMPULSE_MAX_BARS, n - FLAG_BARS_MIN):
        for start in range(max(0, end - FLAG_IMPULSE_MAX_BARS), end):
            base = closes[start]
            if base <= 0:
                continue
            move = (closes[end] - base) / base
            if bullish and move >= FLAG_IMPULSE_PCT:
                if best is None or move > best[0]:
                    best = (move, start, end)
            elif not bullish and move <= -FLAG_IMPULSE_PCT:
                if best is None or move < best[0]:
                    best = (move, start, end)
    if best is None:
        return None
    move, _, pole_end = best
    cons = closes[pole_end + 1 :]
    if len(cons) > FLAG_BARS_MAX:
        cons = cons[-FLAG_BARS_MAX:]
    if len(cons) < FLAG_BARS_MIN:
        return None
    rng = (max(cons) - min(cons)) / closes[pole_end] if closes[pole_end] > 0 else 1
    max_range = abs(move) * FLAG_RANGE_PCT
    if rng > max_range:
        return None
    # Drift should lean against the impulse (classic flag tilt).
    drift = (cons[-1] - cons[0]) / cons[0] if cons[0] > 0 else 0
    if bullish and drift > abs(move) * 0.25:
        return None
    if not bullish and drift < -abs(move) * 0.25:
        return None
    tightness = 1 - rng / max_range if max_range > 0 else 0
    strength = round(
        min(1.0, 0.4 + min(abs(move) / (FLAG_IMPULSE_PCT * 2), 1.0) * 0.3 + tightness * 0.3), 3
    )
    pid = "bull_flag" if bullish else "bear_flag"
    return PatternMatch(
        pattern_id=pid,
        display_name="Bull Flag" if bullish else "Bear Flag",
        direction="bullish" if bullish else "bearish",
        strength=strength,
        rationale=f"{abs(move):.0%} pole, {len(cons)}-bar consolidation holding {rng:.0%} range",
        key_level=max(cons) if bullish else min(cons),
        bar_index=n - 1,
    )


def detect_patterns(
    candles: list[tuple[float, float, float, float, float, float]],
    *,
    deviation: float = ZIGZAG_DEVIATION,
) -> list[PatternMatch]:
    """Detect chart patterns in hourly candles. Pure; fail-open []."""
    try:
        closes = [c[4] for c in candles]
        if len(closes) < MIN_CANDLES:
            return []
        pivots = zigzag(closes, deviation)
        out: list[PatternMatch] = []
        for fn in (
            _detect_double_top,
            _detect_double_bottom,
            _detect_triple_top,
            _detect_triple_bottom,
            _detect_head_shoulders,
            _detect_inverse_head_shoulders,
            _detect_triangle_wedge,
        ):
            m = fn(pivots)
            if m is not None:
                out.append(m)
        bull = _detect_flag(closes, bullish=True)
        if bull is not None:
            out.append(bull)
        bear = _detect_flag(closes, bullish=False)
        if bear is not None:
            out.append(bear)
        # One pattern per direction family: keep the strongest.
        by_dir: dict[str, PatternMatch] = {}
        for m in out:
            key = m.direction
            if key not in by_dir or m.strength > by_dir[key].strength:
                by_dir[key] = m
        return sorted(by_dir.values(), key=lambda m: -m.strength)
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("pattern detection failed: %s", e)
        return []


# ── Candle fetch + snapshot attach ────────────────────────────────────────
_GECKO_BASE = "https://api.geckoterminal.com/api/v2"
_GECKO_NETWORKS = {"solana": "solana", "robinhood": "robinhood"}
_PATTERN_CACHE: dict[str, tuple[list[PatternMatch], float]] = {}
PATTERN_TTL_SECONDS = 30 * 60


async def fetch_patterns(
    chain: str, pair_address: str, limit_hours: int = 120
) -> list[PatternMatch]:
    """Hourly candles -> patterns. Cached 30m. Fail-open []."""
    from fenrir.discovery.second_life import parse_ohlcv

    network = _GECKO_NETWORKS.get((chain or "").lower())
    if not network or not pair_address:
        return []
    key = f"{network}/{pair_address}"
    hit = _PATTERN_CACHE.get(key)
    if hit is not None:
        pats, ts = hit
        if time.time() - ts < PATTERN_TTL_SECONDS:
            return pats
    try:
        import aiohttp

        url = (
            f"{_GECKO_BASE}/networks/{network}/pools/{pair_address}"
            f"/ohlcv/hour?aggregate=1&limit={limit_hours}"
        )
        async with aiohttp.ClientSession(
            trust_env=True, headers={"User-Agent": "FENRIR/2.0 chart-patterns"}
        ) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []
                payload = await resp.json()
        pats = detect_patterns(parse_ohlcv(payload))
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("pattern fetch failed for %s: %s", key, e)
        return []
    _PATTERN_CACHE[key] = (pats, time.time())
    return pats


async def attach_chart_patterns(snap) -> list[PatternMatch]:
    """Fetch + detect, stash on the snapshot. Returns the matches."""
    chain = getattr(snap, "chain", None)
    chain_val = getattr(chain, "value", chain)
    pats = await fetch_patterns(str(chain_val or ""), getattr(snap, "pair_address", "") or "")
    snap.chart_patterns = pats
    return pats


def pattern_tags(pats: list[PatternMatch]) -> list[dict]:
    """Serialize for alerts / gate-tracker records."""
    return [
        {
            "pattern_id": p.pattern_id,
            "display_name": p.display_name,
            "direction": p.direction,
            "strength": p.strength,
            "rationale": p.rationale,
            "key_level": p.key_level,
        }
        for p in pats
    ]
