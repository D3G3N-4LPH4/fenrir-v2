#!/usr/bin/env python3
"""
FENRIR - Second-life baseline (the SAPLING model).

Thesis (2026-10-01, from d3g3n's own 20x): at his $183k entry SAPLING looked
identical to the thousands of launches that died that day. The discriminating
information — community forming, floor holding for days — only existed later.
A gate that fires at sight fires on everything, so the model is staged:

  1. SIGHT (cheap, wide) — existing filters (grad_snipe, rh_onchain, scout).
  2. SURVIVAL (time) — most launches die in 48h. A coin that holds a floor for
     days and keeps its holders killed the noise. Almost free to compute.
  3. RE-IGNITION (the alert) — a survived coin re-accelerates: 1h volume far
     above its own trailing baseline, buy edge back, price lifting off the
     floor. That is the base breakout — the actual entry.

This module supplies stage 2+3: it pulls trailing daily candles from
GeckoTerminal, derives the floor/baseline, and attaches it to a TokenSnapshot
so the declarative ``second_life`` filter in filters.py can evaluate it.
Fail-open everywhere: no candles, no baseline, no fire.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from fenrir.discovery.models import Chain, TokenSnapshot

logger = logging.getLogger("FENRIR.SecondLife")

GECKO_BASE = "https://api.geckoterminal.com/api/v2"

# Chain -> GeckoTerminal network id (mirrors providers/geckoterminal.py).
_GECKO_NETWORKS: dict[Chain, str] = {
    Chain.SOLANA: "solana",
    Chain.ROBINHOOD: "robinhood",
}

LOOKBACK_DAYS = 7
BASELINE_TTL_SECONDS = 6 * 3600.0

# Survival: the trailing low must be at least this fraction of the trailing
# high — a coin that lost >95% from its own window high and flatlined is dead.
MIN_FLOOR_VS_MAX_PCT = 5.0

_baseline_cache: dict[str, tuple[SecondLifeBaseline, float]] = {}


@dataclass
class SecondLifeBaseline:
    """Trailing price/volume baseline for one pool, in price space (no supply math)."""

    floor_price_usd: float  # median of daily lows — the base the coin defended
    max_price_usd: float  # trailing window high
    low_price_usd: float  # trailing window low (death check)
    median_1h_volume_usd: float  # median daily volume / 24
    lookback_days: int


def parse_ohlcv(payload: object) -> list[tuple[float, float, float, float, float, float]]:
    """Pure parser: GeckoTerminal ohlcv payload -> (ts, o, h, l, c, volume_usd)."""
    try:
        if not isinstance(payload, dict):
            return []
        rows = (payload.get("data") or {}).get("attributes", {}).get("ohlcv_list")
        if not isinstance(rows, list):
            return []
        out: list[tuple[float, float, float, float, float, float]] = []
        for r in rows:
            if not isinstance(r, list | tuple) or len(r) < 6:
                continue
            ts, o, h, low, c, v = (float(x) for x in r[:6])
            if ts > 0 and h > 0:
                out.append((ts, o, h, low, c, v))
        return sorted(out, key=lambda r: r[0])
    except Exception:  # noqa: BLE001 - parser must never raise
        return []


def build_baseline(
    candles: list[tuple[float, float, float, float, float, float]],
    bucket_seconds: float = 86400.0,
) -> SecondLifeBaseline | None:
    """Pure: hourly candles -> daily buckets -> baseline.

    Needs >=3 full days before the current one. The latest (still-forming /
    ignition) day is excluded so a breakout in progress can't contaminate the
    floor or the trailing max it is measured against.
    """
    if not candles:
        return None
    buckets: dict[int, list[tuple[float, float, float, float, float, float]]] = {}
    for r in candles:
        buckets.setdefault(int(r[0] // bucket_seconds), []).append(r)
    days = sorted(buckets)
    # Drop the latest (still-forming / ignition) day: the baseline must describe
    # the base BEFORE the breakout, or the spike contaminates floor and max.
    base_days = days[:-1]
    if len(base_days) < 3:
        return None
    daily_lows = [min(r[3] for r in buckets[d]) for d in base_days]
    daily_highs = [max(r[2] for r in buckets[d]) for d in base_days]
    daily_vols = [sum(r[5] for r in buckets[d]) for d in base_days]
    daily_lows.sort()
    daily_vols.sort()
    n = len(daily_lows)
    floor = daily_lows[n // 2] if n % 2 else (daily_lows[n // 2 - 1] + daily_lows[n // 2]) / 2
    med_vol = daily_vols[n // 2] if n % 2 else (daily_vols[n // 2 - 1] + daily_vols[n // 2]) / 2
    return SecondLifeBaseline(
        floor_price_usd=floor,
        max_price_usd=max(daily_highs),
        low_price_usd=daily_lows[0],
        median_1h_volume_usd=med_vol / 24.0,
        lookback_days=len(base_days),
    )


def survived(baseline: SecondLifeBaseline) -> bool:
    """Stage 2: the coin never went to zero relative to its own range."""
    if baseline.max_price_usd <= 0:
        return False
    return (baseline.low_price_usd / baseline.max_price_usd * 100.0) >= MIN_FLOOR_VS_MAX_PCT


async def fetch_baseline(
    chain: Chain, pair_address: str, lookback_days: int = LOOKBACK_DAYS
) -> SecondLifeBaseline | None:
    """Trailing daily candles for one pool -> baseline. Cached 6h. Fail-open None."""
    network = _GECKO_NETWORKS.get(chain)
    if network is None or not pair_address:
        return None
    key = f"{network}/{pair_address}/{lookback_days}"
    hit = _baseline_cache.get(key)
    if hit is not None:
        baseline, fetched_at = hit
        if time.time() - fetched_at < BASELINE_TTL_SECONDS:
            return baseline
    try:
        import aiohttp

        url = (
            f"{GECKO_BASE}/networks/{network}/pools/{pair_address}"
            f"/ohlcv/hour?aggregate=1&limit={lookback_days * 24}"
        )
        async with aiohttp.ClientSession(
            trust_env=True, headers={"User-Agent": "FENRIR/2.0 second-life"}
        ) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning("GeckoTerminal ohlcv HTTP %d for %s", resp.status, key)
                    return None
                payload = await resp.json()
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.warning("GeckoTerminal ohlcv failed for %s: %s", key, e)
        return None
    fresh = build_baseline(parse_ohlcv(payload))
    if fresh is not None:
        _baseline_cache[key] = (fresh, time.time())
    return fresh


async def attach_baseline(
    snap: TokenSnapshot,
    lookback_days: int = LOOKBACK_DAYS,
    pair_override: str | None = None,
) -> TokenSnapshot:
    """Attach trailing baseline fields to a snapshot. No-op (None fields) on failure."""
    pair = pair_override or snap.pair_address
    if not pair:
        return snap
    baseline = await fetch_baseline(snap.chain, pair, lookback_days)
    if baseline is None:
        return snap
    snap.base_floor_price_usd = baseline.floor_price_usd
    snap.base_max_price_usd = baseline.max_price_usd
    snap.base_median_1h_volume_usd = baseline.median_1h_volume_usd
    snap.base_lookback_days = float(baseline.lookback_days)
    return snap


async def oldest_pool(chain: Chain, token_address: str) -> tuple[str, float] | None:
    """Oldest liquid DexScreener pool for a token -> (pair_address, created_at_ms).

    The snapshot's chosen pair is the most liquid *now*, which for a migrated
    runner is often a hours-old pool. The baseline and the token's true age need
    the oldest pool. Fail-open None.
    """
    try:
        from fenrir.discovery.providers.dexscreener import DexScreenerProvider

        ds = DexScreenerProvider(timeout_seconds=10)
        try:
            pairs = await ds.fetch_pairs(token_address)
        finally:
            await ds.close()
        best: str | None = None
        best_ts: float = float("inf")
        for p in pairs:
            addr = p.get("pairAddress")
            ts = p.get("pairCreatedAt")
            liq = ((p.get("liquidity") or {}).get("usd")) or 0
            if not addr or not isinstance(ts, int | float) or liq <= 0:
                continue
            if ts < best_ts:
                best, best_ts = addr, float(ts)
        return (best, best_ts) if best else None
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.warning("oldest_pool failed: %s", e)
        return None


async def oldest_pool_address(chain: Chain, token_address: str) -> str | None:
    """Oldest liquid pool address only (see oldest_pool). Fail-open None."""
    res = await oldest_pool(chain, token_address)
    return res[0] if res else None
