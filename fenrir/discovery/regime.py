#!/usr/bin/env python3
"""Market-regime overlay for the scout's gates (level 1: tagging).

Memecoins don't have regimes — they have events. But the *market they swim
in* does: a volatility_breakout alert fired while SOL is grinding up is a
different bet than the same alert fired while SOL is chopping sideways.

This module classifies SOL's market state deterministically from trailing
hourly closes. No HMM, no model — a transparent z-score rule:

    z = (24h log return) / (24h realized vol)

|z| >= 1 means the day's net move exceeded one full daily-vol: a directional
day. Otherwise the market is chopping.

Regimes: ``trend_up`` | ``chop`` | ``trend_down`` | ``unknown``.

Level 1 wires this into the gate tracker only as a tag (``regime`` on each
record + ``--by-regime`` on the report). Nothing is suppressed or rescored —
pure measurement until forward alerts validate a split. Fail-open everywhere:
any fetch/parse problem yields ``unknown``, never an exception.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time

log = logging.getLogger("fenrir.regime")

TREND_UP = "trend_up"
CHOP = "chop"
TREND_DOWN = "trend_down"
UNKNOWN = "unknown"

REGIMES = (TREND_UP, CHOP, TREND_DOWN, UNKNOWN)

WSOL_MINT = "So11111111111111111111111111111111111111112"
_GECKO_BASE = "https://api.geckoterminal.com/api/v2"

# Need a 24h return window plus a 24h vol window behind it.
_MIN_CLOSES = 49
# |z| >= 1: the day's move exceeded one daily-vol -> directional.
_TREND_Z = 1.0

# Caches: SOL pool resolution is stable (24h); closes are fresh enough at 10m.
_POOL_CACHE: dict[str, tuple[str, float]] = {}
_POOL_TTL = 24 * 3600
_CLOSES_CACHE: dict[str, tuple[list[tuple[float, float]], float]] = {}
_CLOSES_TTL = 10 * 60


def classify_regime(closes: list[float]) -> str:
    """Pure: trailing hourly closes (oldest -> newest) -> regime.

    Fail-open: any bad input yields ``unknown``.
    """
    try:
        xs = [float(c) for c in closes if c is not None and float(c) > 0]
    except (TypeError, ValueError):
        return UNKNOWN
    if len(xs) < _MIN_CLOSES:
        return UNKNOWN
    try:
        rets = [math.log(xs[i + 1] / xs[i]) for i in range(len(xs) - 1)]
        window = rets[-24:]
        ret_24 = sum(window)
        mean = ret_24 / 24.0
        var = sum((r - mean) ** 2 for r in window) / 24.0
        vol_24 = math.sqrt(var)
        if vol_24 <= 0:
            return CHOP  # a market that doesn't move is chop by definition
        z = ret_24 / (vol_24 * math.sqrt(24.0))
        if z >= _TREND_Z:
            return TREND_UP
        if z <= -_TREND_Z:
            return TREND_DOWN
        return CHOP
    except (ValueError, ZeroDivisionError, OverflowError):
        return UNKNOWN


def classify_at(pairs: list[tuple[float, float]], ts: float) -> str:
    """Regime as of a past timestamp: slice (ts, close) pairs at ts, classify.

    Used to backfill the gate tracker — what was SOL doing when each alert
    fired? Fail-open ``unknown``.
    """
    try:
        closes = [c for t, c in pairs if t <= ts]
        return classify_regime(closes)
    except Exception:  # noqa: BLE001 - fail-open
        return UNKNOWN


async def _fetch_json(url: str) -> dict | None:
    try:
        import aiohttp

        async with aiohttp.ClientSession(
            trust_env=True, headers={"User-Agent": "FENRIR/2.0 regime"}
        ) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
                return payload if isinstance(payload, dict) else None
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("regime fetch failed for %s: %s", url, e)
        return None


_STABLE_QUOTES = ("USDC", "USDT", "USDH", "UXD", "PYUSD", "FDUSD")


async def _resolve_sol_pool() -> str | None:
    """A SOL/USD pool on GeckoTerminal Solana: SOL as base, stablecoin quote.

    The token-pools endpoint is not liquidity-sorted, so "highest reserve"
    alone can grab a memecoin/SOL pool (seen live: XRPN/SOL). Prefer
    ``SOL / <stable>`` by name, highest reserve wins; fall back to any pool
    whose base price sits in a sane SOL range. Cached 24h. Fail-open None.
    """
    hit = _POOL_CACHE.get("sol")
    if hit is not None:
        addr, ts = hit
        if time.time() - ts < _POOL_TTL:
            return addr
    payload = await _fetch_json(f"{_GECKO_BASE}/networks/solana/tokens/{WSOL_MINT}/pools?page=1")
    if not payload:
        return None
    cands: list[tuple[str, float]] = []
    for item in payload.get("data") or []:
        attrs = (item or {}).get("attributes") or {}
        pool_addr = attrs.get("address")
        name = str(attrs.get("name") or "")
        if not pool_addr:
            continue
        try:
            liq = float(attrs.get("reserve_in_usd") or 0)
        except (TypeError, ValueError):
            liq = 0.0
        base, _, quote = name.partition(" / ")
        if base.strip() == "SOL" and any(s in quote for s in _STABLE_QUOTES):
            cands.append((pool_addr, liq))
    if not cands:
        # Fallback: any pool whose base price looks like SOL ($10-$10k).
        for item in payload.get("data") or []:
            attrs = (item or {}).get("attributes") or {}
            pool_addr = attrs.get("address")
            if not pool_addr:
                continue
            try:
                px = float(attrs.get("base_token_price_usd") or 0)
                liq = float(attrs.get("reserve_in_usd") or 0)
            except (TypeError, ValueError):
                continue
            if 10.0 <= px <= 10_000.0:
                cands.append((pool_addr, liq))
    if not cands:
        return None
    best = max(cands, key=lambda kv: kv[1])[0]
    _POOL_CACHE["sol"] = (best, time.time())
    return best


async def fetch_sol_hourly(limit: int = 500) -> list[tuple[float, float]]:
    """Hourly (ts, close) for SOL, newest last. Cached 10m. Fail-open []."""
    hit = _CLOSES_CACHE.get("sol")
    if hit is not None:
        pairs, ts = hit
        if time.time() - ts < _CLOSES_TTL and pairs:
            return pairs
    pool = await _resolve_sol_pool()
    if not pool:
        return []
    payload = await _fetch_json(
        f"{_GECKO_BASE}/networks/solana/pools/{pool}/ohlcv/hour?aggregate=1&limit={limit}"
    )
    if not payload:
        return []
    try:
        from fenrir.discovery.second_life import parse_ohlcv

        rows = parse_ohlcv(payload)
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("regime ohlcv parse failed: %s", e)
        return []
    pairs = [(float(ts), float(c)) for ts, _o, _h, _l, c, _v in rows if c > 0]
    if pairs:
        _CLOSES_CACHE["sol"] = (pairs, time.time())
    return pairs


def current_regime() -> str:
    """SOL's regime right now. Sync wrapper for script contexts. Fail-open."""
    try:
        pairs = asyncio.run(fetch_sol_hourly(200))
        if not pairs:
            return UNKNOWN
        return classify_regime([c for _, c in pairs])
    except Exception:  # noqa: BLE001 - fail-open (incl. running event loop)
        return UNKNOWN


def regime_at(ts: float) -> str:
    """SOL's regime at a past epoch timestamp. Sync wrapper. Fail-open."""
    try:
        pairs = asyncio.run(fetch_sol_hourly(720))
        if not pairs:
            return UNKNOWN
        return classify_at(pairs, ts)
    except Exception:  # noqa: BLE001 - fail-open
        return UNKNOWN
