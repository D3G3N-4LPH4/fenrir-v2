#!/usr/bin/env python3
"""Graduation snipe watcher: catch pump.fun graduations within minutes.

A pump.fun token graduates when its bonding curve closes (account reclaimed
on Raydium migration). This watcher polls curve state for in-flight tokens
tracked in ``pumpfun_curves.json``; when a tracked curve disappears or flips
``complete``, the token just graduated. It then pulls the fresh post-
graduation pair off DexScreener and alerts immediately when the first
minutes of tape are buy-driven — the velocity trigger a 10-minute polling
scout structurally misses (a coin can graduate and run to $1M between
cycles, as cNFTs did in ~20 minutes).

Snipe gates (all must pass):
  - 5m: >=15 buys and buy/sell >= 1.5 (on >=20 txns), or
    1h: >=50 buys and buy/sell >= 1.3            (early buy edge)
  - 5m price change >= -5%                       (not instantly dumping)
  - liquidity >= $20k                            (exit exists)
  - mcap <= $5M                                  (don't chase the mooned)
  - safety: not honeypot, mint authority not live, sell tax <= 15%
    (each skipped when the provider didn't supply it)

State: ``hidden_files/grad_snipe.json`` maps mint ->
  {"graduated_at": epoch, "alerted": bool, "reason": str}.
  Graduations are recorded even when the gates fail, so each mint is
  evaluated exactly once.

Usage:
  python tools/grad_snipe.py          # one watch cycle (alerts live)
  python tools/grad_snipe.py --dry-run   # evaluate, print, don't send/prune
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import aiohttp  # noqa: E402

from fenrir.discovery.alerts import escape_md  # noqa: E402
from fenrir.discovery.models import Chain, TokenSnapshot  # noqa: E402
from fenrir.discovery.providers.geckoterminal import GeckoTerminalProvider  # noqa: E402
from fenrir.discovery.providers.pumpfun import (  # noqa: E402
    DEFAULT_STATE_PATH as CURVE_STATE_PATH,
    PumpFunProvider,
)

GOAL_DIR = os.path.expanduser("~/workspace/goals/token-scout-watch/hidden_files")
SNIPE_STATE_PATH = os.path.join(GOAL_DIR, "grad_snipe.json")

# Curves at/above this progress are "in the graduation window" and watched.
WATCH_PROGRESS_PCT = 70.0
# Ignore tracked curves untouched for longer than this (stale watchlist).
WATCH_STALE_SECONDS = 30 * 60

# ── Snipe gates ──────────────────────────────────────────────────────
MIN_M5_BUYS = 15
MIN_M5_BS_RATIO = 1.5
MIN_M5_TXNS = 20
MIN_H1_BUYS = 50
MIN_H1_BS_RATIO = 1.3
MIN_M5_CHANGE_PCT = -5.0
MIN_LIQUIDITY_USD = 20_000.0
MAX_MCAP_USD = 5_000_000.0
MAX_SELL_TAX_PCT = 15.0

# Venues a fresh graduate trades on, preferred first.
_GRAD_VENUES = ("raydium", "pumpswap", "meteora", "orca")


# ── State ────────────────────────────────────────────────────────────


def _load_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ── Graduation detection (pure logic — unit tested) ──────────────────


def is_graduated(prev_entry: dict, state) -> bool:
    """True when a tracked, previously-incomplete curve just finished.

    ``state`` is None when the curve PDA returns no account — pump.fun
    closes the curve account on migration, so disappearance IS graduation.
    """
    if prev_entry.get("complete"):
        return False  # already graduated in an earlier run
    if state is None:
        return True
    return bool(getattr(state, "complete", False))


def graduation_inflow_sol(entry: dict) -> float | None:
    """SOL that entered the curve in its final tracked stretch."""
    last_sol = entry.get("last_sol")
    prev_sol = entry.get("prev_sol")
    if last_sol is None or prev_sol is None:
        return None
    return float(round(last_sol - prev_sol, 4))


def pick_pair(pairs: list[dict]) -> dict | None:
    """Fresh graduate's venue pair, else the deepest pool available."""
    if not pairs:
        return None
    for venue in _GRAD_VENUES:
        venue_pairs = [p for p in pairs if (p.get("dexId") or "").lower() == venue]
        if venue_pairs:
            return max(venue_pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)
    return max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)


def _ratio(buys: int | None, sells: int | None) -> float | None:
    if buys is None or sells is None:
        return None
    if sells == 0:
        return float("inf") if buys > 0 else None
    return buys / sells


def snipe_check(feat: dict, safety) -> tuple[bool, list[str]]:
    """Apply the snipe gates. Returns (passed, failure_reasons)."""
    fails: list[str] = []

    m5b, m5s = feat.get("m5_buys"), feat.get("m5_sells")
    h1b, h1s = feat.get("h1_buys"), feat.get("h1_sells")
    m5_ratio = _ratio(m5b, m5s) if m5b is not None else None
    h1_ratio = _ratio(h1b, h1s) if h1b is not None else None

    early_edge = (
        m5_ratio is not None
        and (m5b or 0) + (m5s or 0) >= MIN_M5_TXNS
        and (m5b or 0) >= MIN_M5_BUYS
        and m5_ratio >= MIN_M5_BS_RATIO
    )
    hourly_edge = h1_ratio is not None and (h1b or 0) >= MIN_H1_BUYS and h1_ratio >= MIN_H1_BS_RATIO
    if not (early_edge or hourly_edge):
        r5 = f"{m5_ratio:.2f}" if m5_ratio not in (None, float("inf")) else "n/a"
        r1 = f"{h1_ratio:.2f}" if h1_ratio not in (None, float("inf")) else "n/a"
        fails.append(f"no early buy edge (5m {m5b}/{m5s} b/s={r5}, 1h b/s={r1})")

    chg5 = feat.get("price_change_m5_pct")
    if chg5 is not None and chg5 < MIN_M5_CHANGE_PCT:
        fails.append(f"5m {chg5:+.1f}% — dumping out of the gate")

    liq = feat.get("liquidity_usd") or 0.0
    if liq < MIN_LIQUIDITY_USD:
        fails.append(f"LP ${liq:,.0f} < ${MIN_LIQUIDITY_USD:,.0f}")

    mcap = feat.get("market_cap_usd") or 0.0
    if mcap > MAX_MCAP_USD:
        fails.append(f"MCap ${mcap:,.0f} > ${MAX_MCAP_USD:,.0f} — already mooned")

    if safety is not None:
        if safety.honeypot:
            fails.append("honeypot")
        if safety.mint_disabled is False:
            fails.append("mint authority live")
        sell_tax = safety.sell_tax_pct or 0
        if sell_tax > MAX_SELL_TAX_PCT:
            fails.append(f"sell tax {sell_tax}%")

    return (not fails), fails


# ── Network ──────────────────────────────────────────────────────────


async def fetch_ds_pairs(mint: str, timeout: int = 20) -> list[dict]:
    """DexScreener pairs for a mint (fail-open: [] on error)."""
    url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
    try:
        async with aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={"User-Agent": "FENRIR/2.0 grad-snipe"},
        ) as s:
            async with s.get(url) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                return data.get("pairs") or []
    except Exception:  # noqa: BLE001 - discovery is fail-open
        return []


def pair_features(pair: dict) -> dict:
    tx = pair.get("txns") or {}
    m5 = tx.get("m5") or {}
    h1 = tx.get("h1") or {}
    chg = pair.get("priceChange") or {}
    return {
        "symbol": (pair.get("baseToken") or {}).get("symbol"),
        "name": (pair.get("baseToken") or {}).get("name"),
        "dex": pair.get("dexId"),
        "url": pair.get("url"),
        "price_usd": float(pair.get("priceUsd") or 0),
        "market_cap_usd": float(pair.get("marketCap") or 0),
        "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
        "m5_buys": m5.get("buys"),
        "m5_sells": m5.get("sells"),
        "h1_buys": h1.get("buys"),
        "h1_sells": h1.get("sells"),
        "price_change_m5_pct": chg.get("m5"),
        "price_change_h1_pct": chg.get("h1"),
    }


def _send_telegram(text: str) -> bool:
    import subprocess

    notify = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_notify.py")
    try:
        r = subprocess.run(
            [sys.executable, notify, "--parse-mode", "Markdown", text],
            capture_output=True,
            text=True,
            timeout=90,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001 - fail-open
        return False


def format_snipe_alert(
    mint: str, feat: dict, inflow_sol: float | None, graduated_ago_s: float
) -> str:
    sym = escape_md(feat.get("symbol") or "???")
    name = escape_md(feat.get("name") or "")
    header = f"\U0001f680 *{sym}*"
    if name and name.lower() != sym.lower().replace("\\", ""):
        header += f" \u2014 {name}"
    lines = [header, "Solana \u00b7 just graduated \u00b7 via grad_snipe", ""]
    ago = f"{graduated_ago_s / 60:.0f}m ago" if graduated_ago_s >= 60 else "<1m ago"
    grad_line = f"\U0001f393 graduated ~{ago}"
    if inflow_sol is not None:
        grad_line += f" \u00b7 final curve inflow {inflow_sol:.1f} SOL"
    lines.append(grad_line)
    lines.append(
        f"\U0001f4b0 mcap ${feat['market_cap_usd']:,.0f} \u00b7 "
        f"liq ${feat['liquidity_usd']:,.0f}"
    )
    m5b, m5s = feat.get("m5_buys"), feat.get("m5_sells")
    flow = f"5m {m5b if m5b is not None else '?'} buys / {m5s if m5s is not None else '?'} sells"
    chg5 = feat.get("price_change_m5_pct")
    if chg5 is not None:
        flow += f" \u00b7 {chg5:+.1f}% 5m"
    lines.append(f"\U0001f4c8 {flow}")
    lines += ["", f"`{mint}`"]
    if feat.get("url"):
        lines.append(f"[DexScreener]({feat['url']})")
    return "\n".join(lines)


# ── Watch cycle ──────────────────────────────────────────────────────


async def _discover_curves() -> int:
    """Seed the watchlist: record curve readings for fresh pump.fun pools."""
    gt = GeckoTerminalProvider(timeout_seconds=15)
    provider = PumpFunProvider()
    try:
        base = await gt.fetch_new_pool_addresses(Chain.SOLANA, 60)
        if not base:
            return 0
        states = await provider.curve_states(base)
        now = time.time()
        n = 0
        for mint, state in states.items():
            if state.complete:
                continue
            provider.record_reading(mint, state, now)
            n += 1
        provider.prune_state()
        return n
    finally:
        await provider.close()
        await gt.close()


async def _detect_graduations(
    provider: PumpFunProvider, snipe_state: dict, now: float
) -> list[tuple[str, dict]]:
    """Re-read watched curves; return [(mint, curve_entry)] newly graduated."""
    store = PumpFunProvider._load_state(CURVE_STATE_PATH)
    watched = [
        (mint, e)
        for mint, e in store.items()
        if e.get("last_progress", 0) >= WATCH_PROGRESS_PCT
        and now - e.get("last_check", 0) < WATCH_STALE_SECONDS
        and mint not in snipe_state
        and not e.get("complete")
    ]
    if not watched:
        return []
    states = await provider.curve_states([m for m, _ in watched])
    out = []
    for mint, entry in watched:
        state = states.get(mint)  # None = account closed = migrated
        if is_graduated(entry, state):
            out.append((mint, entry))
        elif state is not None:
            provider.record_reading(mint, state, now)  # keep velocity fresh
    return out


async def run_cycle(dry_run: bool = False) -> dict:
    """One watch cycle. Returns {"graduated": [...], "alerted": [...]}."""
    from tools.evaluate import enrich_safety  # noqa

    now = time.time()
    snipe_state = _load_json(SNIPE_STATE_PATH)
    provider = PumpFunProvider()
    summary: dict = {"graduated": [], "alerted": []}
    try:
        await _discover_curves()
        graduations = await _detect_graduations(provider, snipe_state, now)
        for mint, entry in graduations:
            inflow = graduation_inflow_sol(entry)
            pairs = await fetch_ds_pairs(mint)
            pair = pick_pair(pairs)
            reason = ""
            alerted = False
            if pair is None:
                reason = "no DexScreener pair yet"
            else:
                feat = pair_features(pair)
                snap = TokenSnapshot(
                    chain=Chain.SOLANA,
                    token_address=mint,
                    symbol=feat["symbol"] or "",
                    market_cap_usd=feat["market_cap_usd"],
                    liquidity_usd=feat["liquidity_usd"],
                )
                try:
                    await enrich_safety(snap, None)
                    safety = snap.safety
                except Exception:  # noqa: BLE001 - safety best-effort
                    safety = None
                ok, fails = snipe_check(feat, safety)
                if ok:
                    msg = format_snipe_alert(mint, feat, inflow, 0)
                    if dry_run:
                        print(msg + "\n" + "─" * 40)
                        alerted = True
                    elif _send_telegram(msg):
                        alerted = True
                    else:
                        reason = "telegram send failed"
                else:
                    reason = "; ".join(fails)
            snipe_state[mint] = {
                "graduated_at": now,
                "alerted": alerted,
                "reason": reason,
            }
            summary["graduated"].append(mint)
            if alerted:
                summary["alerted"].append(mint)
            print(
                f"{'ALERT' if alerted else 'skip'} {mint[:12]}… "
                f"{reason or 'snipe gates passed'}"
            )
        if not dry_run:
            _save_json(snipe_state, SNIPE_STATE_PATH)
        return summary
    finally:
        await provider.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="pump.fun graduation snipe watcher")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate and print, don't send alerts or write state",
    )
    args = ap.parse_args()
    summary = asyncio.run(run_cycle(dry_run=args.dry_run))
    print(f"graduated={len(summary['graduated'])} " f"alerted={len(summary['alerted'])}")


if __name__ == "__main__":
    main()
