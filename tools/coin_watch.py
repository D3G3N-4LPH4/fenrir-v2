#!/usr/bin/env python3
"""Coin watch — keep an eye on specific tokens for data.

A lightweight time-series tracker for coins d3g3n wants watched (e.g. the SI
vertical runner that failed the gates). Every tick snapshots price/mcap/liq/
volume/flow from DexScreener and appends it to a per-coin series; Telegram
pings only on violent moves (>=50% vs the previous tick).

Usage:
  python tools/coin_watch.py add <address> --label SI
  python tools/coin_watch.py tick        # snapshot all watched coins
  python tools/coin_watch.py report      # print current state + series stats
  python tools/coin_watch.py remove <address>

State: <scout-goal>/hidden_files/coin_watch.json (outside the repo).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse

STATE_PATH = os.path.expanduser(
    "~/workspace/goals/token-scout-watch/hidden_files/coin_watch.json")
ALERT_MOVE_PCT = 50.0  # Telegram ping when |move vs prev tick| >= this
MAX_POINTS = 1008      # ~21 days at 30-min ticks


def load_state(path: str = STATE_PATH) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"coins": {}}


def save_state(state: dict, path: str = STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def fetch_snapshot(addr: str, timeout: int = 20) -> dict | None:
    """Full-venue snapshot from DexScreener (best pair by liquidity)."""
    url = f"https://api.dexscreener.com/latest/dex/tokens/{urllib.parse.quote(addr)}"
    try:
        out = subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        data = json.loads(out.stdout or "{}")
    except Exception:
        return None
    pairs = data.get("pairs") or []
    if not pairs:
        return None
    best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)
    try:
        price = float(best["priceUsd"])
    except (KeyError, TypeError, ValueError):
        return None
    chg = best.get("priceChange") or {}
    txns = best.get("txns") or {}
    vol = best.get("volume") or {}
    m5 = txns.get("m5") or {}
    h1 = txns.get("h1") or {}
    try:
        mcap = best.get("marketCap")
        mcap = float(mcap) if mcap is not None else None
        liq = (best.get("liquidity") or {}).get("usd")
        liq = float(liq) if liq is not None else None
    except (TypeError, ValueError):
        mcap, liq = None, None
    return {
        "price": price, "mcap": mcap, "liq": liq,
        "chg_5m": chg.get("m5"), "chg_1h": chg.get("h1"), "chg_24h": chg.get("h24"),
        "vol_5m": vol.get("m5"), "vol_1h": vol.get("h1"), "vol_24h": vol.get("h24"),
        "buys_5m": m5.get("buys"), "sells_5m": m5.get("sells"),
        "buys_1h": h1.get("buys"), "sells_1h": h1.get("sells"),
        "dex": best.get("dexId"), "pair": best.get("pairAddress"),
    }


def _send_telegram(text: str) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        subprocess.run(
            [sys.executable, os.path.join(here, "telegram_notify.py"), text],
            capture_output=True, timeout=60,
        )
    except Exception:
        pass


def _fmt_usd(x) -> str:
    if x is None:
        return "?"
    if x >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"${x/1_000:.1f}k"
    return f"${x:.2f}"


def cmd_add(args) -> int:
    state = load_state()
    addr = args.address
    coins = state.setdefault("coins", {})
    if addr in coins:
        print(f"already watching {addr}")
        return 0
    snap = fetch_snapshot(addr)
    now = time.time()
    coins[addr] = {
        "label": args.label or addr[:8],
        "added_ts": now,
        "base_price": snap["price"] if snap else None,
        "series": [],
    }
    if snap:
        coins[addr]["series"].append({"ts": now, **snap})
    save_state(state)
    print(f"watching {coins[addr]['label']} ({addr[:8]}…) "
          f"@ {_fmt_usd(snap['mcap']) if snap else '?'} mcap" if snap else "watching (no snapshot yet)")
    return 0


def cmd_tick(args) -> int:
    state = load_state()
    coins = state.get("coins", {})
    now = time.time()
    for addr, coin in coins.items():
        snap = fetch_snapshot(addr)
        if not snap:
            continue
        series = coin.setdefault("series", [])
        prev = series[-1] if series else None
        series.append({"ts": now, **snap})
        del series[:-MAX_POINTS]
        # violent-move alert vs previous tick
        if prev and prev.get("price") and snap["price"]:
            move = (snap["price"] / prev["price"] - 1) * 100
            if abs(move) >= ALERT_MOVE_PCT:
                arrow = "🚀" if move > 0 else "📉"
                _send_telegram(
                    f"{arrow} <b>{coin['label']}</b> moved {move:+.0f}% since last check\n"
                    f"mcap {_fmt_usd(snap['mcap'])} | 5m {_fmt_usd(snap['vol_5m'])} vol\n"
                    f"<code>{addr}</code>")
                coin["last_alert_ts"] = now
    save_state(state)
    print(f"ticked {len(coins)} coin(s)")
    return 0


def cmd_report(args) -> int:
    state = load_state()
    coins = state.get("coins", {})
    if not coins:
        print("no coins watched")
        return 0
    for addr, coin in coins.items():
        series = coin.get("series", [])
        label = coin.get("label", addr[:8])
        print(f"== {label} ({addr[:8]}…) — {len(series)} points ==")
        if not series:
            print("  no data yet")
            continue
        first, last = series[0], series[-1]
        base = coin.get("base_price") or first.get("price")
        if base and last.get("price"):
            print(f"  since watch start: {(last['price']/base-1)*100:+.1f}% "
                  f"| mcap {_fmt_usd(last.get('mcap'))} | liq {_fmt_usd(last.get('liq'))}")
        peak = max((p.get("price") or 0) for p in series)
        if peak and last.get("price"):
            print(f"  vs series peak: {(last['price']/peak-1)*100:+.1f}%")
        b5, s5 = last.get("buys_5m"), last.get("sells_5m")
        if b5 is not None and s5:
            print(f"  5m flow: {b5}/{s5} buys/sells (edge {b5/s5:.2f}x) "
                  f"| 5m vol {_fmt_usd(last.get('vol_5m'))} | 5m chg {last.get('chg_5m')}")
    return 0


def cmd_remove(args) -> int:
    state = load_state()
    if args.address in state.get("coins", {}):
        del state["coins"][args.address]
        save_state(state)
        print("removed")
    else:
        print("not watched")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch specific coins for data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("address")
    a.add_argument("--label", default="")
    sub.add_parser("tick")
    sub.add_parser("report")
    r = sub.add_parser("remove")
    r.add_argument("address")
    args = ap.parse_args()
    return {"add": cmd_add, "tick": cmd_tick, "report": cmd_report,
            "remove": cmd_remove}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
