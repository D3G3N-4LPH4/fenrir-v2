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
  python tools/coin_watch.py tag <address> --set si-pvp   # group coins into a named set
  python tools/coin_watch.py pvp si-pvp  # side-by-side PVP set view (volume rotation)
  python tools/coin_watch.py remove <address>

PVP sets: coins sharing a ticker/narrative fight for the same capital. The pvp
view aligns their series tick-by-tick and shows each coin's share of the set's
5m volume plus the mcap leader per tick — the rotation signature of a PVP
battle (wild volume swings until a winner emerges or all die out).

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
MAX_POINTS = 5040       # ~7 days at 2-minute live cadence
JUPITER_SEARCH = "https://lite-api.jup.ag/tokens/v2/search"


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


def fetch_holders(addr: str, timeout: int = 10) -> tuple[int | None, float | None]:
    """Best-effort holder count + top-holder % from Jupiter's keyless search.

    Fail-open: returns (None, None) on any error — the series keeps its
    price/volume/flow data regardless.
    """
    url = f"{JUPITER_SEARCH}?query={urllib.parse.quote(addr)}"
    try:
        out = subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        data = json.loads(out.stdout or "null")
    except Exception:
        return None, None
    toks = data if isinstance(data, list) else (data.get("tokens") or [])
    if not toks or not isinstance(toks[0], dict):
        return None, None
    tok = toks[0]
    holders, top_pct = None, None
    try:
        if tok.get("holderCount") is not None:
            holders = int(tok["holderCount"])
    except (TypeError, ValueError):
        pass
    try:
        audit = tok.get("audit") or {}
        if audit.get("topHoldersPercentage") is not None:
            top_pct = float(audit["topHoldersPercentage"])
    except (TypeError, ValueError):
        pass
    return holders, top_pct


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
        "alerts": not args.no_alerts,
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
        holders, top_pct = fetch_holders(addr)
        snap["holders"] = holders
        snap["top_holder_pct"] = top_pct
        series = coin.setdefault("series", [])
        prev = series[-1] if series else None
        series.append({"ts": now, **snap})
        del series[:-MAX_POINTS]
        # violent-move alert vs previous tick (skipped for data-only coins)
        if coin.get("alerts", True) and prev and prev.get("price") and snap["price"]:
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
        mcaps = [p.get("mcap") for p in series if p.get("mcap")]
        if peak and last.get("price"):
            rng = (f" | session mcap range {_fmt_usd(min(mcaps))}–{_fmt_usd(max(mcaps))}"
                   if mcaps else "")
            print(f"  vs series peak: {(last['price']/peak-1)*100:+.1f}%{rng}")
        b5, s5 = last.get("buys_5m"), last.get("sells_5m")
        if b5 is not None and s5:
            print(f"  5m flow: {b5}/{s5} buys/sells (edge {b5/s5:.2f}x) "
                  f"| 5m vol {_fmt_usd(last.get('vol_5m'))} | 5m chg {last.get('chg_5m')}")
        h_first = next((p.get("holders") for p in series if p.get("holders")), None)
        h_last = last.get("holders")
        if h_first and h_last:
            th = last.get("top_holder_pct")
            th_s = f"{th:.1f}%" if isinstance(th, (int, float)) else "?"
            print(f"  holders: {h_first} → {h_last} ({(h_last/h_first-1)*100:+.1f}%) "
                  f"| top holder {th_s}")
    return 0


def cmd_alerts(args) -> int:
    state = load_state()
    coin = state.get("coins", {}).get(args.address)
    if not coin:
        print("not watched")
        return 1
    coin["alerts"] = args.state == "on"
    save_state(state)
    print(f"alerts {'on' if coin['alerts'] else 'off'} for {coin.get('label')}")
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


def cmd_tag(args) -> int:
    state = load_state()
    coin = state.get("coins", {}).get(args.address)
    if not coin:
        print("not watched")
        return 1
    coin["set"] = args.set_name or None
    save_state(state)
    print(f"{coin.get('label')}: set={coin.get('set')}")
    return 0


def _local_hm(ts: float) -> str:
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        return datetime.fromtimestamp(ts, tz=ZoneInfo("America/Los_Angeles")).strftime("%H:%M")
    except Exception:
        return time.strftime("%H:%M", time.localtime(ts))


def cmd_pvp(args) -> int:
    """Side-by-side PVP set view: per-tick mcap, 5m volume, volume share of the
    set, buy/sell edge, and the mcap leader — the rotation signature."""
    state = load_state()
    members = [(addr, c) for addr, c in state.get("coins", {}).items()
               if c.get("set") == args.set_name and c.get("series")]
    if not members:
        print(f"no coins in set '{args.set_name}'")
        return 1
    members.sort(key=lambda ac: ac[1].get("label", ""))
    labels = [c.get("label", a[:8]) for a, c in members]
    n = min(len(c["series"]) for _, c in members)
    show = max(1, min(args.last, n))
    print(f"== PVP '{args.set_name}' — {len(members)} coins, last {show} ticks ==")
    lead_wins: dict[str, int] = {lb: 0 for lb in labels}
    shares: dict[str, list[float]] = {lb: [] for lb in labels}
    for i in range(n - show, n):
        pts = [(lb, c["series"][i]) for lb, (_, c) in zip(labels, members)]
        ts = pts[0][1].get("ts", 0)
        vols = [(lb, p.get("vol_5m") or 0) for lb, p in pts]
        tot_vol = sum(v for _, v in vols) or 1
        mcaps = [(lb, p.get("mcap") or 0) for lb, p in pts]
        leader = max(mcaps, key=lambda x: x[1])[0]
        lead_wins[leader] += 1
        print(f"-- {_local_hm(ts)}  leader: {leader} | set 5m vol {_fmt_usd(tot_vol)}")
        for lb, p in pts:
            v = p.get("vol_5m") or 0
            share = v / tot_vol * 100
            shares[lb].append(share)
            b5, s5 = p.get("buys_5m"), p.get("sells_5m")
            edge = f"{b5/s5:.2f}x" if b5 is not None and s5 else "?"
            chg = p.get("chg_5m")
            chg_s = f"{chg:+.1f}" if isinstance(chg, (int, float)) else "?"
            print(f"   {lb[:10]:<10} mcap {_fmt_usd(p.get('mcap')):>8} "
                  f"vol5m {_fmt_usd(v):>8} share {share:5.1f}% "
                  f"edge {edge:>6} chg5m {chg_s}")
    print("-- session --")
    for lb, (_, c) in zip(labels, members):
        mcaps = [p.get("mcap") or 0 for p in c["series"]]
        sh = shares[lb]
        swing = f"{min(sh):.0f}–{max(sh):.0f}%" if sh else "?"
        print(f"   {lb[:10]:<10} peak {_fmt_usd(max(mcaps)):>8} "
              f"now {_fmt_usd(mcaps[-1]):>8} led {lead_wins[lb]}/{show} ticks "
              f"vol-share swing {swing}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch specific coins for data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("address")
    a.add_argument("--label", default="")
    a.add_argument("--no-alerts", action="store_true",
                   help="data-only: never Telegram-ping for this coin")
    al = sub.add_parser("alerts")
    al.add_argument("address")
    al.add_argument("state", choices=["on", "off"])
    sub.add_parser("tick")
    sub.add_parser("report")
    r = sub.add_parser("remove")
    r.add_argument("address")
    t = sub.add_parser("tag")
    t.add_argument("address")
    t.add_argument("--set-name", default="")
    p = sub.add_parser("pvp")
    p.add_argument("set_name")
    p.add_argument("--last", type=int, default=10)
    args = ap.parse_args()
    return {"add": cmd_add, "tick": cmd_tick, "report": cmd_report,
            "remove": cmd_remove, "alerts": cmd_alerts,
            "tag": cmd_tag, "pvp": cmd_pvp}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
