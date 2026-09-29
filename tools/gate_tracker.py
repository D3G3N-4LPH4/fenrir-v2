#!/usr/bin/env python3
"""Gate-clearance tracker — the scout's feedback loop.

Every token that clears the scout gates (score >= min-score, >=1 filter,
no hard safety fail) and gets alerted is recorded here with its
gate-clearance price AND the full gate context (which filters passed,
score breakdown, playbooks, source, mcap/liq/vol/age/buys/sells).

A tick cron re-prices tracked tokens on a schedule. The report command
shows each token's move up/down from clearance, with its gate context —
that is the dataset used to tune the scout gates and filter thresholds.

Commands:
  record --candidates <json> [--state <path>] [--ts <epoch>]
      Stamp gate clearance for every candidate in the file (scout's
      candidate schema, or a plain list of candidate dicts). Candidates
      already tracked are left alone (first clearance wins). Missing
      prices are fetched once from DexScreener.
  tick [--state <path>]
      Re-price every tracked token via DexScreener (highest-liquidity
      pair), append a tick, and print JSON: per-token status plus
      "movers" — tokens newly crossing +100% (pump_100) or -50%
      (dump_50) since clearance. One-time keys, so each crossing
      surfaces exactly once.
  report [--state <path>] [--days N] [--min-ticks N]
      Scorecard table sorted by % change since clearance: symbol, chain,
      clearance date, price then -> now, % change, peak %, trough %,
      days tracked, score, filters, playbooks, source.

State file: JSON dict addr -> record. Default state path is the
token-scout-watch goal's hidden_files/gate_tracker/tracked.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
from typing import Any

DEFAULT_STATE = os.path.expanduser(
    "~/workspace/goals/token-scout-watch/hidden_files/gate_tracker/tracked.json"
)

PUMP_THRESHOLD_PCT = 100.0  # newly crossed -> "movers" (one-time)
DUMP_THRESHOLD_PCT = -50.0  # newly crossed -> "movers" (one-time)


# --------------------------------------------------------------------------
# price fetching (DexScreener, highest-liquidity pair — same rule as scout)
# --------------------------------------------------------------------------


def fetch_price(addr: str, timeout: int = 20) -> tuple[float | None, float | None]:
    """Return (price_usd, mcap_usd) or (None, None) on failure/no pairs."""
    url = f"https://api.dexscreener.com/latest/dex/tokens/{urllib.parse.quote(addr)}"
    try:
        out = subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout), url],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        data = json.loads(out.stdout or "{}")
    except Exception:
        return None, None
    pairs = data.get("pairs") or []
    if not pairs:
        return None, None
    best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)
    try:
        price = float(best["priceUsd"])
    except (KeyError, TypeError, ValueError):
        return None, None
    mcap = best.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else None
    except (TypeError, ValueError):
        mcap = None
    return price, mcap


# --------------------------------------------------------------------------
# state helpers
# --------------------------------------------------------------------------


def load_state(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def _playbook_summary(c: dict) -> dict:
    pb = c.get("playbooks") or {}
    tags = pb.get("tags") if isinstance(pb, dict) else None
    if isinstance(tags, dict):
        return {k: round(float(v), 2) for k, v in tags.items() if v}
    if isinstance(pb, dict):
        return {k: round(float(v), 2) for k, v in pb.items() if isinstance(v, (int, float)) and v}
    return {}


def _score_overall(c: dict) -> float | None:
    s = c.get("score")
    if isinstance(s, dict):
        v = s.get("overall")
        return float(v) if isinstance(v, (int, float)) else None
    try:
        return float(s) if s is not None else None
    except (TypeError, ValueError):
        return None


def record_candidates(cands: list[dict], state: dict, ts: float) -> list[str]:
    """Stamp gate clearance for new candidates. Returns addresses recorded."""
    recorded = []
    for c in cands:
        addr = c.get("address")
        if not addr or addr in state:
            continue  # first clearance wins; never re-stamp
        price = c.get("price_usd")
        mcap = c.get("market_cap_usd")
        if price is None:
            price, mcap = fetch_price(addr)
        state[addr] = {
            "address": addr,
            "symbol": c.get("symbol"),
            "name": c.get("name"),
            "chain": c.get("chain"),
            "cleared_at": ts,
            "clearance_price": price,
            "clearance_mcap": mcap,
            "clearance_liq": c.get("liquidity_usd"),
            "score": _score_overall(c),
            "filters": list(c.get("passed_filters") or []),
            "filter_warnings": list(c.get("filter_warnings") or []),
            "playbooks": _playbook_summary(c),
            "source": c.get("source"),
            "dexscreener": c.get("dexscreener"),
            "safety_unknown": bool(c.get("safety_unknown")),
            "ticks": [],
            "alerted_moves": [],
        }
        recorded.append(addr)
    return recorded


def pct_change(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then <= 0:
        return None
    return (now - then) / then * 100.0


def tick_state(state: dict, ts: float) -> dict:
    """Re-price every tracked token; append ticks; detect new big movers."""
    movers = []
    for addr, rec in state.items():
        price, mcap = fetch_price(addr)
        tick: dict[str, Any] = {"ts": ts, "price": price, "mcap": mcap}
        rec.setdefault("ticks", []).append(tick)
        if price is None:
            tick["note"] = "no pair data"
            continue
        chg = pct_change(price, rec.get("clearance_price"))
        tick["chg_pct"] = round(chg, 2) if chg is not None else None
        alerted = rec.setdefault("alerted_moves", [])
        if chg is not None:
            if chg >= PUMP_THRESHOLD_PCT and "pump_100" not in alerted:
                alerted.append("pump_100")
                movers.append(
                    {
                        "address": addr,
                        "symbol": rec.get("symbol"),
                        "kind": "pump_100",
                        "chg_pct": round(chg, 1),
                    }
                )
            elif chg <= DUMP_THRESHOLD_PCT and "dump_50" not in alerted:
                alerted.append("dump_50")
                movers.append(
                    {
                        "address": addr,
                        "symbol": rec.get("symbol"),
                        "kind": "dump_50",
                        "chg_pct": round(chg, 1),
                    }
                )
    return {"ticked": len(state), "movers": movers, "ts": ts}


def _last_price(rec: dict) -> tuple[float | None, float | None]:
    for t in reversed(rec.get("ticks", [])):
        if t.get("price") is not None:
            return t["price"], t.get("mcap")
    return None, None


def summarize(rec: dict, now: float) -> dict:
    """One-row summary for the report: move stats + gate context."""
    price_now, mcap_now = _last_price(rec)
    chgs = [t["chg_pct"] for t in rec.get("ticks", []) if t.get("chg_pct") is not None]
    return {
        "symbol": rec.get("symbol") or "?",
        "chain": rec.get("chain") or "?",
        "cleared": time.strftime("%m-%d %H:%M", time.localtime(rec.get("cleared_at", now))),
        "price_then": rec.get("clearance_price"),
        "price_now": price_now,
        "chg_pct": pct_change(price_now, rec.get("clearance_price")),
        "peak_pct": max(chgs) if chgs else None,
        "trough_pct": min(chgs) if chgs else None,
        "days": round((now - rec.get("cleared_at", now)) / 86400, 1),
        "ticks": len(rec.get("ticks", [])),
        "score": rec.get("score"),
        "filters": ",".join(rec.get("filters") or []),
        "playbooks": ",".join(rec.get("playbooks") or {}),
        "source": rec.get("source") or "?",
        "dead": price_now is None and bool(rec.get("ticks")),
        "dexscreener": rec.get("dexscreener"),
    }


def _fmt_price(p: float | None) -> str:
    if p is None:
        return "?"
    if p == 0:
        return "0"
    if abs(p) < 0.0001:
        return f"{p:.2e}"
    if abs(p) < 1:
        return f"{p:.6f}".rstrip("0").rstrip(".")
    return f"{p:,.4f}".rstrip("0").rstrip(".")


def _fmt_pct(p: float | None) -> str:
    if p is None:
        return "?"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def render_report(rows: list[dict]) -> str:
    lines = []
    lines.append("GATE-CLEARANCE TRACKER — move since the scout's gates cleared each token")
    lines.append("")
    hdr = (
        f"{'SYM':<10} {'chain':<9} {'cleared':<11} {'price then→now':<27} "
        f"{'%chg':>8} {'peak':>8} {'low':>8} {'d':>4} {'score':>5}  filters/playbooks · source"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        move = f"{_fmt_price(r['price_then'])}→{_fmt_price(r['price_now'])}"
        if r["dead"]:
            move += " ☠"
        ctx = r["filters"]
        if r["playbooks"]:
            ctx += f" [{r['playbooks']}]"
        ctx += f" · {r['source']}"
        lines.append(
            f"{r['symbol'][:10]:<10} {r['chain'][:9]:<9} {r['cleared']:<11} "
            f"{move:<27} {_fmt_pct(r['chg_pct']):>8} {_fmt_pct(r['peak_pct']):>8} "
            f"{_fmt_pct(r['trough_pct']):>8} {r['days']:>4.1f} "
            f"{(r['score'] if r['score'] is not None else '?'):>5}  {ctx}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def cmd_record(args) -> int:
    with open(args.candidates) as f:
        data = json.load(f)
    cands = data.get("candidates", data) if isinstance(data, dict) else data
    if not isinstance(cands, list):
        print(json.dumps({"error": "candidates file must hold a list or {candidates: [...]}"}))
        return 1
    state = load_state(args.state)
    ts = args.ts or time.time()
    recorded = record_candidates(cands, state, ts)
    save_state(args.state, state)
    print(json.dumps({"recorded": recorded, "tracked_total": len(state), "ts": ts}))
    return 0


def cmd_tick(args) -> int:
    state = load_state(args.state)
    result = tick_state(state, time.time())
    save_state(args.state, state)
    print(json.dumps(result))
    return 0


def cmd_report(args) -> int:
    state = load_state(args.state)
    now = time.time()
    rows = []
    for rec in state.values():
        age_days = (now - rec.get("cleared_at", now)) / 86400
        if args.days and age_days > args.days:
            continue
        if args.min_ticks and len(rec.get("ticks", [])) < args.min_ticks:
            continue
        rows.append(summarize(rec, now))
    # biggest winners first; unknowns/dead sink to the bottom
    rows.sort(key=lambda r: (r["chg_pct"] is None, -(r["chg_pct"] or 0)))
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        print(render_report(rows))
        print(f"\n{len(rows)} token(s) tracked")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate-clearance tracker")
    ap.add_argument("--state", default=DEFAULT_STATE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="stamp gate clearance for alerted candidates")
    r.add_argument(
        "--candidates",
        required=True,
        help="JSON file: list of candidate dicts or {candidates: [...]}",
    )
    r.add_argument("--ts", type=float, default=None)

    t = sub.add_parser("tick", help="re-price all tracked tokens")

    rep = sub.add_parser("report", help="scorecard: move since gate clearance")
    rep.add_argument(
        "--days", type=float, default=0, help="only tokens cleared within the last N days (0=all)"
    )
    rep.add_argument("--min-ticks", type=int, default=0)
    rep.add_argument("--json", action="store_true")

    args = ap.parse_args()
    if args.cmd == "record":
        return cmd_record(args)
    if args.cmd == "tick":
        return cmd_tick(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
