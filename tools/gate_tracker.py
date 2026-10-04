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
  tick [--state <path>] [--notify]
      Re-price every tracked token via DexScreener (highest-liquidity
      pair), append a tick, and print JSON: per-token status plus
      "movers" — tokens newly crossing +100% (pump_100) or -50%
      (dump_50) since clearance — and "nudges": newly crossed +25/+50/+100%
      take-profit levels eligible for a Telegram follow-up. One-time keys,
      so each crossing surfaces exactly once. With --notify, the nudges are
      sent straight to Telegram (late-tier candidates, never alerted in the
      first place, are excluded).
  report [--state <path>] [--days N] [--min-ticks N] [--by-regime]
      Scorecard table sorted by % change since clearance: symbol, chain,
      clearance date, price then -> now, % change, peak %, trough %, days
      tracked, score, filters, playbooks, source, entry tier, SOL regime at
      alert time, and the take-profit levels (+25/+50/+100) each token
      crossed first. --by-regime groups outcomes by SOL regime, then filter —
      the regime overlay's measurement view.

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

# Repo root on sys.path so tools can use the fenrir package when run as scripts.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from fenrir.discovery.regime import REGIMES, current_regime as _current_regime
    from fenrir.discovery.entry_tier import tier_alerts as _tier_alerts
except Exception:  # noqa: BLE001 - regime/tier are fail-open tagging; never break the tracker

    def _current_regime() -> str:
        return "unknown"

    REGIMES = ("trend_up", "chop", "trend_down", "unknown")

    def _tier_alerts(tier: str) -> bool:
        return tier != "late"


DEFAULT_STATE = os.path.expanduser(
    "~/workspace/goals/token-scout-watch/hidden_files/gate_tracker/tracked.json"
)

PUMP_THRESHOLD_PCT = 100.0  # newly crossed -> "movers" (one-time)
DUMP_THRESHOLD_PCT = -50.0  # newly crossed -> "movers" (one-time)

# Take-profit first-crossings (2026-10-03): the scout's alerts catch moves
# but holding kills the basket (median -96% vs 36% peaking >=+50%). Each
# token records the first tick it crossed +25/+50/+100% so reviews measure
# what was actually exitable — capturable PnL, not hold-to-now.
EXIT_LEVELS_PCT = (("tp_25", 25.0), ("tp_50", 50.0), ("tp_100", 100.0))

# Exit follow-up nudges (2026-10-03): when a tracked alert crosses a TP
# level, Telegram gets an actionable nudge. Late-tier candidates were never
# alerted, so they never get nudges either (tier_alerts gate).
NUDGE_COPY = {
    "tp_25": ("🛡️", "move stop to entry — ride free from here"),
    "tp_50": ("⚔️", "take half, let the rest ride"),
    "tp_100": ("🐺", "take the rest or trail tight — don't give it back"),
}


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
    regime_now: str | None = None  # lazy: one SOL-regime fetch per batch, fail-open
    for c in cands:
        addr = c.get("address")
        if not addr or addr in state:
            continue  # first clearance wins; never re-stamp
        price = c.get("price_usd")
        mcap = c.get("market_cap_usd")
        if price is None:
            price, mcap = fetch_price(addr)
        regime = c.get("regime")
        if regime not in REGIMES:
            if regime_now is None:
                regime_now = _current_regime()
            regime = regime_now
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
            "entry_tier": c.get("entry_tier") or "standard",
            "regime": regime,
            "ticks": [],
            "alerted_moves": [],
            "exit_crossings": {},
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
    nudges = []
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
        exits = rec.setdefault("exit_crossings", {})
        if chg is not None:
            for key, level in EXIT_LEVELS_PCT:
                if chg >= level and key not in exits:
                    exits[key] = ts
                    if _tier_alerts(rec.get("entry_tier") or "standard"):
                        nudges.append(
                            {
                                "address": addr,
                                "symbol": rec.get("symbol") or "?",
                                "level": key,
                                "level_pct": level,
                                "chg_pct": round(chg, 1),
                                "dexscreener": rec.get("dexscreener"),
                            }
                        )
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
    return {"ticked": len(state), "movers": movers, "nudges": nudges, "ts": ts}


def format_nudge(n: dict) -> str:
    """One actionable Telegram line for a TP crossing. Bot-facing copy."""
    emoji, guidance = NUDGE_COPY[n["level"]]
    sym = n.get("symbol") or "?"
    text = (
        f"{emoji} {sym} +{n['level_pct']:.0f}% from the alert "
        f"(now {n['chg_pct']:+.1f}%) — {guidance}."
    )
    if n.get("dexscreener"):
        text += f"\n{n['dexscreener']}"
    return text


def send_nudges(nudges: list[dict]) -> dict:
    """Send TP nudge messages to every Telegram chat. Fail-open; returns counts.

    telegram_notify fans out to TELEGRAM_CHAT_IDS itself. Late-tier
    candidates never reach here (tick_state gates on tier_alerts).
    """
    result = {"sent": 0, "failed": 0}
    if not nudges:
        return result
    try:
        import telegram_notify

        env = telegram_notify.load_env(".env")
        token = env.get("TELEGRAM_BOT_TOKEN", "")
        raw_ids = env.get("TELEGRAM_CHAT_IDS", "") or env.get("TELEGRAM_CHAT_ID", "")
        chat_ids = [c.strip() for c in raw_ids.split(",") if c.strip()]
    except Exception:  # noqa: BLE001 - fail-open; the tick must survive
        return result
    if not token or not chat_ids:
        return result
    for n in nudges:
        text = format_nudge(n)
        ok = True
        for cid in chat_ids:
            try:
                resp = telegram_notify.send_message(token, cid, text)
                ok = ok and bool(resp.get("ok"))
            except Exception:  # noqa: BLE001 - per-chat fail-open
                ok = False
        result["sent" if ok else "failed"] += 1
    return result


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
        "entry_tier": rec.get("entry_tier") or "standard",
        "regime": rec.get("regime") or "unknown",
        "exit_crossings": dict(rec.get("exit_crossings") or {}),
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
        tier = r.get("entry_tier") or "standard"
        if tier != "standard":
            ctx += f" · {tier}"
        regime = r.get("regime") or "unknown"
        ctx += f" · {regime}"
        exits = r.get("exit_crossings") or {}
        if exits:
            hit = "/".join(
                k.replace("tp_", "+") for k in ("tp_25", "tp_50", "tp_100") if k in exits
            )
            if hit:
                ctx += f" · TP {hit}"
        lines.append(
            f"{r['symbol'][:10]:<10} {r['chain'][:9]:<9} {r['cleared']:<11} "
            f"{move:<27} {_fmt_pct(r['chg_pct']):>8} {_fmt_pct(r['peak_pct']):>8} "
            f"{_fmt_pct(r['trough_pct']):>8} {r['days']:>4.1f} "
            f"{(r['score'] if r['score'] is not None else '?'):>5}  {ctx}"
        )
    return "\n".join(lines)


def render_by_regime(state: dict, now: float) -> str:
    """Outcome table grouped by SOL regime at alert time, then filter.

    The regime overlay's measurement view: does the same filter behave
    differently in trend_up vs chop vs trend_down? Level 1 is tag-only —
    this table is the evidence future suppression/scoring decisions work from.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for rec in state.values():
        regime = rec.get("regime") or "unknown"
        filters = rec.get("filters") or ["(none)"]
        row = summarize(rec, now)
        for f in filters:
            groups.setdefault((regime, f), []).append(row)

    def stats(rows: list[dict]) -> tuple[int, str, str, str]:
        priced = [r for r in rows if r["chg_pct"] is not None]
        n = len(rows)
        if not priced:
            return n, "?", "?", "?"
        green = sum(1 for r in priced if (r["chg_pct"] or 0) > 0)
        chgs = sorted(r["chg_pct"] or 0 for r in priced)
        median = chgs[len(chgs) // 2]
        peak50 = sum(1 for r in priced if (r["peak_pct"] or 0) >= 50)
        return (
            n,
            f"{100.0 * green / len(priced):.0f}%",
            _fmt_pct(median),
            f"{100.0 * peak50 / len(priced):.0f}%",
        )

    lines = []
    lines.append("OUTCOMES BY SOL REGIME AT ALERT TIME (then filter)")
    lines.append("")
    lines.append(
        f"{'regime':<11} {'filter':<20} {'n':>3} {'green':>5} {'median':>8} {'peak50%':>7}"
    )
    lines.append("-" * 60)
    for regime in REGIMES:
        reg_groups = sorted(
            ((f, rows) for (rg, f), rows in groups.items() if rg == regime),
            key=lambda kv: -len(kv[1]),
        )
        if not reg_groups:
            continue
        for f, rows in reg_groups:
            n, green, median, peak50 = stats(rows)
            lines.append(f"{regime:<11} {f[:20]:<20} {n:>3} {green:>5} {median:>8} {peak50:>7}")
        lines.append("")
    lines.append("green = % priced tokens green now · median = median move since clearance")
    lines.append("peak50% = % that peaked >=+50% at some point (capturable)")
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
    if args.notify:
        result["notify"] = send_nudges(result.get("nudges") or [])
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
    if args.by_regime:
        print(render_by_regime(state, now))
        print(f"\n{len(rows)} token(s) tracked")
        return 0
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
    t.add_argument(
        "--notify",
        action="store_true",
        help="send Telegram nudges for newly crossed +25/+50/+100% levels",
    )

    rep = sub.add_parser("report", help="scorecard: move since gate clearance")
    rep.add_argument(
        "--days", type=float, default=0, help="only tokens cleared within the last N days (0=all)"
    )
    rep.add_argument("--min-ticks", type=int, default=0)
    rep.add_argument("--json", action="store_true")
    rep.add_argument(
        "--by-regime",
        action="store_true",
        help="outcomes grouped by SOL regime at alert time, then filter",
    )

    args = ap.parse_args()
    if args.cmd == "record":
        return cmd_record(args)
    if args.cmd == "tick":
        return cmd_tick(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
