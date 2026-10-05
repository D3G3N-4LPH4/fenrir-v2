#!/usr/bin/env python3
"""Misfit watch — track the tokens the gates rejected.

Every scout run surfaces candidates that score well (``>= MISFIT_MIN_SCORE``,
the "WORTH A LOOK" band) but fit no FENRIR entry filter. They used to be
dropped silently — TERMINAL ran +1055% after being dropped, because no
filter fit and nothing tracked it.

This tool records those misfits with their first-seen price AND keeps
re-pricing them. Movers (newly +100% / -50% vs first-seen) surface exactly
once — the measurement loop for the rejected cohort, mirroring the gate
tracker for the alerted one.

Commands:
  record --scout-output <json> [--state <path>] [--ts <epoch>]
      Record every misfit in a scout output file (its "misfits" list), or a
      plain JSON list of misfit dicts. Already-tracked addresses are left
      alone (first sighting wins). Missing prices are fetched once from
      DexScreener.
  tick [--state <path>] [--notify]
      Re-price every tracked misfit via DexScreener (highest-liquidity
      pair), append a tick, and print JSON with "movers" — misfits newly
      crossing +100% (pump_100) or -50% (dump_50) since first-seen.
      One-time keys, so each crossing surfaces exactly once. With --notify,
      mover flags are sent straight to Telegram.
  report [--state <path>] [--json]
      Scorecard table sorted by % change since first-seen: symbol, chain,
      first-seen date, price then -> now, % change, peak %, trough %, days
      tracked, score, source.

State file: JSON dict addr -> record. Default state path is the
token-scout-watch goal's hidden_files/misfit_watch/tracked.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

# Repo root on sys.path so tools can use the fenrir package when run as scripts.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from gate_tracker import (  # noqa: E402
    fetch_price,
    load_state,
    pct_change,
    save_state,
)

DEFAULT_STATE = os.path.expanduser(
    "~/workspace/goals/token-scout-watch/hidden_files/misfit_watch/tracked.json"
)

PUMP_THRESHOLD_PCT = 100.0  # newly crossed -> "movers" (one-time)
DUMP_THRESHOLD_PCT = -50.0  # newly crossed -> "movers" (one-time)

MOVER_COPY = {
    "pump_100": ("🚀", "pumped"),
    "dump_50": ("🩸", "dumped"),
}


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------


def _score_overall(c: dict) -> float | None:
    s = c.get("score")
    if isinstance(s, dict):
        return s.get("overall")
    return s if isinstance(s, int | float) else None


def record_misfits(cands: list[dict], state: dict, ts: float) -> list[str]:
    """Stamp first-seen for new misfits. Returns addresses recorded."""
    recorded = []
    for c in cands:
        addr = c.get("address")
        if not addr:
            continue
        # Normalize EVM addresses to lowercase: a checksummed 0x address and
        # its lowercase form are the same token and must never double-track
        # (this exact bug double-fired a Telegram flag on 2026-10-04).
        # Solana base58 addresses are case-sensitive — leave them alone.
        if addr.startswith("0x"):
            addr = addr.lower()
        if addr in state:
            continue  # first sighting wins; never re-stamp
        price = c.get("price_usd")
        mcap = c.get("market_cap_usd")
        if price is None:
            price, mcap = fetch_price(addr)
        state[addr] = {
            "address": addr,
            "symbol": c.get("symbol"),
            "name": c.get("name"),
            "chain": c.get("chain"),
            "first_seen": ts,
            "first_price": price,
            "first_mcap": mcap,
            "first_liq": c.get("liquidity_usd"),
            "score": _score_overall(c),
            "source": c.get("source"),
            "dexscreener": c.get("dexscreener"),
            "safety_unknown": bool(c.get("safety_unknown")),
            "filter_warnings": list(c.get("filter_warnings") or []),
            "ticks": [],
            "alerted_moves": [],
        }
        recorded.append(addr)
    return recorded


# --------------------------------------------------------------------------
# tick
# --------------------------------------------------------------------------


def tick_state(state: dict, ts: float) -> dict:
    """Re-price every tracked misfit; append ticks; detect new big movers."""
    movers = []
    for addr, rec in state.items():
        price, mcap = fetch_price(addr)
        tick: dict[str, Any] = {"ts": ts, "price": price, "mcap": mcap}
        rec.setdefault("ticks", []).append(tick)
        if price is None:
            tick["note"] = "no pair data"
            continue
        chg = pct_change(price, rec.get("first_price"))
        tick["chg_pct"] = round(chg, 2) if chg is not None else None
        if chg is None:
            continue
        alerted = rec.setdefault("alerted_moves", [])
        if chg >= PUMP_THRESHOLD_PCT and "pump_100" not in alerted:
            alerted.append("pump_100")
            movers.append(
                {
                    "address": addr,
                    "symbol": rec.get("symbol"),
                    "kind": "pump_100",
                    "chg_pct": round(chg, 1),
                    "dexscreener": rec.get("dexscreener"),
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
                    "dexscreener": rec.get("dexscreener"),
                }
            )
    return {"ticked": len(state), "movers": movers, "ts": ts}


def format_mover(m: dict) -> str:
    """One Telegram line flagging a misfit mover. Bot-facing copy."""
    emoji, verb = MOVER_COPY[m["kind"]]
    sym = m.get("symbol") or "?"
    text = (
        f"{emoji} Misfit {verb}: {sym} {m['chg_pct']:+.1f}% since first seen — "
        f"scored well but fit no entry filter."
    )
    if m.get("dexscreener"):
        text += f"\n{m['dexscreener']}"
    return text


_FLAG_PACE_SECONDS = 1.0  # gap between consecutive sends (Telegram flood-control)


def _load_telegram(notify) -> tuple[str, list[str]]:
    """Read bot token + chat ids from .env. Raises on any problem."""
    env = notify.load_env(".env")
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    raw_ids = env.get("TELEGRAM_CHAT_IDS", "") or env.get("TELEGRAM_CHAT_ID", "")
    chat_ids = [c.strip() for c in raw_ids.split(",") if c.strip()]
    return token, chat_ids


def _post_flag(
    notify, token: str, chat_ids: list[str], text: str, parse_mode: str = ""
) -> tuple[bool, str]:
    """Send one flag to every chat; honor a Telegram 429 retry_after once.

    Returns (ok, detail). Transport failures raise out of send_message only
    after its own retries — caught here so one flag can't kill the batch.
    """
    for cid in chat_ids:
        try:
            resp = notify.send_message(token, cid, text, parse_mode=parse_mode)
        except Exception as e:  # noqa: BLE001 - transport failure after retries
            return False, f"transport error: {type(e).__name__}"
        if resp.get("ok"):
            continue
        params = resp.get("parameters") or {}
        wait = params.get("retry_after")
        err = resp.get("description", "unknown error")
        if isinstance(wait, int | float) and 0 < wait <= 60:
            time.sleep(wait + 0.5)
            try:
                resp = notify.send_message(token, cid, text, parse_mode=parse_mode)
            except Exception as e:  # noqa: BLE001
                return False, f"transport error on retry: {type(e).__name__}"
            if resp.get("ok"):
                continue
            err = resp.get("description", "unknown error")
        return False, f"telegram api error: {err}"
    return True, "sent"


def send_movers(movers: list[dict]) -> dict:
    """Send misfit-mover flags to every Telegram chat. Fail-open.

    Returns counts plus per-symbol sent/failed lists, so a failed flag is
    always identifiable (a 2026-10-04 tick lost 1 of 5 flags to the channel
    with no record of which one). telegram_notify fans out to
    TELEGRAM_CHAT_IDS itself.
    """
    result: dict[str, Any] = {"sent": 0, "failed": 0, "sent_symbols": [], "failed_symbols": []}
    if not movers:
        return result
    try:
        import telegram_notify

        token, chat_ids = _load_telegram(telegram_notify)
    except Exception:  # noqa: BLE001 - fail-open; the tick must survive
        return result
    if not token or not chat_ids:
        return result
    for i, m in enumerate(movers):
        sym = m.get("symbol") or "?"
        ok, detail = _post_flag(telegram_notify, token, chat_ids, format_mover(m))
        if ok:
            result["sent"] += 1
            result["sent_symbols"].append(sym)
        else:
            result["failed"] += 1
            result["failed_symbols"].append(sym)
            print(f"misfit_watch: mover flag FAILED for {sym}: {detail}", file=sys.stderr)
        if i < len(movers) - 1:
            time.sleep(_FLAG_PACE_SECONDS)
    return result


def send_rejection_alerts(cands: list[dict]) -> dict:
    """Alert the moment a misfit is recorded: full scout card, gate-rejected.

    ``cands`` are the newly-recorded misfit dicts (scout candidate schema).
    Fail-open; returns counts plus per-symbol sent/failed lists.
    """
    result: dict[str, Any] = {"sent": 0, "failed": 0, "sent_symbols": [], "failed_symbols": []}
    if not cands:
        return result
    try:
        import telegram_notify

        from fenrir.discovery.alerts import format_scout_alert

        token, chat_ids = _load_telegram(telegram_notify)
    except Exception:  # noqa: BLE001 - fail-open; recording must survive
        return result
    if not token or not chat_ids:
        return result
    for i, c in enumerate(cands):
        sym = c.get("symbol") or c.get("ticker") or "?"
        try:
            text = format_scout_alert(c)
        except Exception as e:  # noqa: BLE001 - one bad card must not kill the batch
            result["failed"] += 1
            result["failed_symbols"].append(sym)
            print(
                f"misfit_watch: rejection card FAILED for {sym}: {type(e).__name__}",
                file=sys.stderr,
            )
            continue
        ok, detail = _post_flag(telegram_notify, token, chat_ids, text, parse_mode="Markdown")
        if ok:
            result["sent"] += 1
            result["sent_symbols"].append(sym)
        else:
            result["failed"] += 1
            result["failed_symbols"].append(sym)
            print(f"misfit_watch: rejection card FAILED for {sym}: {detail}", file=sys.stderr)
        if i < len(cands) - 1:
            time.sleep(_FLAG_PACE_SECONDS)
    return result


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _last_price(rec: dict) -> tuple[float | None, float | None]:
    for t in reversed(rec.get("ticks", [])):
        if t.get("price") is not None:
            return t["price"], t.get("mcap")
    return None, None


def summarize(rec: dict, now: float) -> dict:
    """One-row summary for the report: move stats + misfit context."""
    price_now, mcap_now = _last_price(rec)
    chgs = [t["chg_pct"] for t in rec.get("ticks", []) if t.get("chg_pct") is not None]
    return {
        "symbol": rec.get("symbol") or "?",
        "chain": rec.get("chain") or "?",
        "first_seen": time.strftime("%m-%d %H:%M", time.localtime(rec.get("first_seen", now))),
        "price_then": rec.get("first_price"),
        "price_now": price_now,
        "mcap_now": mcap_now,
        "chg_pct": pct_change(price_now, rec.get("first_price")),
        "peak_pct": max(chgs) if chgs else None,
        "trough_pct": min(chgs) if chgs else None,
        "days": round((now - rec.get("first_seen", now)) / 86400, 1),
        "ticks": len(rec.get("ticks", [])),
        "score": rec.get("score"),
        "source": rec.get("source"),
        "moves": list(rec.get("alerted_moves", [])),
    }


def format_report(rows: list[dict]) -> str:
    lines = [
        f"{'SYM':<10} {'CHAIN':<9} {'SEEN':<11} {'CHG%':>8} {'PEAK%':>8} "
        f"{'TROUGH%':>8} {'DAYS':>5} {'SCORE':>6}  MOVES  SOURCE"
    ]
    for r in rows:
        chg = r["chg_pct"]
        chg_s = f"{chg:+.1f}" if chg is not None else "?"
        peak = f"{r['peak_pct']:+.1f}" if r["peak_pct"] is not None else "?"
        trough = f"{r['trough_pct']:+.1f}" if r["trough_pct"] is not None else "?"
        score = f"{r['score']:.0f}" if r["score"] is not None else "?"
        lines.append(
            f"{r['symbol']:<10.10} {r['chain']:<9.9} {r['first_seen']:<11} "
            f"{chg_s:>8} {peak:>8} {trough:>8} {r['days']:>5} {score:>6}  "
            f"{','.join(r['moves']):<6} {r['source'] or '?'}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_candidates(path: str) -> list[dict]:
    """Accept a scout output file (uses its "misfits" list) or a plain list."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("misfits") or []
    return data if isinstance(data, list) else []


def main() -> int:
    ap = argparse.ArgumentParser(description="Misfit watch — track gate-rejected scorers")
    ap.add_argument("--state", default=DEFAULT_STATE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="stamp first-seen for misfit candidates")
    r.add_argument(
        "--scout-output",
        required=True,
        help="scout.py JSON output (its 'misfits' list), or a JSON list of misfit dicts",
    )
    r.add_argument("--ts", type=float, default=None)
    r.add_argument(
        "--notify",
        action="store_true",
        help="alert Telegram the moment a misfit is recorded (gate-rejected card)",
    )

    t = sub.add_parser("tick", help="re-price all tracked misfits")
    t.add_argument("--notify", action="store_true", help="send mover flags to Telegram")

    rep = sub.add_parser("report", help="scorecard: move since first-seen")
    rep.add_argument("--json", action="store_true")

    args = ap.parse_args()
    state = load_state(args.state)
    now = time.time()

    if args.cmd == "record":
        cands = _load_candidates(args.scout_output)
        recorded = record_misfits(cands, state, args.ts or now)
        save_state(args.state, state)
        out = {"recorded": recorded, "total": len(state)}
        if args.notify:
            fresh = []
            for c in cands:
                a = c.get("address") or ""
                if a.startswith("0x"):
                    a = a.lower()
                if a in recorded:
                    fresh.append(c)
            out["notify"] = send_rejection_alerts(fresh)
        print(json.dumps(out))
    elif args.cmd == "tick":
        out = tick_state(state, now)
        save_state(args.state, state)
        if args.notify:
            movers: list[dict] = out["movers"]  # type: ignore[assignment]
            out["notify"] = send_movers(movers)
        print(json.dumps(out))
    elif args.cmd == "report":
        rows = [summarize(rec, now) for rec in state.values()]
        rows.sort(
            key=lambda r: (r["chg_pct"] is None, -(r["chg_pct"] or 0)),
        )
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            print(format_report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
