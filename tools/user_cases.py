#!/usr/bin/env python3
"""User-shared coin case log — turns d3g3n's shared coins into tuning data.

Commands:
  add --eval-json <file>   log one case from `tools/evaluate.py --json` output
  reprice                  fetch current prices, fill 1h/4h/24h outcome deltas
  report                   print the case table with outcomes

State: ~/workspace/goals/token-scout-watch/hidden_files/user_cases.json

Each case keeps the full share-time snapshot (price, mcap, which filters
passed/failed and why, score) plus outcome buckets filled in as time passes.
This is the qualitative complement to the gate tracker's unbiased clearance
dataset: it captures the coins a human flagged — including near-misses and
post-verticals the automatic pipeline never records. That's exactly the
material filter tuning needs: edge cases with known outcomes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.discovery.providers.dexscreener import DexScreenerProvider  # noqa: E402
from fenrir.discovery.models import Chain  # noqa: E402

STATE = os.path.expanduser("~/workspace/goals/token-scout-watch/hidden_files/user_cases.json")


def _load() -> list:
    try:
        with open(STATE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(cases: list) -> None:
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cases, f, indent=1)
    os.replace(tmp, STATE)


def cmd_add(args: argparse.Namespace) -> int:
    with open(args.eval_json) as f:
        ev = json.load(f)
    cases = _load()
    addr = ev.get("address", "")
    if any(c.get("address") == addr for c in cases):
        print(f"already logged: {addr}")
        return 0
    filters = ev.get("filters", {})
    score = ev.get("score", {})
    cases.append(
        {
            "address": addr,
            "symbol": ev.get("symbol"),
            "chain": ev.get("chain"),
            "shared_at": time.time(),
            "note": args.note or "",
            "at_share": {
                "price_usd": ev.get("price_usd"),
                "market_cap_usd": ev.get("market_cap_usd"),
                "liquidity_usd": ev.get("liquidity_usd"),
                "volume_24h_usd": ev.get("volume_24h_usd"),
            },
            "passed_filters": sorted(k for k, r in filters.items() if r.get("passed")),
            "failed_filters": {
                k: r.get("failures", []) for k, r in filters.items() if not r.get("passed")
            },
            "score": score.get("overall") if isinstance(score, dict) else None,
            "outcomes": {},  # move_1h/move_4h/move_24h filled by reprice
        }
    )
    _save(cases)
    print(f"logged {ev.get('symbol')} ({addr[:10]}…) — {len(cases)} cases")
    return 0


async def _price_now(ds: DexScreenerProvider, addr: str, chain: str) -> float | None:
    try:
        ch = Chain(chain) if chain else None
        snap = await ds.fetch_snapshot(addr, chain=ch)
    except Exception:
        return None
    return snap.price_usd if snap else None


async def cmd_reprice(_args: argparse.Namespace) -> int:
    cases = _load()
    if not cases:
        print("no cases logged")
        return 0
    ds = DexScreenerProvider(timeout_seconds=15)
    now = time.time()
    updated = 0
    try:
        for c in cases:
            px = await _price_now(ds, c["address"], c.get("chain") or "")
            if px is None or not c["at_share"].get("price_usd"):
                await asyncio.sleep(0.4)
                continue
            base = c["at_share"]["price_usd"]
            move = (px - base) / base if base else 0.0
            elapsed_h = (now - c["shared_at"]) / 3600.0
            oc = c.setdefault("outcomes", {})
            for bucket, need_h in (("move_1h", 1), ("move_4h", 4), ("move_24h", 24)):
                if elapsed_h >= need_h and bucket not in oc:
                    oc[bucket] = round(move, 4)
                    updated += 1
            oc["last_price_usd"] = px
            oc["checked_at"] = now
            await asyncio.sleep(0.4)
    finally:
        await ds.close()
    _save(cases)
    print(f"repriced {len(cases)} cases, filled {updated} outcome buckets")
    return 0


def cmd_report(_args: argparse.Namespace) -> int:
    cases = _load()
    if not cases:
        print("no cases logged")
        return 0
    for c in cases:
        age_h = (time.time() - c["shared_at"]) / 3600.0
        oc = c.get("outcomes", {})
        o = (
            " ".join(f"{k}={v:+.0%}" for k, v in oc.items() if k.startswith("move_"))
            or "outcomes pending"
        )
        pf = ",".join(c.get("passed_filters", [])) or "none"
        print(
            f"{c.get('symbol')} [{c.get('chain')}] shared {age_h:.1f}h ago "
            f"score={c.get('score')} passed={pf} | {o}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="user-shared coin case log")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--eval-json", required=True)
    a.add_argument("--note", default="")
    sub.add_parser("reprice")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "add":
        return cmd_add(args)
    if args.cmd == "reprice":
        return asyncio.run(cmd_reprice(args))
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
