#!/usr/bin/env python3
"""
FENRIR user watch — d3g3n is the sight layer.

Tracks contracts he drops (chat, groups) and pings on RE-IGNITION via the
second_life filter: a survived coin lifting off its own base. This is the
delivery vehicle for the SAPLING model — the scout can't catch a coin with no
momentum, but a watch on a coin d3g3n already vetted can catch its second leg.

State: ~/workspace/goals/token-scout-watch/hidden_files/user_watch.json

Telegram sends are NOT implemented here — alerts stay paused per d3g3n's
2026-09-30 standing order. `tick` prints a machine-readable report; the cron
handoff surfaces re-ignitions in chat.
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

DEFAULT_STATE = os.path.expanduser(
    "~/workspace/goals/token-scout-watch/hidden_files/user_watch.json"
)


def _load_state(path: str) -> dict:
    try:
        with open(path) as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def add_entry(
    address: str,
    symbol: str = "???",
    chain: str = "solana",
    pair_address: str | None = None,
    price_usd: float = 0.0,
    mcap_usd: float = 0.0,
    note: str = "",
    source: str = "manual",
    state_path: str = DEFAULT_STATE,
) -> dict:
    """Idempotent add. Returns the entry."""
    state = _load_state(state_path)
    key = address.lower()
    now = time.time()
    entry: dict = state.get(key, {})
    entry.update(
        {
            "address": address,
            "symbol": symbol,
            "chain": chain,
            "pair_address": pair_address,
            "added_at": entry.get("added_at", now),
            "added_by": entry.get("added_by", source),
            "note": note or entry.get("note", ""),
            "entry_price_usd": entry.get("entry_price_usd", price_usd),
            "entry_mcap_usd": entry.get("entry_mcap_usd", mcap_usd),
            "last_price_usd": price_usd,
            "last_mcap_usd": mcap_usd,
            "last_check": now,
            "reignition_alerted": entry.get("reignition_alerted", False),
            "reignition_hits": entry.get("reignition_hits", 0),
        }
    )
    state[key] = entry
    _save_state(state_path, state)
    return entry


async def _tick_one(entry: dict) -> dict:
    from fenrir.discovery.filters import FilterEngine, FilterName
    from fenrir.discovery.models import Chain
    from fenrir.discovery.providers.dexscreener import DexScreenerProvider
    from fenrir.discovery.second_life import attach_baseline, oldest_pool

    result: dict = {
        "address": entry["address"],
        "symbol": entry.get("symbol", "???"),
        "ok": False,
    }
    ds = DexScreenerProvider(timeout_seconds=15)
    try:
        chain = Chain(entry.get("chain", "solana"))
        snap = await ds.fetch_snapshot(entry["address"], chain=chain)
    except Exception as e:  # noqa: BLE001 - per-coin fail-open
        result["error"] = f"snapshot failed: {e}"
        return result
    finally:
        await ds.close()
    if snap is None:
        result["error"] = "no liquid pair"
        return result
    # Baseline needs the OLDEST pool (the snapshot's pair is the most liquid
    # now — often hours old on a migrated runner). Cache the resolution, and
    # correct the token age from the oldest pool's creation time.
    pair_override = entry.get("history_pair_address")
    if not pair_override and (snap.age_minutes or 0) < 3 * 24 * 60:
        try:
            res = await oldest_pool(chain, entry["address"])
        except Exception:  # noqa: BLE001 - fail-open
            res = None
        if res:
            pair_override, created_ms = res
            entry["history_pair_address"] = pair_override
            true_age_m = (time.time() * 1000 - created_ms) / 60000.0
            if true_age_m > (snap.age_minutes or 0):
                snap.age_minutes = true_age_m
    try:
        snap = await attach_baseline(snap, pair_override=pair_override)
    except Exception as e:  # noqa: BLE001 - baseline fail-open
        result["baseline_error"] = str(e)
    engine = FilterEngine()
    fr = engine.evaluate(snap, FilterName.SECOND_LIFE)
    result.update(
        {
            "ok": True,
            "price_usd": snap.price_usd,
            "mcap_usd": snap.market_cap_usd,
            "passed": fr.passed,
            "failures": fr.failures,
            "warnings": fr.warnings,
            "baseline_attached": snap.base_floor_price_usd is not None,
        }
    )
    entry["symbol"] = snap.symbol
    entry["last_price_usd"] = snap.price_usd
    entry["last_mcap_usd"] = snap.market_cap_usd
    entry["last_check"] = time.time()
    if fr.passed:
        entry["reignition_hits"] = entry.get("reignition_hits", 0) + 1
    result["already_alerted"] = bool(entry.get("reignition_alerted"))
    return result


async def _amain_tick(state_path: str) -> dict:
    state = _load_state(state_path)
    fired: list[dict] = []
    checked = 0
    for key, entry in state.items():
        checked += 1
        r = await _tick_one(entry)
        if r.get("passed") and not r.get("already_alerted"):
            entry["reignition_alerted"] = True
            fired.append(r)
    _save_state(state_path, state)
    return {"checked": checked, "fired": fired, "ts": time.time()}


def main() -> int:
    ap = argparse.ArgumentParser(description="FENRIR user watch (second-life re-ignition)")
    ap.add_argument("--state", default=DEFAULT_STATE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="track a contract")
    p_add.add_argument("address")
    p_add.add_argument("--note", default="")
    p_add.add_argument("--source", default="manual")

    p_rm = sub.add_parser("remove", help="untrack a contract")
    p_rm.add_argument("address")

    sub.add_parser("list", help="show tracked contracts")
    p_tick = sub.add_parser("tick", help="re-price all tracked, check second_life")
    p_tick.add_argument("--json", action="store_true")

    args = ap.parse_args()

    if args.cmd == "add":
        entry = add_entry(args.address, note=args.note, source=args.source, state_path=args.state)
        print(json.dumps({"added": entry["symbol"], "address": entry["address"]}))
        return 0
    if args.cmd == "remove":
        state = _load_state(args.state)
        gone = state.pop(args.address.lower(), None)
        _save_state(args.state, state)
        print(json.dumps({"removed": gone is not None}))
        return 0
    if args.cmd == "list":
        state = _load_state(args.state)
        print(
            json.dumps(
                [
                    {
                        "symbol": e.get("symbol"),
                        "address": e.get("address"),
                        "entry_mcap": e.get("entry_mcap_usd"),
                        "last_mcap": e.get("last_mcap_usd"),
                        "reignition_alerted": e.get("reignition_alerted"),
                        "note": e.get("note"),
                    }
                    for e in state.values()
                ],
                indent=2,
            )
        )
        return 0
    if args.cmd == "tick":
        report = asyncio.run(_amain_tick(args.state))
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"checked {report['checked']} coin(s)")
            for f in report["fired"]:
                print(
                    f"🔥 RE-IGNITION: {f['symbol']} ${f['mcap_usd']:,.0f} mcap "
                    f"({f['address'][:12]}…)"
                )
            if not report["fired"]:
                print("no re-ignitions")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
