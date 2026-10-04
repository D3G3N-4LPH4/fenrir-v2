#!/usr/bin/env python3
"""FENRIR hot watch — the 2-minute fast lane between scout polls.

The 10-minute scout seeds per-token acceleration history
(hidden_files/accel_history.json). Between polls, this tool re-checks the hot
list — tokens with a fresh observation that is already accelerating, plus any
fresh caller-confluence addresses — through the full filter engine and alerts
immediately when one clears the bar, instead of waiting for the next cycle.

Alert-worthy: any entry filter passed AND score >= 60 AND not alerted in the
last 24h (the same dedup contract as the scout's seen.json, which is shared).

State:
  accel_history.json — hot-list source; updated with each fresh observation
  seen.json          — alert dedup, shared with the scout
  confluence.json    — fresh caller-confluence addresses (written by channel_poll)

Sends Telegram alerts itself via tools/telegram_notify.py (Markdown), exactly
like the scout's alerts. Alerted candidates are also written to
/tmp/hot_new_candidates.json for the gate-tracker record step.
Prints a JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.scout import evaluate_address  # noqa: E402
from fenrir.discovery.acceleration import AccelTracker  # noqa: E402
from fenrir.discovery.alerts import format_scout_alert  # noqa: E402
from fenrir.discovery.filters import FilterEngine  # noqa: E402
from fenrir.discovery.playbooks import PlaybookTagger  # noqa: E402
from fenrir.discovery.providers.dexscreener import DexScreenerProvider  # noqa: E402
from fenrir.discovery.providers.geckoterminal import GeckoTerminalProvider  # noqa: E402
from fenrir.discovery.providers.goplus import GoPlusProvider  # noqa: E402
from fenrir.discovery.providers.robinhood_safety import RobinhoodSafetyProvider  # noqa: E402
from fenrir.discovery.scoring import ScoringEngine  # noqa: E402
from fenrir.discovery.seen import load as load_seen  # noqa: E402
from fenrir.discovery.seen import record_alert, save as save_seen  # noqa: E402
from fenrir.discovery.seen import should_alert

GOAL_HIDDEN = os.path.expanduser("~/workspace/goals/token-scout-watch/hidden_files")
SEEN_PATH = os.path.join(GOAL_HIDDEN, "seen.json")
CONFLUENCE_PATH = os.path.join(GOAL_HIDDEN, "confluence.json")


def _send_telegram(text: str) -> bool:
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


def _load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _confluence_hot(
    limit: int = 10, fresh_minutes: float = 60.0, now: float | None = None
) -> list[str]:
    """Fresh caller-confluence addresses: 2+ independent channels called the
    same contract within the window. Leading indicator, no history needed."""
    now = time.time() if now is None else now
    data = _load_json(CONFLUENCE_PATH, {})
    out = []
    for addr, info in data.items():
        if not isinstance(info, dict):
            continue
        if now - float(info.get("ts", 0)) > fresh_minutes * 60:
            continue
        if len(set(info.get("sources") or [])) >= 2:
            out.append(addr)
    return out[:limit]


async def amain() -> int:
    ap = argparse.ArgumentParser(description="FENRIR hot watch (2m fast lane)")
    ap.add_argument("--seen", default=SEEN_PATH)
    ap.add_argument("--min-score", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=25, help="max hot tokens to re-check per run")
    ap.add_argument(
        "--dry-run", action="store_true", help="evaluate but don't send Telegram alerts"
    )
    args = ap.parse_args()

    accel = AccelTracker(AccelTracker.default_state_path())
    hot = accel.hot_candidates(limit=args.limit)
    for addr in _confluence_hot():
        if addr not in hot and len(hot) < args.limit:
            hot.append(addr)

    summary: dict = {"hot_list": len(hot), "scanned": 0, "alerted": [], "send_failures": 0}
    if not hot:
        print(json.dumps(summary))
        return 0

    ds = DexScreenerProvider(timeout_seconds=15)
    gt = GeckoTerminalProvider(timeout_seconds=15)
    gp = GoPlusProvider(timeout_seconds=10)
    local_safety = RobinhoodSafetyProvider()
    engine = FilterEngine()
    scorer = ScoringEngine()
    tagger = PlaybookTagger()
    seen = load_seen(args.seen)
    now = time.time()
    new_cands: list[dict] = []
    try:
        for addr in hot:
            try:
                cand = await evaluate_address(
                    "hot_watch",
                    addr,
                    None,
                    ds,
                    gp,
                    engine,
                    scorer,
                    tagger,
                    args.min_score,
                    accel,
                    local_safety=local_safety,
                )
            except Exception:
                cand = None
            summary["scanned"] += 1
            if cand is None:
                await asyncio.sleep(0.4)
                continue
            if not should_alert(seen, addr, now=now):
                await asyncio.sleep(0.4)
                continue  # already flagged within 24h — the scout owns it
            text = format_scout_alert(cand)
            sent = True
            if not args.dry_run:
                sent = _send_telegram(text)
            if sent:
                record_alert(
                    seen,
                    addr,
                    symbol=cand.get("symbol"),
                    score=(cand.get("score") or {}).get("overall"),
                    now=now,
                )
                new_cands.append(cand)
                summary["alerted"].append(cand.get("symbol"))
            else:
                summary["send_failures"] += 1
            await asyncio.sleep(0.4)
    finally:
        await ds.close()
        await gt.close()
        await gp.close()
        accel.save()

    save_seen(args.seen, seen)
    if new_cands:
        _save_json("/tmp/hot_new_candidates.json", new_cands)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
