#!/usr/bin/env python3
"""FENRIR scout pipeline — one 10-minute tick for machine-side runs.

Replicates the sandbox cron: scout -> dedup (seen.json, 24h, lowercase) ->
Telegram alerts -> gate-tracker record.

Usage:
    python tools/scout_tick.py [--state-dir ~/.fenrir-scout] [--min-score 60]

State files (created as needed):
    <state-dir>/seen.json          address -> {symbol, first_seen, last_alerted, score}
    <state-dir>/gate_tracker/tracked.json   gate-clearance records

Exit 0 on success (even with zero candidates). Non-zero on total failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

REALERT_SECONDS = 86400


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, **kw)  # noqa: S603


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=os.path.expanduser("~/.fenrir-scout"))
    ap.add_argument("--min-score", type=float, default=60.0)
    ap.add_argument("--chains", nargs="+", default=["solana", "robinhood"])
    args = ap.parse_args()

    os.makedirs(args.state_dir, exist_ok=True)
    gt_dir = os.path.join(args.state_dir, "gate_tracker")
    os.makedirs(gt_dir, exist_ok=True)
    seen_path = os.path.join(args.state_dir, "seen.json")
    # Reuse the interpreter running this script (the venv python under systemd
    # or an activated venv) — portable across Linux and Windows, no hardcoded
    # .venv/bin vs .venv/Scripts path.
    py = sys.executable

    # 1. discovery
    r = run(
        [
            py,
            "tools/scout.py",
            "--chains",
            *args.chains,
            "--limit",
            "25",
            "--min-score",
            str(args.min_score),
        ],
        cwd=REPO_ROOT,
    )
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
        candidates = out.get("candidates", [])
    except Exception as e:  # noqa: BLE001
        print(f"scout_tick: scout output unparseable: {e}", file=sys.stderr)
        print(f"scout_tick: stderr tail: {r.stderr[-500:]}", file=sys.stderr)
        return 1
    print(
        f"scout_tick: scanned={out.get('scanned')} "
        f"by_source={out.get('by_source')} candidates={len(candidates)}",
        file=sys.stderr,
    )

    # 2. dedup vs seen.json (lowercase keys, 24h re-alert)
    try:
        seen = json.load(open(seen_path)) if os.path.exists(seen_path) else {}
    except Exception:  # noqa: BLE001
        seen = {}
    seen = {str(k).lower(): v for k, v in seen.items()}
    now = time.time()
    new_cands = []
    for c in candidates:
        addr = str(c.get("address", ""))
        key = addr.lower()
        entry = seen.get(key)
        if entry is None or (now - float(entry.get("last_alerted", 0))) > REALERT_SECONDS:
            new_cands.append(c)
            seen[key] = {
                "symbol": c.get("symbol"),
                "first_seen": entry.get("first_seen", now) if entry else now,
                "last_alerted": now,
                "score": (c.get("score") or {}).get("overall"),
            }
        else:
            # refresh score/symbol on repeat sightings
            entry["score"] = (c.get("score") or {}).get("overall")
            entry["symbol"] = c.get("symbol")
    json.dump(seen, open(seen_path, "w"), indent=1)

    # 3. telegram alerts for NEW candidates
    # Entry tiers (0052): late-tier candidates (volatility_breakout where the
    # move is already done) are logged to the gate tracker but NEVER alerted.
    from fenrir.discovery.alerts import format_scout_alert
    from fenrir.discovery.entry_tier import tier_alerts

    alertable = [c for c in new_cands if tier_alerts(c.get("entry_tier", "standard"))]
    n_late = len(new_cands) - len(alertable)
    if n_late:
        print(
            f"scout_tick: {n_late} late-tier suppressed from Telegram (still recorded)",
            file=sys.stderr,
        )

    notify = os.path.join(REPO_ROOT, "tools", "telegram_notify.py")
    for c in alertable:
        text = format_scout_alert(c)
        r2 = run([py, notify, "--parse-mode", "Markdown", text], cwd=REPO_ROOT)
        if r2.returncode != 0:
            print(
                f"scout_tick: telegram send failed for {c.get('symbol')}: {r2.stderr[-300:]}",
                file=sys.stderr,
            )

    # 4. gate-tracker record (full list, including late tier — measurement stays honest)
    if new_cands:
        nc_path = os.path.join(args.state_dir, "new_candidates.json")
        json.dump(new_cands, open(nc_path, "w"))
        r3 = run(
            [
                py,
                "tools/gate_tracker.py",
                "--state",
                os.path.join(gt_dir, "tracked.json"),
                "record",
                "--candidates",
                nc_path,
            ],
            cwd=REPO_ROOT,
        )
        if r3.returncode != 0:
            print(f"scout_tick: gate_tracker record failed: {r3.stderr[-300:]}", file=sys.stderr)

    print(f"scout_tick: done. alerts={len(alertable)} late_suppressed={n_late}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
