#!/usr/bin/env python3
"""FENRIR - Perceptor CLI: on-chain forensics for Robinhood Chain tokens.

Usage:
  tools/perceptor.py investigate <address> [--chain-id 4663] [--wait 300]
  tools/perceptor.py status <address>
  tools/perceptor.py sweep            # refresh all pending scans, print verdicts

``investigate`` starts a scan (one POST per address, ever — cached) and, with
``--wait``, blocks until the verdict lands. ``status`` prints the cached
verdict or the pending investigation id. ``sweep`` is for cron follow-ups:
it re-checks every pending scan and prints the ones that completed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.discovery.providers.perceptor import (  # noqa: E402
    ROBINHOOD_CHAIN_ID,
    PerceptorProvider,
    PerceptorReport,
)
from fenrir.discovery.alerts import format_perceptor_verdict  # noqa: E402

# Per-entry deadline for `sweep` refreshes (the provider's own request
# timeout is 15s; this is the backstop for a wedged coroutine).
SWEEP_ENTRY_TIMEOUT_SECONDS = 30.0


def print_report(report, address: str) -> None:
    s = report.safety
    print(f"{address}")
    print(f"  verdict: {report.band_label or report.band} — {report.headline or ''}")
    for label, value in report.checks:
        print(f"  {label}: {value}")
    for label, tone in report.signals:
        if str(tone).lower() in ("medium", "high"):
            print(f"  ! {label}")
    print(f"  honeypot: {s.honeypot} | buy tax: {s.buy_tax_pct}% | sell tax: {s.sell_tax_pct}%")
    print(
        f"  mint disabled: {s.mint_disabled} | blacklist: {s.blacklist_present} | "
        f"ownership renounced: {s.ownership_renounced}"
    )
    print(
        f"  LP locked/burned: {s.lp_locked_or_burned}"
        + (f" ({s.lp_locked_pct:.0f}%)" if s.lp_locked_pct is not None else "")
    )
    if s.risk_flags:
        print(f"  risk flags: {'; '.join(s.risk_flags[:6])}")
    print(f"  https://www.perceptor.info/?investigation={report.investigation_id}")


async def cmd_investigate(args) -> int:
    p = PerceptorProvider()
    try:
        inv_id = await p.ensure_investigation(args.chain_id, args.address)
        if not inv_id:
            print("Failed to start investigation.")
            return 1
        if args.wait <= 0:
            print(f"Investigation started: {inv_id}")
            print("Re-run with --wait or `status` to fetch the verdict.")
            return 0
        print(f"Scan running ({inv_id}) — waiting up to {args.wait:.0f}s…", flush=True)
        report = await p.investigate(args.chain_id, args.address, timeout_seconds=args.wait)
        if report is None:
            print("Scan did not complete in time. Re-run `status` later.")
            return 2
        print_report(report, args.address)
        return 0
    finally:
        await p.close()


async def cmd_status(args) -> int:
    p = PerceptorProvider()
    try:
        report = await p.refresh_report(args.address)
        if report is not None:
            print_report(report, args.address)
            return 0
        entry = p._load_cache().get(args.address.lower(), {})
        inv_id = entry.get("investigation_id")
        if inv_id:
            print(f"Scan still pending: {inv_id}")
            return 2
        print("No investigation on record — run `investigate` first.")
        return 1
    finally:
        await p.close()


async def cmd_sweep(args) -> int:
    p = PerceptorProvider()
    try:
        cache = p._load_cache()
        # Revisit pending scans AND completed scans that were never notified:
        # when --notify is skipped (alerts paused), a completed entry would
        # otherwise never be picked up again once its status flips to
        # "complete".
        pending = [
            a
            for a, e in cache.items()
            if e.get("status") == "pending"
            or (e.get("status") == "complete" and not e.get("followup_sent"))
        ]
        if not pending:
            print(json.dumps({"checked": 0, "completed": [], "notified": []}))
            return 0
        # Bounded concurrency + per-entry deadline: a wedged refresh must not
        # stall the sweep (it runs inside the 10-minute scout cron).
        sem = asyncio.Semaphore(4)

        async def _refresh(addr: str) -> tuple[str, PerceptorReport | None]:
            async with sem:
                try:
                    report = await asyncio.wait_for(
                        p.refresh_report(addr), timeout=SWEEP_ENTRY_TIMEOUT_SECONDS
                    )
                except Exception:  # noqa: BLE001 - fail-open
                    report = None
                return addr, report

        refreshed = await asyncio.gather(*(_refresh(a) for a in pending))
        completed = []
        notified = []
        for addr, report in refreshed:
            if report is None:
                continue
            entry = cache[addr]
            completed.append(
                {
                    "address": addr,
                    "symbol": (entry.get("context") or {}).get("symbol"),
                    "band": report.band,
                    "band_label": report.band_label,
                    "headline": report.headline,
                    "investigation_id": report.investigation_id,
                    "honeypot": report.safety.honeypot,
                    "buy_tax_pct": report.safety.buy_tax_pct,
                    "sell_tax_pct": report.safety.sell_tax_pct,
                    "mint_disabled": report.safety.mint_disabled,
                    "lp_locked_or_burned": report.safety.lp_locked_or_burned,
                }
            )
            if args.notify and not entry.get("followup_sent"):
                msg = format_perceptor_verdict(addr, entry.get("context"), report)
                if _send_telegram(msg):
                    entry["followup_sent"] = True
                    p._save_cache()
                    notified.append(addr)
                # else: leave followup_sent unset so the next sweep retries
        print(
            json.dumps(
                {"checked": len(pending), "completed": completed, "notified": notified},
                indent=1,
            )
        )
        return 0
    finally:
        await p.close()


def _send_telegram(text: str) -> bool:
    """Deliver one message via tools/telegram_notify.py (reads .env itself)."""
    import subprocess

    notify = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_notify.py")
    try:
        r = subprocess.run(
            [sys.executable, notify, "--parse-mode", "Markdown", text],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if r.returncode != 0:
            print(f"telegram follow-up failed: {r.stderr.strip()}", file=sys.stderr)
        return r.returncode == 0
    except Exception as e:  # noqa: BLE001 - fail-open
        print(f"telegram follow-up failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Perceptor on-chain forensics CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("investigate", help="start (or reuse) a scan")
    inv.add_argument("address")
    inv.add_argument("--chain-id", type=int, default=ROBINHOOD_CHAIN_ID)
    inv.add_argument(
        "--wait", type=float, default=300.0, help="seconds to wait for the verdict (0 = just start)"
    )
    inv.set_defaults(func=cmd_investigate)

    st = sub.add_parser("status", help="cached verdict or pending id")
    st.add_argument("address")
    st.set_defaults(func=cmd_status)

    sw = sub.add_parser("sweep", help="refresh all pending scans")
    sw.add_argument(
        "--notify",
        action="store_true",
        help="send a Telegram follow-up for each newly completed verdict",
    )
    sw.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    code: int = asyncio.run(args.func(args))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
