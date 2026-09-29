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
)


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
    print(f"  mint disabled: {s.mint_disabled} | blacklist: {s.blacklist_present} | "
          f"ownership renounced: {s.ownership_renounced}")
    print(f"  LP locked/burned: {s.lp_locked_or_burned}"
          + (f" ({s.lp_locked_pct:.0f}%)" if s.lp_locked_pct is not None else ""))
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
        report = await p.investigate(args.chain_id, args.address,
                                     timeout_seconds=args.wait)
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


async def cmd_sweep(_args) -> int:
    p = PerceptorProvider()
    try:
        cache = p._load_cache()
        pending = [a for a, e in cache.items() if e.get("status") == "pending"]
        if not pending:
            print(json.dumps({"checked": 0, "completed": []}))
            return 0
        completed = []
        for addr in pending:
            report = await p.refresh_report(addr)
            if report is not None:
                completed.append({
                    "address": addr,
                    "band": report.band,
                    "band_label": report.band_label,
                    "headline": report.headline,
                    "investigation_id": report.investigation_id,
                    "honeypot": report.safety.honeypot,
                    "buy_tax_pct": report.safety.buy_tax_pct,
                    "sell_tax_pct": report.safety.sell_tax_pct,
                    "mint_disabled": report.safety.mint_disabled,
                    "lp_locked_or_burned": report.safety.lp_locked_or_burned,
                })
        print(json.dumps({"checked": len(pending), "completed": completed}, indent=1))
        return 0
    finally:
        await p.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Perceptor on-chain forensics CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("investigate", help="start (or reuse) a scan")
    inv.add_argument("address")
    inv.add_argument("--chain-id", type=int, default=ROBINHOOD_CHAIN_ID)
    inv.add_argument("--wait", type=float, default=300.0,
                     help="seconds to wait for the verdict (0 = just start)")
    inv.set_defaults(func=cmd_investigate)

    st = sub.add_parser("status", help="cached verdict or pending id")
    st.add_argument("address")
    st.set_defaults(func=cmd_status)

    sw = sub.add_parser("sweep", help="refresh all pending scans")
    sw.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
