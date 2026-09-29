#!/usr/bin/env python3
"""FENRIR - Robinhood on-chain pair watch (2-minute fast lane).

Polls Uniswap v4 ``Initialize`` events on Robinhood Chain and records every
new base-paired pool in the first-seen registry
(``~/workspace/goals/token-scout-watch/hidden_files/rh_pairs.json``).

This script only maintains the registry — it does NOT evaluate or alert.
The 10-minute scout consumes ``fresh_addresses()`` and runs new tokens
through the normal filter/scoring pipeline once DexScreener has indexed
them. Run every 2 minutes via the ``rh-pair-watch`` cron (disabled until the
machine/scout migration is complete).

Exit 0 always (fail-open); prints a one-line summary for cron logs.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from fenrir.discovery.providers.rh_onchain import RobinhoodPairMonitor


async def main() -> int:
    monitor = RobinhoodPairMonitor()
    try:
        result = await monitor.sync()
    finally:
        await monitor.close()
    new = result.get("new", [])
    latest = result.get("latest_block")
    if result.get("error"):
        print(f"rh_pair_watch: RPC unreachable (cursor kept)")
        return 0
    if result.get("up_to_date"):
        print(f"rh_pair_watch: up to date at block {latest}")
        return 0
    print(
        f"rh_pair_watch: block {latest} "
        f"({result.get('scanned_blocks', 0)} scanned), "
        f"{len(new)} new pool(s)"
    )
    for addr in new[:20]:
        print(f"  new: {addr}")
    if len(new) > 20:
        print(f"  ... and {len(new) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
