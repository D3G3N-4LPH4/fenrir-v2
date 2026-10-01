"""Alert dedup store shared by the scout and the hot watch.

Address keys are normalized to lowercase on both read and write so a
checksummed EVM address and its lowercase form can never create two
entries and slip past the 24h re-alert window (bug found 2026-09-30:
ROBINPEPE re-alerted inside 24h because one writer used a checksummed
key while the lookup used a lowercase one).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

REALERT_SECONDS = 86400  # 24h re-alert window, same contract as the scout


def normalize(address: str) -> str:
    """Canonical key for an address: lowercase, whitespace stripped."""
    return address.strip().lower()


def load(path: str) -> dict[str, dict[str, Any]]:
    """Load the seen store, merging any legacy mixed-case keys (keeps freshest)."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    merged: dict[str, dict[str, Any]] = {}
    for key, entry in raw.items():
        nkey = normalize(key)
        prev = merged.get(nkey)
        if prev is None:
            merged[nkey] = entry
            continue
        old_ts = float(prev.get("last_alerted", 0))
        new_ts = float(entry.get("last_alerted", 0))
        if new_ts > old_ts:
            merged[nkey] = entry
    return merged


def save(path: str, seen: dict[str, dict[str, Any]]) -> None:
    """Atomic write of the seen store."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(seen, f)
    os.replace(tmp, path)


def should_alert(seen: dict[str, dict[str, Any]], address: str, now: float | None = None) -> bool:
    """True when the address was never alerted or the re-alert window expired."""
    entry = seen.get(normalize(address), {})
    last_alerted = float(entry.get("last_alerted", 0))
    ts = time.time() if now is None else now
    return ts - last_alerted > REALERT_SECONDS


def record_alert(
    seen: dict[str, dict[str, Any]],
    address: str,
    *,
    symbol: str | None,
    score: float | None,
    now: float | None = None,
) -> None:
    """Mark an address as alerted, refreshing score/symbol and last_alerted."""
    ts = time.time() if now is None else now
    key = normalize(address)
    entry = seen.get(key, {})
    seen[key] = {
        "symbol": symbol,
        "first_seen": entry.get("first_seen", ts),
        "last_alerted": ts,
        "score": score,
    }
