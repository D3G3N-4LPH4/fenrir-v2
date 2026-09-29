#!/usr/bin/env python3
"""
FENRIR discovery: poll-over-poll acceleration tracking.

Static snapshot gates miss the *start* of a move: a coin going 0.9 → 1.1 → 1.4
buy/sell across three polls is screaming, but any single snapshot just sees
"1.4". This module keeps a small per-token history of each scout poll and
computes growth factors the entry filters can gate on:

  - ``accel_txn_growth``: 1h transaction count vs the previous poll
    (activity accelerating into the token)
  - ``accel_holder_growth``: holder count vs the previous poll
    (new wallets arriving — the pre-run signature)
  - ``accel_edge_delta``: 1h buy-pressure delta vs the previous poll
    (buy edge strengthening, not just present)

The history persists as JSON next to the other scout state so acceleration
survives restarts. Entries older than ``max_age_hours`` are pruned; each
address keeps at most ``max_history`` observations.

The scout attaches the growth metrics to every evaluated TokenSnapshot
*before* the FilterEngine runs, so filters with acceleration thresholds
(e.g. ``momentum_transition``) fail closed when there is no prior poll —
the filter only fires from the second sighting on, which is exactly the
pre-run window it is designed for.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("FENRIR.Accel")


def _growth(cur: float | None, prev: float | None) -> float | None:
    """Multiplicative growth cur/prev. None when either side is unknown."""
    if cur is None or prev is None:
        return None
    if prev <= 0:
        # No prior activity: any current activity is "infinite" growth, but an
        # unbounded value breaks thresholds — cap it high and let the raw
        # minimums (txns, holders) do the sanity work.
        return 999.0 if cur > 0 else 1.0
    return cur / prev


class AccelTracker:
    """Per-token poll-over-poll history with growth-factor computation."""

    def __init__(
        self,
        state_path: str | Path,
        max_history: int = 6,
        max_age_hours: float = 24.0,
    ) -> None:
        self.state_path = Path(os.path.expanduser(str(state_path)))
        self.max_history = max_history
        self.max_age_seconds = max_age_hours * 3600.0
        # address -> list of observations (oldest first)
        self._hist: dict[str, list[dict[str, Any]]] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return
        if isinstance(raw, dict):
            now = time.time()
            for addr, obs in raw.items():
                if isinstance(obs, list):
                    fresh = [o for o in obs if now - float(o.get("ts", 0)) < self.max_age_seconds]
                    if fresh:
                        self._hist[addr] = fresh[-self.max_history :]

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._hist))
            tmp.replace(self.state_path)
        except OSError as exc:
            logger.warning(f"accel state save failed: {exc}")

    # ── recording ────────────────────────────────────────────────────

    def record(self, snap: Any, now: float | None = None) -> dict[str, float | None] | None:
        """Append this poll's observation for the snapshot's token.

        Attaches ``accel_txn_growth`` / ``accel_holder_growth`` /
        ``accel_edge_delta`` / ``accel_polls_seen`` to the snapshot and
        returns the growth dict, or None when this is the first sighting
        (no previous poll to compare against).
        """
        now = time.time() if now is None else now
        addr = getattr(snap, "token_address", "") or ""
        if not addr:
            return None

        txns_1h = int(getattr(snap, "txns_1h_buys", 0) or 0) + int(
            getattr(snap, "txns_1h_sells", 0) or 0
        )
        holders = getattr(snap, "holder_count", None)
        edge = getattr(snap, "buy_pressure_1h", None)
        try:
            edge = float(edge) if edge is not None else None
        except (TypeError, ValueError):
            edge = None

        obs = {
            "ts": now,
            "txns_1h": txns_1h,
            "holders": int(holders) if holders is not None else None,
            "edge": edge,
        }
        hist = self._hist.setdefault(addr, [])
        hist.append(obs)
        del hist[: -self.max_history]

        growth: dict[str, float | None] | None = None
        if len(hist) >= 2:
            prev = hist[-2]
            prev_txns = prev.get("txns_1h")
            prev_holders = prev.get("holders")
            prev_edge = prev.get("edge")
            # prev txns of 0 with current activity > 0 means the tape just woke
            # up — _growth caps that at 999x ("infinite", bounded).
            growth = {
                "accel_txn_growth": _growth(float(txns_1h), float(prev_txns) if prev_txns else 0.0),
                "accel_holder_growth": _growth(
                    float(holders) if holders is not None else None,
                    float(prev_holders) if prev_holders is not None else None,
                ),
                "accel_edge_delta": (
                    (edge - prev_edge) if edge is not None and prev_edge is not None else None
                ),
            }

        snap.accel_txn_growth = growth["accel_txn_growth"] if growth else None
        snap.accel_holder_growth = growth["accel_holder_growth"] if growth else None
        snap.accel_edge_delta = growth["accel_edge_delta"] if growth else None
        snap.accel_polls_seen = len(hist)
        return growth

    # ── introspection ────────────────────────────────────────────────

    def polls_seen(self, address: str) -> int:
        return len(self._hist.get(address, []))

    def hot_candidates(
        self,
        within_minutes: float = 30.0,
        min_txn_growth: float = 1.25,
        min_holder_growth: float = 1.3,
        min_edge_delta: float = 0.05,
        limit: int = 25,
        now: float | None = None,
    ) -> list[str]:
        """Addresses showing a pulse since the last poll — the fast-lane list.

        An address qualifies when its latest observation is fresh AND any of
        the acceleration dimensions is already moving: txn growth, holder
        growth, or buy-edge delta vs the previous poll. Growth is recomputed
        from the last two observations so this works without a live snapshot.
        Most-recent first, capped at ``limit``.
        """
        now = time.time() if now is None else now
        window = within_minutes * 60.0
        hot: list[tuple[float, str]] = []
        for addr, obs in self._hist.items():
            if len(obs) < 2:
                continue
            last, prev = obs[-1], obs[-2]
            if now - float(last.get("ts", 0)) > window:
                continue
            txn_g = _growth(float(last.get("txns_1h") or 0), float(prev.get("txns_1h") or 0))
            hold_g = _growth(
                float(last["holders"]) if last.get("holders") is not None else None,
                float(prev["holders"]) if prev.get("holders") is not None else None,
            )
            edge_d = (
                (last["edge"] - prev["edge"])
                if last.get("edge") is not None and prev.get("edge") is not None
                else None
            )
            if (
                (txn_g is not None and txn_g >= min_txn_growth)
                or (hold_g is not None and hold_g >= min_holder_growth)
                or (edge_d is not None and edge_d >= min_edge_delta)
            ):
                hot.append((float(last["ts"]), addr))
        hot.sort(reverse=True)
        return [addr for _, addr in hot[:limit]]

    def prune(self, now: float | None = None) -> int:
        """Drop addresses with no observation inside the age window. Returns count dropped."""
        now = time.time() if now is None else now
        stale = [
            addr
            for addr, obs in self._hist.items()
            if not obs or now - float(obs[-1].get("ts", 0)) >= self.max_age_seconds
        ]
        for addr in stale:
            del self._hist[addr]
        return len(stale)

    @staticmethod
    def default_state_path() -> Path:
        return Path(
            os.path.expanduser(
                "~/workspace/goals/token-scout-watch/hidden_files/accel_history.json"
            )
        )
