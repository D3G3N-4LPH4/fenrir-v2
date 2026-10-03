"""Entry-tier classification for scout candidates.

Gate-tracker review (2026-10-03, 42 alerts since 10-01) showed the scout's
failure mode is not finding moves — 36% of alerts peaked >=+50% after
clearance — but *timing*: volatility_breakout alerts fire 10-25 minutes late
into an already-vertical 1h candle (Startup +57% 1h at alert, dead -98% 40
minutes later; Stryker, swordinu, same shape). Median vb move since
clearance: -98%.

The tier splits entries so the alert path can treat them differently:
- ``ignition``: pre-momentum entries (curve_ignition / graduation_watch).
  The vertical move hasn't happened yet — these keep alerting.
- ``standard``: normal entries — alert as before.
- ``late``: volatility_breakout where the move is already mostly done
  (1h change deep into the filter band, or the bonding curve already
  graduated). Buying here is buying the exit. These are logged to the gate
  tracker for measurement but do NOT fire Telegram alerts.

The classification is deliberately conservative: when the lateness signals
are missing (unknown 1h move, no curve data), the tier stays ``standard``.
"""

from __future__ import annotations

from typing import Any

TIER_IGNITION = "ignition"
TIER_STANDARD = "standard"
TIER_LATE = "late"

# volatility_breakout allows +40%..+120% 1h. Past +80% the entry is the exit:
# two-thirds of the allowed band is already printed on the candle.
LATE_MOVE_1H_PCT = 80.0
# A graduated bonding curve (>=85% progress) means the vertical move already
# played out on the curve — the DexScreener momentum is the distribution leg.
LATE_BOND_PROGRESS_PCT = 85.0

_IGNITION_FILTERS = frozenset({"curve_ignition", "graduation_watch"})


def _fnum(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def classify_entry_tier(cand: dict) -> str:
    """Return ``ignition`` | ``standard`` | ``late`` for a candidate dict."""
    filters = set(cand.get("passed_filters") or [])
    if filters & _IGNITION_FILTERS:
        return TIER_IGNITION
    if "volatility_breakout" in filters:
        move_1h = _fnum(cand.get("price_change_1h_pct"))
        bond = _fnum(cand.get("bond_progress_pct"))
        if (move_1h is not None and move_1h >= LATE_MOVE_1H_PCT) or (
            bond is not None and bond >= LATE_BOND_PROGRESS_PCT
        ):
            return TIER_LATE
    return TIER_STANDARD


def tier_alerts(tier: str) -> bool:
    """Whether this tier fires a Telegram alert. Late entries are logged
    to the gate tracker for measurement but never alerted."""
    return tier != TIER_LATE
