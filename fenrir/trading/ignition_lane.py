"""Rules-only block-zero lane.

Ignition is a slot race. A Claude call here is the whole trade. This gate
runs in the create callback and either returns a fixed buy or skips.
The model stays on migration, reversal, and the Jupiter scanner.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IgnitionDecision:
    buy: bool
    amount_sol: float
    reason: str


def evaluate_ignition(
    token_data: dict,
    *,
    min_liquidity_sol: float = 0.5,
    max_market_cap_sol: float = 30.0,
    amount_sol: float = 0.05,
    blocked_creators: set[str] | None = None,
) -> IgnitionDecision:
    """Hard filters only. No confidence, no ensemble, no size override."""
    symbol = token_data.get("symbol") or "???"
    creator = token_data.get("creator") or ""
    if blocked_creators and creator in blocked_creators:
        return IgnitionDecision(False, 0.0, f"creator blocked {creator[:8]}")

    liq = float(token_data.get("initial_liquidity_sol") or 0.0)
    if liq < min_liquidity_sol:
        return IgnitionDecision(False, 0.0, f"{symbol} liquidity {liq:.3f} < {min_liquidity_sol}")

    mcap = float(token_data.get("market_cap_sol") or 0.0)
    if mcap > max_market_cap_sol:
        return IgnitionDecision(False, 0.0, f"{symbol} mcap {mcap:.2f} > {max_market_cap_sol}")

    curve = token_data.get("bonding_curve_state")
    if curve is not None and getattr(curve, "complete", False):
        return IgnitionDecision(False, 0.0, f"{symbol} already migrated")

    return IgnitionDecision(True, amount_sol, f"{symbol} rules pass")
