#!/usr/bin/env python3
"""
FENRIR - Cross-Venue Arbitrage Divergence Detector (Phase 4, read-only)

First slice of the arbitrage strategy. The same token can trade at different prices
on different venues — most relevantly a pump.fun bonding curve versus a graduated
Raydium / Jupiter pool, or two AMM pools. When the gap is wide enough to clear the
round-trip costs, buying the cheap venue and selling the expensive one is a
market-neutral profit.

This module is the DETECTOR only — it is read-only and never executes. Given per-venue
quotes for a token it computes the *net-of-cost* edge honestly:

    net_bps = gross_spread_bps
              - (buy_fee + buy_slippage)      # cheap leg
              - (sell_fee + sell_slippage)     # expensive leg
              - tx_cost_bps                    # on-chain cost amortized over size

The tx cost and slippage terms are what make a naive "prices differ, free money"
reading wrong: on-chain fees, AMM price impact, and the venue swap fees routinely
exceed the raw gap. Only an opportunity whose net edge clears the threshold (and is
positive) is ``actionable``. Atomic multi-leg execution via the Jito bundle path is a
later, gated PR; nothing here signs a transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

_BPS = 10_000.0


@dataclass
class VenueQuote:
    """A token's price + depth on one venue.

    ``price`` is SOL-per-token. ``liquidity_sol`` is the pool depth used to estimate
    price impact (0 = unknown → penalized as maximally illiquid). ``fee_bps`` is the
    venue's swap fee (pump.fun curve ≈ 100 bps, Raydium ≈ 25 bps).
    """

    venue: str
    price: float
    liquidity_sol: float = 0.0
    fee_bps: int = 100


@dataclass
class ArbConfig:
    """Thresholds + cost model for the divergence detector."""

    # Minimum net edge (after all costs) to call an opportunity actionable.
    min_net_edge_bps: float = 50.0
    # Cap on the modeled per-leg AMM slippage (bps).
    max_slippage_bps: float = 500.0
    # Round-trip on-chain cost in SOL (both swap legs + priority fee / Jito tip),
    # amortized over the trade size into bps.
    tx_cost_sol: float = 0.002
    # Venues thinner than this are ignored (can't source/exit size cleanly).
    min_liquidity_sol: float = 0.0


@dataclass
class ArbOpportunity:
    """A costed cross-venue divergence for a given trade size."""

    token_address: str
    size_sol: float
    buy_venue: str
    sell_venue: str
    buy_price: float
    sell_price: float
    gross_edge_bps: float
    buy_cost_bps: float  # buy-leg fee + slippage
    sell_cost_bps: float  # sell-leg fee + slippage
    tx_cost_bps: float
    net_edge_bps: float
    est_profit_sol: float
    min_net_edge_bps: float

    @property
    def actionable(self) -> bool:
        """Whether the net edge clears the threshold and is genuinely profitable."""
        return self.net_edge_bps >= self.min_net_edge_bps and self.est_profit_sol > 0

    @property
    def total_cost_bps(self) -> float:
        return self.buy_cost_bps + self.sell_cost_bps + self.tx_cost_bps


class ArbitrageDetector:
    """Read-only cross-venue divergence detector.

    ``evaluate`` returns the full costed breakdown for the widest pair (or None when
    there is nothing to arbitrage); ``detect`` returns it only when it is actionable.
    No execution, no network — quotes are supplied by the caller.
    """

    def __init__(self, config: ArbConfig | None = None) -> None:
        self.config = config or ArbConfig()

    def slippage_bps(self, size_sol: float, liquidity_sol: float) -> float:
        """Modeled AMM price impact for trading ``size_sol`` against ``liquidity_sol``.

        A linear proxy for constant-product impact (fraction of the pool consumed),
        capped. Unknown/zero liquidity is treated as maximally illiquid.
        """
        if liquidity_sol <= 0:
            return self.config.max_slippage_bps
        return min(self.config.max_slippage_bps, (size_sol / liquidity_sol) * _BPS)

    def evaluate(
        self, token_address: str, quotes: list[VenueQuote], size_sol: float
    ) -> ArbOpportunity | None:
        """Cost the widest usable venue pair. Returns None when fewer than two venues
        are usable or there is no positive price divergence between distinct venues."""
        if size_sol <= 0:
            raise ValueError("size_sol must be positive")

        usable = [
            q for q in quotes if q.price > 0 and q.liquidity_sol >= self.config.min_liquidity_sol
        ]
        if len(usable) < 2:
            return None

        buy = min(usable, key=lambda q: q.price)  # buy where it is cheapest
        sell = max(usable, key=lambda q: q.price)  # sell where it is dearest
        if buy.venue == sell.venue or sell.price <= buy.price:
            return None  # no divergence to capture

        gross_edge_bps = (sell.price - buy.price) / buy.price * _BPS
        buy_cost_bps = buy.fee_bps + self.slippage_bps(size_sol, buy.liquidity_sol)
        sell_cost_bps = sell.fee_bps + self.slippage_bps(size_sol, sell.liquidity_sol)
        tx_cost_bps = (self.config.tx_cost_sol / size_sol) * _BPS

        net_edge_bps = gross_edge_bps - buy_cost_bps - sell_cost_bps - tx_cost_bps
        est_profit_sol = size_sol * net_edge_bps / _BPS

        return ArbOpportunity(
            token_address=token_address,
            size_sol=size_sol,
            buy_venue=buy.venue,
            sell_venue=sell.venue,
            buy_price=buy.price,
            sell_price=sell.price,
            gross_edge_bps=gross_edge_bps,
            buy_cost_bps=buy_cost_bps,
            sell_cost_bps=sell_cost_bps,
            tx_cost_bps=tx_cost_bps,
            net_edge_bps=net_edge_bps,
            est_profit_sol=est_profit_sol,
            min_net_edge_bps=self.config.min_net_edge_bps,
        )

    def detect(
        self, token_address: str, quotes: list[VenueQuote], size_sol: float
    ) -> ArbOpportunity | None:
        """Return the opportunity only when it is actionable (net edge clears the
        threshold and is profitable); otherwise None."""
        opp = self.evaluate(token_address, quotes, size_sol)
        if opp is None or not opp.actionable:
            return None
        return opp
