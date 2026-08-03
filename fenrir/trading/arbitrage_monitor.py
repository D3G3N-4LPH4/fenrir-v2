#!/usr/bin/env python3
"""
FENRIR - Arbitrage Monitor: real per-venue quotes (Phase 4.4b, read-only)

Wires the pure divergence detector (``fenrir.trading.arbitrage``) to REAL price
sources and surfaces actionable opportunities as events — without executing. This is
the read-only step of the arbitrage live path: prove we can see genuine, cost-cleared
cross-venue dislocations on real data before building the atomic executor.

Both legs must be priced in the same unit (SOL per token):
  - the pump.fun bonding curve exposes ``get_price()`` (SOL/token) and its SOL depth
    via ``real_sol_reserves``;
  - a graduated AMM pool prices in SOL via the aggregated price feed, with its SOL
    liquidity from the DexScreener snapshot (``liquidity_sol``).

The monitor takes injected async quote sources (so it is fully testable with no
network) and, in production, is handed closures that read the curve + feed. It never
signs a transaction; ``ArbitrageDetector.detect`` already refuses non-actionable
gaps, and atomic Jito-bundle execution is a later, gated PR.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fenrir.events.types import arbitrage_opportunity_event
from fenrir.trading.arbitrage import ArbitrageDetector, ArbOpportunity, VenueQuote

# An async source that yields one venue's quote for the token (or None if unavailable).
QuoteSource = Callable[[], Awaitable[VenueQuote | None]]

_LAMPORTS_PER_SOL = 1_000_000_000


def venue_quote_from_curve(
    curve_state: Any,
    fee_bps: int = 100,
    venue: str = "pumpfun_curve",
) -> VenueQuote | None:
    """Build a VenueQuote from a decoded bonding-curve state.

    Price is SOL/token from the curve; the SOL depth proxy is ``real_sol_reserves``
    (actual SOL committed to the curve). Returns None if the curve is unpriceable.
    """
    price = curve_state.get_price()
    if price <= 0:
        return None
    real_sol = getattr(curve_state, "real_sol_reserves", 0) or 0
    liquidity_sol = real_sol / _LAMPORTS_PER_SOL
    return VenueQuote(venue=venue, price=price, liquidity_sol=liquidity_sol, fee_bps=fee_bps)


def venue_quote_from_amm(
    price_sol: float | None,
    liquidity_sol: float | None,
    venue: str,
    fee_bps: int = 25,
) -> VenueQuote | None:
    """Build a VenueQuote from a graduated AMM pool priced in SOL.

    ``price_sol`` is SOL/token (aggregated price feed); ``liquidity_sol`` is the pool's
    SOL depth (DexScreener snapshot). Returns None when price is missing/non-positive.
    """
    if price_sol is None or price_sol <= 0:
        return None
    return VenueQuote(
        venue=venue,
        price=price_sol,
        liquidity_sol=liquidity_sol or 0.0,
        fee_bps=fee_bps,
    )


class ArbitrageMonitor:
    """Read-only monitor: collect real per-venue quotes, run the detector, surface any
    actionable opportunity on the event bus. Never executes."""

    def __init__(
        self,
        detector: ArbitrageDetector | None = None,
        size_sol: float = 0.1,
        event_bus: Any = None,
        logger: Any = None,
    ) -> None:
        self.detector = detector or ArbitrageDetector()
        self.size_sol = size_sol
        self._bus = event_bus
        self._logger = logger
        self.scans = 0
        self.opportunities = 0

    async def _collect(self, sources: list[QuoteSource]) -> list[VenueQuote]:
        quotes: list[VenueQuote] = []
        for src in sources:
            try:
                q = await src()
            except Exception as e:  # noqa: BLE001 - one bad venue must not sink the scan
                self._log("warning", f"arb quote source error: {e}")
                continue
            if q is not None:
                quotes.append(q)
        return quotes

    async def scan(self, token_address: str, sources: list[QuoteSource]) -> ArbOpportunity | None:
        """Collect quotes from the sources and return an ACTIONABLE opportunity (net
        edge clears the threshold), emitting an event for it. Read-only."""
        self.scans += 1
        quotes = await self._collect(sources)
        opp = self.detector.detect(token_address, quotes, self.size_sol)
        if opp is None:
            return None

        self.opportunities += 1
        self._log(
            "info",
            f"ARB {token_address[:8]}... {opp.buy_venue}->{opp.sell_venue} "
            f"net={opp.net_edge_bps:.0f}bps ~{opp.est_profit_sol:.4f} SOL",
        )
        if self._bus is not None:
            await self._bus.emit(
                arbitrage_opportunity_event(
                    token_address=token_address,
                    buy_venue=opp.buy_venue,
                    sell_venue=opp.sell_venue,
                    net_edge_bps=opp.net_edge_bps,
                    est_profit_sol=opp.est_profit_sol,
                    size_sol=opp.size_sol,
                    gross_edge_bps=opp.gross_edge_bps,
                )
            )
        return opp

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        method = getattr(self._logger, level, None)
        try:
            if callable(method):
                method(msg)
                return
        except TypeError:
            pass
        fallback = getattr(self._logger, "warning", None) or getattr(self._logger, "info", None)
        if callable(fallback):
            fallback(msg)
