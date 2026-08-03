#!/usr/bin/env python3
"""
FENRIR - Arbitrage monitor tests (Phase 4.4b, read-only real quotes)

Covers the curve/AMM → VenueQuote bridges, the read-only monitor's collect →
detect → emit flow, actionability gating, resilience to a failing quote source, and
that it never executes. No network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fenrir.trading.arbitrage import ArbConfig, ArbitrageDetector, VenueQuote
from fenrir.trading.arbitrage_monitor import (
    ArbitrageMonitor,
    venue_quote_from_amm,
    venue_quote_from_curve,
)

TOKEN = "So11111111111111111111111111111111111111112"


def _source(quote: VenueQuote | None) -> Any:
    async def _get() -> VenueQuote | None:
        return quote

    return _get


class TestCurveBridge:
    def test_from_curve(self) -> None:
        # price = virtual_sol/virtual_token/1e9 handled by get_price(); provide it directly.
        curve = SimpleNamespace(get_price=lambda: 0.0005, real_sol_reserves=30 * 10**9)
        q = venue_quote_from_curve(curve)
        assert q is not None
        assert q.venue == "pumpfun_curve"
        assert q.price == pytest.approx(0.0005)
        assert q.liquidity_sol == pytest.approx(30.0)
        assert q.fee_bps == 100

    def test_from_curve_unpriceable(self) -> None:
        curve = SimpleNamespace(get_price=lambda: 0.0, real_sol_reserves=0)
        assert venue_quote_from_curve(curve) is None


class TestAmmBridge:
    def test_from_amm(self) -> None:
        q = venue_quote_from_amm(price_sol=0.00052, liquidity_sol=80.0, venue="raydium")
        assert q is not None
        assert q.venue == "raydium"
        assert q.price == pytest.approx(0.00052)
        assert q.liquidity_sol == pytest.approx(80.0)
        assert q.fee_bps == 25

    def test_from_amm_missing_price(self) -> None:
        assert venue_quote_from_amm(None, 80.0, "raydium") is None
        assert venue_quote_from_amm(0.0, 80.0, "raydium") is None

    def test_from_amm_none_liquidity_is_zero(self) -> None:
        q = venue_quote_from_amm(0.001, None, "raydium")
        assert q is not None
        assert q.liquidity_sol == 0.0


class TestMonitorScan:
    async def test_emits_actionable_opportunity(self) -> None:
        bus = SimpleNamespace(emit=AsyncMock())
        mon = ArbitrageMonitor(
            ArbitrageDetector(ArbConfig(min_net_edge_bps=50.0, tx_cost_sol=0.002)),
            size_sol=1.0,
            event_bus=bus,
        )
        sources = [
            _source(VenueQuote("pumpfun_curve", price=1.00, liquidity_sol=100.0, fee_bps=100)),
            _source(VenueQuote("raydium", price=1.05, liquidity_sol=100.0, fee_bps=25)),
        ]
        opp = await mon.scan(TOKEN, sources)
        assert opp is not None and opp.actionable
        assert mon.opportunities == 1
        bus.emit.assert_awaited_once()
        ev = bus.emit.await_args.args[0]
        assert ev.event_type == "ARBITRAGE_OPPORTUNITY"
        assert ev.data["buy_venue"] == "pumpfun_curve"
        assert ev.data["sell_venue"] == "raydium"

    async def test_no_emit_when_not_actionable(self) -> None:
        bus = SimpleNamespace(emit=AsyncMock())
        mon = ArbitrageMonitor(
            ArbitrageDetector(ArbConfig(min_net_edge_bps=50.0)),
            size_sol=1.0,
            event_bus=bus,
        )
        sources = [
            _source(VenueQuote("pumpfun_curve", price=1.00, liquidity_sol=100.0, fee_bps=100)),
            _source(VenueQuote("raydium", price=1.01, liquidity_sol=100.0, fee_bps=25)),
        ]
        opp = await mon.scan(TOKEN, sources)
        assert opp is None
        assert mon.opportunities == 0
        bus.emit.assert_not_awaited()

    async def test_survives_failing_source(self) -> None:
        async def boom() -> VenueQuote | None:
            raise RuntimeError("curve fetch failed")

        mon = ArbitrageMonitor(size_sol=1.0)
        sources = [
            boom,
            _source(VenueQuote("a", price=1.00, liquidity_sol=100.0, fee_bps=25)),
            _source(VenueQuote("b", price=1.06, liquidity_sol=100.0, fee_bps=25)),
        ]
        opp = await mon.scan(TOKEN, sources)
        # One source errored but the other two still produced an actionable pair.
        assert opp is not None
        assert mon.scans == 1

    async def test_none_quotes_dropped(self) -> None:
        mon = ArbitrageMonitor(size_sol=1.0)
        sources = [
            _source(None),
            _source(VenueQuote("a", price=1.0, liquidity_sol=100.0)),
        ]
        # Only one real quote → no pair → nothing.
        assert await mon.scan(TOKEN, sources) is None

    async def test_no_bus_is_fine(self) -> None:
        mon = ArbitrageMonitor(
            ArbitrageDetector(ArbConfig(min_net_edge_bps=10.0)),
            size_sol=1.0,
        )
        sources = [
            _source(VenueQuote("a", price=1.00, liquidity_sol=100.0, fee_bps=10)),
            _source(VenueQuote("b", price=1.10, liquidity_sol=100.0, fee_bps=10)),
        ]
        opp = await mon.scan(TOKEN, sources)
        assert opp is not None  # returns the opportunity even with no event bus
