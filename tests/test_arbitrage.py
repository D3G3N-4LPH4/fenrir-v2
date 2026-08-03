#!/usr/bin/env python3
"""
FENRIR - Cross-venue arbitrage divergence detector tests (Phase 4, read-only)

Covers the slippage model, the net-of-cost edge arithmetic, actionability
thresholding, venue selection (buy cheapest / sell dearest), and the guards
(too few venues, no divergence, same venue, thin liquidity, bad size). No network,
no execution.
"""

from __future__ import annotations

import pytest

from fenrir.trading.arbitrage import (
    ArbConfig,
    ArbitrageDetector,
    VenueQuote,
)

TOKEN = "So11111111111111111111111111111111111111112"


class TestSlippageModel:
    def test_fraction_of_pool(self) -> None:
        d = ArbitrageDetector()
        # 1 SOL into a 100 SOL pool → 1% impact = 100 bps.
        assert d.slippage_bps(1.0, 100.0) == pytest.approx(100.0)

    def test_capped(self) -> None:
        d = ArbitrageDetector(ArbConfig(max_slippage_bps=500.0))
        # 10 SOL into a 10 SOL pool would be 10000 bps → capped at 500.
        assert d.slippage_bps(10.0, 10.0) == 500.0

    def test_unknown_liquidity_is_max(self) -> None:
        d = ArbitrageDetector(ArbConfig(max_slippage_bps=500.0))
        assert d.slippage_bps(1.0, 0.0) == 500.0


class TestEvaluate:
    def _quotes(self) -> list[VenueQuote]:
        return [
            VenueQuote("pumpfun", price=1.00, liquidity_sol=100.0, fee_bps=100),
            VenueQuote("raydium", price=1.05, liquidity_sol=100.0, fee_bps=25),
        ]

    def test_costed_breakdown(self) -> None:
        d = ArbitrageDetector(ArbConfig(tx_cost_sol=0.002, min_net_edge_bps=50.0))
        opp = d.evaluate(TOKEN, self._quotes(), size_sol=1.0)
        assert opp is not None
        # buy cheapest (pumpfun @1.00), sell dearest (raydium @1.05).
        assert opp.buy_venue == "pumpfun"
        assert opp.sell_venue == "raydium"
        # gross = (1.05-1.00)/1.00 = 500 bps
        assert opp.gross_edge_bps == pytest.approx(500.0)
        # buy cost = 100 fee + 100 slip; sell cost = 25 fee + 100 slip
        assert opp.buy_cost_bps == pytest.approx(200.0)
        assert opp.sell_cost_bps == pytest.approx(125.0)
        # tx = 0.002/1.0 = 20 bps
        assert opp.tx_cost_bps == pytest.approx(20.0)
        # net = 500 - 200 - 125 - 20 = 155 bps
        assert opp.net_edge_bps == pytest.approx(155.0)
        assert opp.est_profit_sol == pytest.approx(1.0 * 155.0 / 10_000)
        assert opp.actionable is True

    def test_below_threshold_not_actionable(self) -> None:
        d = ArbitrageDetector(ArbConfig(tx_cost_sol=0.002, min_net_edge_bps=50.0))
        quotes = [
            VenueQuote("pumpfun", price=1.00, liquidity_sol=100.0, fee_bps=100),
            VenueQuote("raydium", price=1.01, liquidity_sol=100.0, fee_bps=25),
        ]
        opp = d.evaluate(TOKEN, quotes, size_sol=1.0)
        assert opp is not None
        # gross 100 - 200 - 125 - 20 = -245 bps → not actionable, negative profit.
        assert opp.net_edge_bps == pytest.approx(-245.0)
        assert opp.est_profit_sol < 0
        assert opp.actionable is False

    def test_selects_widest_pair_across_three_venues(self) -> None:
        d = ArbitrageDetector()
        quotes = [
            VenueQuote("a", price=1.02, liquidity_sol=100.0, fee_bps=25),
            VenueQuote("b", price=0.98, liquidity_sol=100.0, fee_bps=25),  # cheapest
            VenueQuote("c", price=1.06, liquidity_sol=100.0, fee_bps=25),  # dearest
        ]
        opp = d.evaluate(TOKEN, quotes, size_sol=1.0)
        assert opp is not None
        assert opp.buy_venue == "b"
        assert opp.sell_venue == "c"

    def test_total_cost_bps(self) -> None:
        d = ArbitrageDetector(ArbConfig(tx_cost_sol=0.002))
        opp = d.evaluate(TOKEN, self._quotes(), size_sol=1.0)
        assert opp is not None
        assert opp.total_cost_bps == pytest.approx(200.0 + 125.0 + 20.0)


class TestGuards:
    def test_too_few_venues(self) -> None:
        d = ArbitrageDetector()
        one = [VenueQuote("a", price=1.0, liquidity_sol=100.0)]
        assert d.evaluate(TOKEN, one, size_sol=1.0) is None

    def test_no_divergence(self) -> None:
        d = ArbitrageDetector()
        flat = [
            VenueQuote("a", price=1.0, liquidity_sol=100.0),
            VenueQuote("b", price=1.0, liquidity_sol=100.0),
        ]
        assert d.evaluate(TOKEN, flat, size_sol=1.0) is None

    def test_thin_liquidity_filtered(self) -> None:
        d = ArbitrageDetector(ArbConfig(min_liquidity_sol=50.0))
        quotes = [
            VenueQuote("a", price=1.00, liquidity_sol=100.0),
            VenueQuote("b", price=1.10, liquidity_sol=10.0),  # too thin → filtered
        ]
        # Only one usable venue remains → no pair.
        assert d.evaluate(TOKEN, quotes, size_sol=1.0) is None

    def test_zero_price_ignored(self) -> None:
        d = ArbitrageDetector()
        quotes = [
            VenueQuote("a", price=0.0, liquidity_sol=100.0),
            VenueQuote("b", price=1.10, liquidity_sol=100.0),
        ]
        assert d.evaluate(TOKEN, quotes, size_sol=1.0) is None

    def test_bad_size_raises(self) -> None:
        d = ArbitrageDetector()
        with pytest.raises(ValueError):
            d.evaluate(TOKEN, [], size_sol=0.0)


class TestDetect:
    def test_returns_only_actionable(self) -> None:
        d = ArbitrageDetector(ArbConfig(min_net_edge_bps=50.0))
        wide = [
            VenueQuote("pumpfun", price=1.00, liquidity_sol=100.0, fee_bps=100),
            VenueQuote("raydium", price=1.05, liquidity_sol=100.0, fee_bps=25),
        ]
        narrow = [
            VenueQuote("pumpfun", price=1.00, liquidity_sol=100.0, fee_bps=100),
            VenueQuote("raydium", price=1.01, liquidity_sol=100.0, fee_bps=25),
        ]
        assert d.detect(TOKEN, wide, size_sol=1.0) is not None
        assert d.detect(TOKEN, narrow, size_sol=1.0) is None

    def test_size_amortizes_tx_cost(self) -> None:
        # Larger size spreads the fixed tx cost over more notional → higher net edge,
        # but also more slippage. With deep liquidity, net edge rises with size.
        d = ArbitrageDetector(ArbConfig(tx_cost_sol=0.01))
        quotes = [
            VenueQuote("a", price=1.00, liquidity_sol=100_000.0, fee_bps=10),
            VenueQuote("b", price=1.03, liquidity_sol=100_000.0, fee_bps=10),
        ]
        small = d.evaluate(TOKEN, quotes, size_sol=0.1)
        large = d.evaluate(TOKEN, quotes, size_sol=2.0)
        assert small is not None and large is not None
        # tx cost bps: small = 0.01/0.1=1000 bps, large = 0.01/2=50 bps.
        assert small.tx_cost_bps > large.tx_cost_bps
        assert large.net_edge_bps > small.net_edge_bps
