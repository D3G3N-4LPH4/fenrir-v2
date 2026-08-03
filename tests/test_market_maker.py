#!/usr/bin/env python3
"""
FENRIR - Market-making inventory primitive tests (Phase 4, read-only/simulation)

Covers the quoting model (inventory skew direction, spread, clamps), inventory
bookkeeping (weighted-avg cost basis, realized PnL, cash), and the paper-trading
simulator's key properties: oscillation captures spread, a trend is bounded by the
inventory cap, and live execution is refused. No network.
"""

from __future__ import annotations

import pytest

from fenrir.trading.market_maker import (
    InventoryState,
    MarketMaker,
    MarketMakerConfig,
)


class TestConfig:
    def test_min_spread_clamp(self) -> None:
        cfg = MarketMakerConfig(spread_bps=10, min_spread_bps=60)
        assert cfg.spread_bps == 60

    def test_live_execution_refused(self) -> None:
        with pytest.raises(NotImplementedError):
            MarketMaker(simulation=False)


class TestQuote:
    def _mm_with_inventory(self, base_tokens: float, mid: float) -> MarketMaker:
        mm = MarketMaker()
        mm.inventory = InventoryState(base_tokens=base_tokens, avg_entry_price=mid, cash_sol=1.0)
        return mm

    def test_symmetric_at_target_inventory(self) -> None:
        # target ratio 0.5, cap 1.0 → target value 0.5; at mid 1.0 that's 0.5 tokens.
        mm = self._mm_with_inventory(0.5, 1.0)
        q = mm.quote(1.0)
        assert q.inventory_deviation == pytest.approx(0.0, abs=1e-9)
        assert q.skew_bps == pytest.approx(0.0, abs=1e-9)
        assert q.reservation == pytest.approx(1.0)
        # spread 200bps → ±1% around mid.
        assert q.bid_price == pytest.approx(0.99)
        assert q.ask_price == pytest.approx(1.01)
        assert q.spread_bps == pytest.approx(200.0, abs=1e-6)

    def test_over_inventory_skews_quotes_down(self) -> None:
        # 0.9 tokens @ mid 1.0 → value 0.9, deviation +0.4 → reservation below mid.
        mm = self._mm_with_inventory(0.9, 1.0)
        q = mm.quote(1.0)
        assert q.inventory_deviation == pytest.approx(0.4)
        assert q.skew_bps > 0
        assert q.reservation < 1.0
        # both quotes pulled below the symmetric 0.99 / 1.01.
        assert q.bid_price < 0.99
        assert q.ask_price < 1.01

    def test_under_inventory_skews_quotes_up(self) -> None:
        # empty inventory → deviation -0.5 → reservation above mid (rebuild).
        mm = MarketMaker()
        q = mm.quote(1.0)
        assert q.inventory_deviation == pytest.approx(-0.5)
        assert q.skew_bps < 0
        assert q.reservation > 1.0
        assert q.bid_price > 0.99
        assert q.ask_price > 1.01

    def test_rejects_nonpositive_mid(self) -> None:
        with pytest.raises(ValueError):
            MarketMaker().quote(0.0)


class TestInventoryBookkeeping:
    def test_buy_updates_cash_and_basis(self) -> None:
        mm = MarketMaker()
        mm.inventory = InventoryState(cash_sol=1.0)
        fill = mm.record_fill("buy", price=1.0, size_sol=0.05)
        assert fill is not None and fill.side == "buy"
        assert mm.inventory.base_tokens == pytest.approx(0.05)
        assert mm.inventory.cash_sol == pytest.approx(0.95)
        assert mm.inventory.avg_entry_price == pytest.approx(1.0)

    def test_weighted_average_cost_basis(self) -> None:
        mm = MarketMaker()
        mm.inventory = InventoryState(cash_sol=1.0)
        mm.record_fill("buy", price=1.0, size_sol=0.05)  # 0.05 tokens @ 1.0
        mm.record_fill("buy", price=2.0, size_sol=0.05)  # 0.025 tokens @ 2.0
        # avg = (0.05*1.0 + 0.025*2.0) / 0.075 = 0.10/0.075 = 1.3333
        assert mm.inventory.base_tokens == pytest.approx(0.075)
        assert mm.inventory.avg_entry_price == pytest.approx(1.3333, abs=1e-3)

    def test_sell_realizes_pnl(self) -> None:
        mm = MarketMaker()
        mm.inventory = InventoryState(cash_sol=1.0)
        mm.record_fill("buy", price=1.0, size_sol=0.10)  # 0.10 tokens @ 1.0
        fill = mm.record_fill("sell", price=1.20, size_sol=0.06)  # sell 0.05 tokens @ 1.2
        assert fill is not None and fill.side == "sell"
        # sold 0.06/1.2 = 0.05 tokens; realized = 0.05 * (1.2 - 1.0) = 0.01
        assert mm.inventory.realized_pnl_sol == pytest.approx(0.01)
        assert mm.inventory.base_tokens == pytest.approx(0.05)

    def test_cannot_sell_without_inventory(self) -> None:
        mm = MarketMaker()
        mm.inventory = InventoryState(cash_sol=1.0)
        assert mm.record_fill("sell", price=1.0, size_sol=0.05) is None

    def test_cannot_buy_without_cash(self) -> None:
        mm = MarketMaker()
        mm.inventory = InventoryState(cash_sol=0.0)
        assert mm.record_fill("buy", price=1.0, size_sol=0.05) is None

    def test_can_buy_respects_cap(self) -> None:
        mm = MarketMaker(MarketMakerConfig(max_inventory_sol=0.1, order_size_sol=0.05))
        mm.inventory = InventoryState(base_tokens=0.09, avg_entry_price=1.0, cash_sol=1.0)
        # value 0.09 @ mid 1.0; +0.05 would exceed the 0.1 cap.
        assert mm.can_buy(1.0) is False


class TestSimulate:
    def test_oscillation_captures_spread(self) -> None:
        # Prints oscillate ±4% around a stable 1.0 — amplitude wider than the half
        # spread (1%), so resting quotes fill on both sides around a stable fair value.
        prices = [1.0, 0.96, 1.04, 0.96, 1.04, 0.96, 1.04, 0.96, 1.04, 0.96, 1.04, 1.0]
        mm = MarketMaker()
        res = mm.simulate(prices)
        assert res.buys > 0
        assert res.sells > 0
        assert res.realized_pnl_sol > 0  # spread captured
        # Inventory never breached the cap.
        assert res.max_inventory_value_sol <= mm.config.max_inventory_sol + 1e-9

    def test_downtrend_bounded_by_inventory_cap(self) -> None:
        # A sustained decline: the maker keeps buying dips but only up to the cap,
        # then stops — bounded inventory, an honest adverse-selection loss.
        prices = [1.0 * (0.97**i) for i in range(60)]
        mm = MarketMaker()
        res = mm.simulate(prices)
        assert res.sells == 0  # price never rises to the ask
        assert res.buys > 0
        assert res.max_inventory_value_sol <= mm.config.max_inventory_sol + 1e-9
        assert res.ending_inventory_tokens > 0
        assert res.unrealized_pnl_sol < 0  # bought a falling knife

    def test_flat_market_no_fills(self) -> None:
        # A dead-flat print stream never crosses the quotes → no activity.
        res = MarketMaker().simulate([1.0] * 20)
        assert res.buys == 0
        assert res.sells == 0
        assert res.realized_pnl_sol == 0.0

    def test_empty_series(self) -> None:
        res = MarketMaker().simulate([])
        assert res.fills == []
        assert res.total_pnl_sol == 0.0
