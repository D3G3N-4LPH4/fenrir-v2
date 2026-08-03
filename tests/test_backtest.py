#!/usr/bin/env python3
"""
FENRIR - Backtester tests (Phase 6, backtesting rigor)

Replays REAL momentum signals through the SignalBacktester and verifies each exit path
(take-profit, stop-loss, trailing-stop, max-hold, end-of-data) using the strategy's own
TradeParams, plus non-entry on a rejected snapshot, the unified strength on trades, and
the metric math (win rate, profit factor, expectancy, Sharpe sign, max drawdown). No
network.
"""

from __future__ import annotations

from typing import Any

import pytest

from fenrir.backtest import (
    BacktestSample,
    BacktestTrade,
    SignalBacktester,
    compute_metrics,
    max_drawdown_pct,
)
from fenrir.config import BotConfig
from fenrir.filters import MarketData
from fenrir.strategies.momentum import MomentumStrategy

TOKEN = "So11111111111111111111111111111111111111112"


@pytest.fixture
def strat() -> MomentumStrategy:
    return MomentumStrategy(BotConfig())


def _fire_md(**over: Any) -> MarketData:
    """A MarketData momentum fires on (uptrend, accelerating volume, buyers)."""
    base: dict[str, Any] = dict(
        token_address=TOKEN,
        pair_address="PAIR",
        dex_id="raydium",
        age_minutes=120.0,
        market_cap_usd=500_000.0,
        price_usd=0.001,
        liquidity_usd=100_000.0,
        volume_5m_usd=30_000.0,
        volume_1h_usd=200_000.0,
        txns_5m_buys=70,
        txns_5m_sells=30,
        price_change_5m_pct=2.0,
        price_change_1h_pct=25.0,
        price_change_24h_pct=150.0,
    )
    base.update(over)
    return MarketData(**base)  # type: ignore[arg-type]


def _sample(
    prices: list[float], frame_seconds: float = 600.0, md: MarketData | None = None
) -> BacktestSample:
    return BacktestSample(
        token_address=TOKEN,
        token_data={"token_address": TOKEN},
        market_data=md if md is not None else _fire_md(),
        forward_prices=prices,
        frame_seconds=frame_seconds,
    )


# Momentum TradeParams: take_profit 60%, stop_loss 15%, trailing 15%, max_hold 180min.


class TestExitPaths:
    def test_take_profit(self, strat: MomentumStrategy) -> None:
        res = SignalBacktester().run(strat, [_sample([1.0, 1.2, 1.6, 1.7])])
        assert res.samples_entered == 1
        t = res.trades[0]
        assert t.exit_reason == "take_profit"
        assert t.exit_price == pytest.approx(1.6)
        assert t.pnl_pct == pytest.approx(60.0)

    def test_stop_loss(self, strat: MomentumStrategy) -> None:
        res = SignalBacktester().run(strat, [_sample([1.0, 0.9, 0.8])])
        t = res.trades[0]
        assert t.exit_reason == "stop_loss"
        assert t.pnl_pct == pytest.approx(-20.0)

    def test_trailing_stop(self, strat: MomentumStrategy) -> None:
        # Peaks at 1.4 → trail 1.19; drop to 1.1 trips the trail (not the -15% hard stop).
        res = SignalBacktester().run(strat, [_sample([1.0, 1.3, 1.4, 1.1])])
        t = res.trades[0]
        assert t.exit_reason == "trailing_stop"
        assert t.pnl_pct == pytest.approx(10.0)

    def test_max_hold(self, strat: MomentumStrategy) -> None:
        # frame 600s, max_hold 180min → 18 frames; flat prices trip nothing else.
        res = SignalBacktester().run(strat, [_sample([1.0] * 25, frame_seconds=600.0)])
        t = res.trades[0]
        assert t.exit_reason == "max_hold"
        assert t.hold_frames == 18
        assert t.pnl_pct == pytest.approx(0.0)

    def test_end_of_data(self, strat: MomentumStrategy) -> None:
        res = SignalBacktester().run(strat, [_sample([1.0, 1.05, 1.1], frame_seconds=600.0)])
        t = res.trades[0]
        assert t.exit_reason == "end_of_data"
        assert t.pnl_pct == pytest.approx(10.0)


class TestEntry:
    def test_no_entry_on_rejected_snapshot(self, strat: MomentumStrategy) -> None:
        # Flat 1h change → momentum rejects → no trade.
        md = _fire_md(price_change_1h_pct=0.0, price_change_5m_pct=0.0)
        res = SignalBacktester().run(strat, [_sample([1.0, 1.6], md=md)])
        assert res.samples_evaluated == 1
        assert res.samples_entered == 0
        assert res.trades == []

    def test_trade_carries_unified_strength(self, strat: MomentumStrategy) -> None:
        res = SignalBacktester().run(strat, [_sample([1.0, 1.6, 1.7])])
        t = res.trades[0]
        assert 0.0 < t.strength <= 1.0  # normalized Signal strength

    def test_empty_forward_prices_skipped(self, strat: MomentumStrategy) -> None:
        res = SignalBacktester().run(strat, [_sample([])])
        assert res.samples_evaluated == 0
        assert res.trades == []


class TestMetrics:
    def _trade(self, pnl: float) -> BacktestTrade:
        return BacktestTrade(TOKEN, "s", 1.0, 1.0, "end_of_data", 1, pnl, 0.5)

    def test_basic_stats(self) -> None:
        m = compute_metrics([self._trade(p) for p in (10.0, -5.0, -8.0, 12.0)])
        assert m.trades == 4
        assert m.wins == 2
        assert m.losses == 2
        assert m.win_rate == pytest.approx(0.5)
        assert m.expectancy_pct == pytest.approx(2.25)
        assert m.total_return_pct == pytest.approx(9.0)
        # gross profit 22, gross loss 13 → 1.6923
        assert m.profit_factor == pytest.approx(22.0 / 13.0)
        assert m.sharpe > 0  # positive expectancy

    def test_no_losses_profit_factor_zero(self) -> None:
        m = compute_metrics([self._trade(5.0), self._trade(3.0)])
        assert m.profit_factor == 0.0  # undefined denominator → 0 sentinel
        assert m.win_rate == 1.0

    def test_max_drawdown(self) -> None:
        # cumulative 10,5,-3,9 → peak 10, trough -3 → dd 13.
        assert max_drawdown_pct([10.0, -5.0, -8.0, 12.0]) == pytest.approx(13.0)

    def test_empty_is_zero(self) -> None:
        m = compute_metrics([])
        assert m.trades == 0
        assert m.win_rate == 0.0
        assert m.sharpe == 0.0


class TestResult:
    def test_to_dict(self, strat: MomentumStrategy) -> None:
        res = SignalBacktester().run(strat, [_sample([1.0, 1.6, 1.7])])
        d = res.to_dict()
        assert d["strategy_id"] == "momentum"
        assert d["samples_entered"] == 1
        assert d["metrics"]["trades"] == 1
