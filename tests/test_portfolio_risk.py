#!/usr/bin/env python3
"""
FENRIR - PortfolioRiskManager tests (Phase 2a)

Acceptance criteria: correlation flagging (shared creator / launch-window cluster),
the portfolio drawdown breaker triggering across strategies, and aggregate exposure.
Pure logic, no I/O.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fenrir.core.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _mgr(**cfg) -> PortfolioRiskManager:
    return PortfolioRiskManager(PortfolioRiskConfig(**cfg))


class TestExposureCap:
    def test_blocks_when_projected_exposure_exceeds_cap(self):
        m = _mgr(max_total_exposure_sol=0.3)
        m.record_open("A", "sniper", 0.2, now=T0)
        d = m.check("sniper", 0.2)  # 0.2 + 0.2 = 0.4 > 0.3
        assert d.allowed is False
        assert "exposure_cap" in d.flags

    def test_allows_within_cap(self):
        m = _mgr(max_total_exposure_sol=0.5)
        m.record_open("A", "sniper", 0.2, now=T0)
        assert m.check("sniper", 0.2).allowed is True

    def test_cap_disabled_when_zero(self):
        m = _mgr(max_total_exposure_sol=0.0)
        m.record_open("A", "sniper", 100.0, now=T0)
        assert m.check("sniper", 100.0).allowed is True

    def test_total_exposure_tracks_opens_and_closes(self):
        m = _mgr()
        m.record_open("A", "s", 0.2, now=T0)
        m.record_open("B", "s", 0.3, now=T0)
        assert m.total_exposure_sol == 0.5
        m.record_close("A", 0.05)
        assert m.total_exposure_sol == 0.3


class TestCreatorCorrelation:
    def test_blocks_third_position_from_same_creator(self):
        m = _mgr(max_positions_per_creator=2)
        m.record_open("A", "sniper", 0.1, creator="DEV1", now=T0)
        m.record_open("B", "reversal", 0.1, creator="DEV1", now=T0)  # cross-strategy
        d = m.check("sniper", 0.1, creator="DEV1")
        assert d.allowed is False
        assert "creator_concentration" in d.flags

    def test_different_creators_are_independent(self):
        m = _mgr(max_positions_per_creator=2)
        m.record_open("A", "s", 0.1, creator="DEV1", now=T0)
        m.record_open("B", "s", 0.1, creator="DEV1", now=T0)
        assert m.check("s", 0.1, creator="DEV2").allowed is True

    def test_no_creator_is_not_correlated(self):
        m = _mgr(max_positions_per_creator=1)
        m.record_open("A", "s", 0.1, creator=None, now=T0)
        assert m.check("s", 0.1, creator=None).allowed is True

    def test_closing_frees_creator_slot(self):
        m = _mgr(max_positions_per_creator=2)
        m.record_open("A", "s", 0.1, creator="DEV1", now=T0)
        m.record_open("B", "s", 0.1, creator="DEV1", now=T0)
        m.record_close("A", 0.0)
        assert m.check("s", 0.1, creator="DEV1").allowed is True


class TestLaunchWindowCluster:
    def test_blocks_when_too_many_opened_in_window(self):
        m = _mgr(max_positions_per_window=3, launch_window_seconds=60)
        for i, tok in enumerate(("A", "B", "C")):
            m.record_open(tok, "s", 0.1, now=T0 + timedelta(seconds=i * 10))
        d = m.check("s", 0.1, now=T0 + timedelta(seconds=30))
        assert d.allowed is False
        assert "window_cluster" in d.flags

    def test_opens_outside_window_do_not_count(self):
        m = _mgr(max_positions_per_window=2, launch_window_seconds=60)
        m.record_open("A", "s", 0.1, now=T0)
        m.record_open("B", "s", 0.1, now=T0)
        # Two minutes later, the earlier opens age out of the window.
        assert m.check("s", 0.1, now=T0 + timedelta(seconds=130)).allowed is True


class TestDrawdownBreaker:
    def test_trips_across_strategies_and_halts_all_buys(self):
        m = _mgr(max_drawdown_sol=0.3, drawdown_min_trades=3)
        # Peak equity after a win, then losses across DIFFERENT strategies.
        m.record_open("A", "sniper", 0.1, now=T0)
        m.record_close("A", +0.5)  # peak = 0.5
        for tok, strat in (("B", "sniper"), ("C", "reversal"), ("D", "graduation")):
            m.record_open(tok, strat, 0.1, now=T0)
            m.record_close(tok, -0.15)  # realized 0.5 -> 0.05, drawdown 0.45 > 0.3
        assert m.breaker_tripped is True
        # A different strategy's buy is now blocked too.
        d = m.check("volume_anomaly", 0.05)
        assert d.allowed is False
        assert "drawdown_breaker" in d.flags

    def test_does_not_trip_below_min_trades(self):
        m = _mgr(max_drawdown_sol=0.1, drawdown_min_trades=5)
        m.record_open("A", "s", 0.1, now=T0)
        m.record_close("A", -0.5)  # big loss but only 1 trade
        assert m.breaker_tripped is False
        assert m.check("s", 0.05).allowed is True

    def test_reset_rearms_and_rebaselines_peak(self):
        m = _mgr(max_drawdown_sol=0.3, drawdown_min_trades=1)
        m.record_open("A", "s", 0.1, now=T0)
        m.record_close("A", +0.5)
        m.record_open("B", "s", 0.1, now=T0)
        m.record_close("B", -0.4)  # drawdown 0.4 > 0.3 -> tripped
        assert m.breaker_tripped is True
        m.reset_breaker()
        assert m.breaker_tripped is False
        assert m.drawdown_sol == 0.0  # peak rebaselined to current
        assert m.check("s", 0.05).allowed is True

    def test_disabled_when_zero(self):
        m = _mgr(max_drawdown_sol=0.0, drawdown_min_trades=1)
        m.record_open("A", "s", 0.1, now=T0)
        m.record_close("A", -100.0)
        assert m.breaker_tripped is False


class TestStatus:
    def test_status_snapshot(self):
        m = _mgr(max_total_exposure_sol=1.0)
        m.record_open("A", "s", 0.2, creator="DEV", now=T0)
        m.record_close("A", 0.1)
        s = m.status()
        assert s["closed_trades"] == 1
        assert s["realized_pnl_sol"] == 0.1
        assert s["open_positions"] == 0
        assert s["breaker_tripped"] is False
