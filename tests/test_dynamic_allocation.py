#!/usr/bin/env python3
"""
FENRIR - Dynamic capital allocation tests (Phase 2b)

A strategy's effective budget scales with its trailing win rate, within hard
floor/ceiling multipliers so no strategy runs away with the book. Off by default.
"""

from __future__ import annotations

import pytest

from fenrir.core.budget import AllocationConfig, BudgetTracker


def _tracker(**cfg) -> BudgetTracker:
    # AllocationConfig defaults window=20, min_trades=5; cases override as needed.
    return BudgetTracker(AllocationConfig(enabled=True, **cfg))


def _record(tracker: BudgetTracker, sid: str, results: list[bool]) -> None:
    for win in results:
        tracker.record_buy(sid, 0.1)
        tracker.record_sell(sid, 0.11 if win else 0.05, pnl_pct=1.0 if win else -1.0)


class TestAllocationDisabledByDefault:
    def test_off_returns_base_budget(self):
        t = BudgetTracker()  # default AllocationConfig(enabled=False)
        _record(t, "s", [True] * 10)
        assert t.allocation_multiplier("s") == 1.0
        assert t.effective_budget("s", 0.5) == 0.5


class TestFloorCeiling:
    def test_all_wins_hits_ceiling(self):
        t = _tracker(floor_mult=0.5, ceiling_mult=2.0)
        _record(t, "s", [True] * 10)
        assert t.allocation_multiplier("s") == 2.0  # win_rate 1.0 -> ceiling
        assert t.effective_budget("s", 0.5) == 1.0

    def test_all_losses_hits_floor(self):
        t = _tracker(floor_mult=0.5, ceiling_mult=2.0)
        _record(t, "s", [False] * 10)
        assert t.allocation_multiplier("s") == 0.5  # win_rate 0.0 -> floor
        assert t.effective_budget("s", 0.5) == 0.25

    def test_mixed_scales_between(self):
        t = _tracker(floor_mult=0.5, ceiling_mult=2.0)
        _record(t, "s", [True] * 8 + [False] * 2)  # win_rate 0.8
        # 0.5 + 0.8 * (2.0 - 0.5) = 1.7
        assert t.allocation_multiplier("s") == pytest.approx(1.7)

    def test_never_exceeds_ceiling_or_floor(self):
        t = _tracker(floor_mult=0.8, ceiling_mult=1.2)
        _record(t, "hot", [True] * 15)
        _record(t, "cold", [False] * 15)
        assert t.allocation_multiplier("hot") == 1.2
        assert t.allocation_multiplier("cold") == 0.8


class TestMinTrades:
    def test_uses_base_until_enough_samples(self):
        t = _tracker(min_trades=5)
        _record(t, "s", [True] * 4)  # only 4 trades
        assert t.allocation_multiplier("s") == 1.0  # not enough signal yet
        _record(t, "s", [True])  # now 5
        assert t.allocation_multiplier("s") == 2.0

    def test_unknown_strategy_is_base(self):
        t = _tracker()
        assert t.effective_budget("never_traded", 0.5) == 0.5


class TestWindow:
    def test_only_recent_trades_count(self):
        t = _tracker(window=10, min_trades=5)
        # 10 losses then 10 wins; the window sees only the last 10 (all wins).
        _record(t, "s", [False] * 10 + [True] * 10)
        assert t.allocation_multiplier("s") == 2.0

    def test_rebalances_as_performance_decays(self):
        t = _tracker(window=10, floor_mult=0.5, ceiling_mult=2.0)
        _record(t, "s", [True] * 10)
        assert t.allocation_multiplier("s") == 2.0
        _record(t, "s", [False] * 5)  # window now 5 wins / 5 losses = 0.5
        assert t.allocation_multiplier("s") == pytest.approx(1.25)


class TestConfigWiring:
    def test_built_from_config(self, monkeypatch):
        from fenrir.config import BotConfig, TradingMode

        monkeypatch.setenv("ALLOCATION_ENABLED", "true")
        monkeypatch.setenv("ALLOCATION_CEILING_MULT", "3.0")
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        t = cfg.build_budget_tracker()
        assert t.allocation.enabled is True
        assert t.allocation.ceiling_mult == 3.0
