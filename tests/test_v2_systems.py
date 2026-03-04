#!/usr/bin/env python3
"""
FENRIR v2 Systems - Test Suite

Covers: BudgetTracker, AuditChain, HistoricalMemory,
        EventBus, TradingStrategy, AIHealthMonitor
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.core.budget import BudgetTracker, TradeAuthorization
from fenrir.data.audit import AuditChain, AuditEventType
from fenrir.data.historical_memory import HistoricalMemory
from fenrir.events.adapters.health import AIHealthMonitor, DriftType, HealthMonitorConfig
from fenrir.events.bus import EventBus, EventListener
from fenrir.events.types import (
    EventCategory,
    EventSeverity,
    TradeEvent,
    ai_decision_event,
    buy_executed_event,
    sell_executed_event,
    token_detected_event,
)
from fenrir.strategies import STRATEGY_REGISTRY, SniperStrategy
from fenrir.strategies.base import StrategyState, TradingStrategy
from fenrir.strategies.graduation import GraduationStrategy
from fenrir.strategies.sniper import ConservativeSniperStrategy, DegenSniperStrategy


# ═══════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def config():
    return BotConfig()


# ═══════════════════════════════════════════════════════════════════
#  BUDGET TRACKER
# ═══════════════════════════════════════════════════════════════════


class TestBudgetTracker:
    def test_authorize_within_budget(self):
        tracker = BudgetTracker()
        auth = tracker.authorize_trade("sniper", 0.1, budget_sol=1.0, max_positions=5)
        assert auth.allowed
        assert auth.reason == "authorized"

    def test_reject_over_budget(self):
        tracker = BudgetTracker()
        tracker.record_buy("sniper", 0.9)
        auth = tracker.authorize_trade("sniper", 0.2, budget_sol=1.0, max_positions=5)
        assert not auth.allowed
        assert "over budget" in auth.reason

    def test_reject_at_position_limit(self):
        tracker = BudgetTracker()
        tracker.record_buy("sniper", 0.1)
        tracker.record_buy("sniper", 0.1)
        auth = tracker.authorize_trade("sniper", 0.1, budget_sol=1.0, max_positions=2)
        assert not auth.allowed
        assert "position limit" in auth.reason

    def test_reject_paused_strategy(self):
        tracker = BudgetTracker()
        auth = tracker.authorize_trade(
            "sniper", 0.1, budget_sol=1.0, max_positions=5, is_paused=True
        )
        assert not auth.allowed
        assert "paused" in auth.reason

    def test_reject_inactive_strategy(self):
        tracker = BudgetTracker()
        auth = tracker.authorize_trade(
            "sniper", 0.1, budget_sol=1.0, max_positions=5, is_active=False
        )
        assert not auth.allowed
        assert "deactivated" in auth.reason

    def test_global_sol_limit(self):
        tracker = BudgetTracker()
        tracker.set_global_limit(0.5)
        tracker.record_buy("s1", 0.4)
        auth = tracker.authorize_trade("s2", 0.2, budget_sol=1.0, max_positions=5)
        assert not auth.allowed
        assert "Global SOL limit" in auth.reason

    def test_record_buy_increments_positions(self):
        tracker = BudgetTracker()
        tracker.record_buy("sniper", 0.1)
        state = tracker._get_state("sniper")
        assert state.positions_open == 1
        assert state.trades_executed == 1
        assert state.sol_spent == pytest.approx(0.1)

    def test_record_sell_decrements_positions_tracks_win(self):
        tracker = BudgetTracker()
        tracker.record_buy("sniper", 0.1)
        tracker.record_sell("sniper", 0.15, pnl_pct=50.0)
        state = tracker._get_state("sniper")
        assert state.positions_open == 0
        assert state.wins == 1
        assert state.losses == 0

    def test_record_sell_tracks_loss(self):
        tracker = BudgetTracker()
        tracker.record_buy("sniper", 0.1)
        tracker.record_sell("sniper", 0.07, pnl_pct=-30.0)
        state = tracker._get_state("sniper")
        assert state.losses == 1

    def test_win_rate_calculation(self):
        tracker = BudgetTracker()
        tracker.record_buy("s", 0.1)
        tracker.record_sell("s", 0.15, 50.0)
        tracker.record_buy("s", 0.1)
        tracker.record_sell("s", 0.07, -30.0)
        state = tracker._get_state("s")
        assert state.win_rate == pytest.approx(0.5)

    def test_reset_strategy_clears_daily_counters(self):
        tracker = BudgetTracker()
        tracker.record_buy("sniper", 0.5)
        tracker.reset_strategy("sniper")
        state = tracker._get_state("sniper")
        assert state.sol_spent == 0.0
        assert state.trades_executed == 0

    def test_reset_strategy_preserves_open_positions(self):
        tracker = BudgetTracker()
        tracker.record_buy("sniper", 0.1)
        tracker.reset_strategy("sniper")
        state = tracker._get_state("sniper")
        assert state.positions_open == 1  # Still open

    def test_global_status_aggregates_all_strategies(self):
        tracker = BudgetTracker()
        tracker.record_buy("s1", 0.1)
        tracker.record_buy("s2", 0.2)
        status = tracker.get_global_status()
        assert status["total_sol_spent"] == pytest.approx(0.3)
        assert status["total_positions_open"] == 2
        assert status["strategies_tracked"] == 2

    def test_net_spent_accounts_for_returns(self):
        tracker = BudgetTracker()
        tracker.record_buy("sniper", 0.1)
        tracker.record_sell("sniper", 0.15, 50.0)
        state = tracker._get_state("sniper")
        assert state.net_spent == pytest.approx(-0.05)  # Profit: returned more than spent


# ═══════════════════════════════════════════════════════════════════
#  AUDIT CHAIN
# ═══════════════════════════════════════════════════════════════════


class TestAuditChain:
    def test_record_returns_audit_record(self, tmp_db):
        chain = AuditChain(db_path=tmp_db)
        record = chain.record(AuditEventType.TOKEN_DETECTED, token_address="ABC123")
        assert record.event_type == AuditEventType.TOKEN_DETECTED
        assert record.token_address == "ABC123"
        assert len(record.hash) == 64  # SHA256 hex
        chain.close()

    def test_chain_hashes_linked(self, tmp_db):
        chain = AuditChain(db_path=tmp_db)
        r1 = chain.record(AuditEventType.BOT_STARTED)
        r2 = chain.record(AuditEventType.TOKEN_DETECTED, token_address="ABC")
        assert r2.prev_hash == r1.hash
        chain.close()

    def test_verify_chain_valid(self, tmp_db):
        chain = AuditChain(db_path=tmp_db)
        for i in range(5):
            chain.record(AuditEventType.TOKEN_DETECTED, token_address=f"TKN{i}")
        valid, broken_at = chain.verify_chain()
        assert valid
        assert broken_at is None
        chain.close()

    def test_verify_empty_chain(self, tmp_db):
        chain = AuditChain(db_path=tmp_db)
        valid, broken_at = chain.verify_chain()
        assert valid
        assert broken_at is None
        chain.close()

    def test_verify_detects_tampering(self, tmp_db):
        import sqlite3

        chain = AuditChain(db_path=tmp_db)
        chain.record(AuditEventType.BUY_EXECUTED, payload={"amount": 0.1})
        chain.record(AuditEventType.SELL_EXECUTED, payload={"pnl": 10.0})
        chain.close()

        # Tamper with the first record's payload
        conn = sqlite3.connect(tmp_db)
        conn.execute("UPDATE audit_chain SET payload = '{\"amount\": 9999}' WHERE id = 1")
        conn.commit()
        conn.close()

        chain2 = AuditChain(db_path=tmp_db)
        valid, broken_at = chain2.verify_chain()
        assert not valid
        assert broken_at == 1
        chain2.close()

    def test_session_filter(self, tmp_db):
        chain = AuditChain(db_path=tmp_db, session_id="sess-A")
        chain.record(AuditEventType.TOKEN_DETECTED, token_address="T1")
        chain.record(AuditEventType.BOT_STARTED)
        records = chain.get_session_log(session_id="sess-A")
        assert len(records) == 2
        chain.close()

    def test_token_timeline(self, tmp_db):
        chain = AuditChain(db_path=tmp_db)
        chain.record(AuditEventType.TOKEN_DETECTED, token_address="WOLF")
        chain.record(AuditEventType.BUY_EXECUTED, token_address="WOLF")
        chain.record(AuditEventType.SELL_EXECUTED, token_address="WOLF")
        chain.record(AuditEventType.TOKEN_DETECTED, token_address="OTHER")
        timeline = chain.get_token_timeline("WOLF")
        assert len(timeline) == 3
        assert all(r.token_address == "WOLF" for r in timeline)
        chain.close()

    def test_chain_stats(self, tmp_db):
        chain = AuditChain(db_path=tmp_db)
        chain.record(AuditEventType.BOT_STARTED)
        chain.record(AuditEventType.BOT_STOPPED)
        stats = chain.get_chain_stats()
        assert stats["total_records"] == 2
        assert stats["current_session"] == chain.session_id
        chain.close()

    def test_payload_stored_and_retrieved(self, tmp_db):
        chain = AuditChain(db_path=tmp_db)
        chain.record(
            AuditEventType.BUY_EXECUTED,
            payload={"amount_sol": 0.1, "sig": "abc123"},
        )
        records = chain.get_session_log()
        assert records[0].payload["amount_sol"] == pytest.approx(0.1)
        assert records[0].payload["sig"] == "abc123"
        chain.close()

    def test_chain_persists_across_instances(self, tmp_db):
        chain1 = AuditChain(db_path=tmp_db, session_id="s1")
        chain1.record(AuditEventType.BOT_STARTED)
        last_hash = chain1._last_hash
        chain1.close()

        # New instance picks up where first left off
        chain2 = AuditChain(db_path=tmp_db, session_id="s2")
        assert chain2._last_hash == last_hash
        chain2.close()


# ═══════════════════════════════════════════════════════════════════
#  HISTORICAL MEMORY
# ═══════════════════════════════════════════════════════════════════


class TestHistoricalMemory:
    def test_record_and_retrieve_outcome(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        mem.record_outcome(
            token_address="ABC123",
            token_symbol="WOLF",
            creator_address="CREATOR1",
            initial_liquidity_sol=5.0,
            market_cap_sol=30.0,
            ai_decision="BUY",
            was_bought=True,
            pnl_pct=42.0,
            pnl_sol=0.042,
            hold_time_minutes=12,
        )
        assert mem.get_total_outcomes() == 1
        mem.close()

    def test_creator_stats_created_on_first_outcome(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        mem.record_outcome(
            "T1", creator_address="C1", was_bought=True, pnl_pct=50.0, pnl_sol=0.05
        )
        profile = mem.get_creator_profile("C1")
        assert profile is not None
        assert profile["total_launches"] == 1
        assert profile["tokens_bought"] == 1
        assert profile["tokens_profitable"] == 1
        mem.close()

    def test_creator_rug_detection(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        mem.record_outcome(
            "T1", creator_address="RUGGER", was_bought=True, pnl_pct=-90.0
        )
        profile = mem.get_creator_profile("RUGGER")
        assert profile["rug_count"] == 1
        mem.close()

    def test_creator_stats_aggregated_across_trades(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        mem.record_outcome("T1", creator_address="C1", was_bought=True, pnl_pct=100.0)
        mem.record_outcome("T2", creator_address="C1", was_bought=True, pnl_pct=50.0)
        mem.record_outcome("T3", creator_address="C1", was_bought=True, pnl_pct=-20.0)
        profile = mem.get_creator_profile("C1")
        assert profile["total_launches"] == 3
        assert profile["tokens_bought"] == 3
        assert profile["tokens_profitable"] == 2
        mem.close()

    def test_build_historical_context_creator(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        for i in range(3):
            mem.record_outcome(f"T{i}", creator_address="C1", was_bought=True, pnl_pct=20.0)
        ctx = mem.build_historical_context(creator_address="C1")
        assert "C1" in ctx or "HISTORICAL" in ctx
        mem.close()

    def test_build_historical_context_no_data_returns_empty(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        ctx = mem.build_historical_context(creator_address="UNKNOWN", initial_liquidity_sol=5.0)
        assert ctx == ""
        mem.close()

    def test_liquidity_range_context_requires_minimum_trades(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        # Only 2 trades — below the minimum of 3
        for i in range(2):
            mem.record_outcome(
                f"T{i}", was_bought=True, pnl_pct=10.0, initial_liquidity_sol=5.0
            )
        ctx = mem.build_historical_context(initial_liquidity_sol=5.0)
        assert ctx == ""
        mem.close()

    def test_strategy_performance(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        for _ in range(3):
            mem.record_outcome("T1", was_bought=True, pnl_pct=50.0, strategy_id="sniper")
        mem.record_outcome("T2", was_bought=True, pnl_pct=-20.0, strategy_id="sniper")
        perf = mem.get_strategy_performance("sniper")
        assert perf["total"] == 4
        assert perf["wins"] == 3
        assert perf["win_rate"] == pytest.approx(75.0)
        mem.close()

    def test_total_outcomes_count(self, tmp_db):
        mem = HistoricalMemory(db_path=tmp_db)
        for i in range(7):
            mem.record_outcome(f"T{i}", was_bought=(i % 2 == 0))
        assert mem.get_total_outcomes() == 7
        mem.close()


# ═══════════════════════════════════════════════════════════════════
#  EVENT BUS
# ═══════════════════════════════════════════════════════════════════


class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_reaches_registered_listener(self):
        bus = EventBus()
        received = []

        class Collector(EventListener):
            async def on_event(self, event: TradeEvent) -> None:
                received.append(event)

        bus.register(Collector())
        event = token_detected_event("ABC", "WOLF", "Wolf Token", 5.0, 30.0)
        await bus.emit(event)
        assert len(received) == 1
        assert received[0].event_type == "TOKEN_DETECTED"

    @pytest.mark.asyncio
    async def test_emit_reaches_multiple_listeners(self):
        bus = EventBus()
        counts = [0, 0]

        class Counter(EventListener):
            def __init__(self, idx):
                self.idx = idx

            async def on_event(self, event):
                counts[self.idx] += 1

        bus.register(Counter(0))
        bus.register(Counter(1))
        await bus.emit(token_detected_event("A", "X", "X Token", 1.0, 5.0))
        assert counts == [1, 1]

    @pytest.mark.asyncio
    async def test_severity_filter_blocks_low_severity(self):
        bus = EventBus()
        received = []

        class HighSeverityOnly(EventListener):
            min_severity = EventSeverity.WARNING

            async def on_event(self, event):
                received.append(event)

        bus.register(HighSeverityOnly())
        # INFO event — should be filtered
        await bus.emit(token_detected_event("A", "X", "X Token", 1.0, 5.0))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_severity_filter_passes_high_severity(self):
        bus = EventBus()
        received = []

        class HighSeverityOnly(EventListener):
            min_severity = EventSeverity.WARNING

            async def on_event(self, event):
                received.append(event)

        bus.register(HighSeverityOnly())
        event = TradeEvent(
            event_type="BUY_EXECUTED",
            category=EventCategory.TRADING,
            severity=EventSeverity.CRITICAL,
            message="bought",
        )
        await bus.emit(event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_category_filter(self):
        bus = EventBus()
        received = []

        class AIOnly(EventListener):
            categories = {EventCategory.AI}

            async def on_event(self, event):
                received.append(event)

        bus.register(AIOnly())
        await bus.emit(token_detected_event("A", "X", "X", 1.0, 5.0))  # DETECTION category
        assert len(received) == 0

        ai_evt = TradeEvent(
            event_type="AI_DECISION",
            category=EventCategory.AI,
            message="decided",
        )
        await bus.emit(ai_evt)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_failing_listener_does_not_block_others(self):
        bus = EventBus()
        received = []

        class Crasher(EventListener):
            async def on_event(self, event):
                raise RuntimeError("boom")

        class Stable(EventListener):
            async def on_event(self, event):
                received.append(event)

        bus.register(Crasher())
        bus.register(Stable())
        await bus.emit(token_detected_event("A", "X", "X", 1.0, 5.0))
        assert len(received) == 1
        assert bus.get_stats()["dispatch_errors"] == 1

    @pytest.mark.asyncio
    async def test_unregister_stops_delivery(self):
        bus = EventBus()
        received = []

        class Collector(EventListener):
            async def on_event(self, event):
                received.append(event)

        listener = Collector()
        bus.register(listener)
        bus.unregister(listener)
        await bus.emit(token_detected_event("A", "X", "X", 1.0, 5.0))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_event_count_tracked(self):
        bus = EventBus()

        class Noop(EventListener):
            async def on_event(self, event):
                pass

        bus.register(Noop())
        for _ in range(5):
            await bus.emit(token_detected_event("A", "X", "X", 1.0, 5.0))
        assert bus.get_stats()["events_emitted"] == 5

    @pytest.mark.asyncio
    async def test_shutdown_calls_listener_shutdown(self):
        bus = EventBus()
        shutdown_called = []

        class ShutdownListener(EventListener):
            async def on_event(self, event):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        bus.register(ShutdownListener())
        await bus.shutdown()
        assert len(shutdown_called) == 1


# ═══════════════════════════════════════════════════════════════════
#  TRADING STRATEGIES
# ═══════════════════════════════════════════════════════════════════


class TestStrategyRegistry:
    def test_all_expected_strategies_registered(self):
        assert "sniper" in STRATEGY_REGISTRY
        assert "sniper_conservative" in STRATEGY_REGISTRY
        assert "sniper_degen" in STRATEGY_REGISTRY
        assert "graduation" in STRATEGY_REGISTRY

    def test_registry_returns_correct_classes(self):
        assert STRATEGY_REGISTRY["sniper"] is SniperStrategy
        assert STRATEGY_REGISTRY["graduation"] is GraduationStrategy


class TestSniperStrategy:
    @pytest.fixture
    def sniper(self, config):
        return SniperStrategy(config)

    def test_strategy_id_and_name(self, sniper):
        assert sniper.strategy_id == "sniper"
        assert sniper.display_name == "Launch Sniper"

    @pytest.mark.asyncio
    async def test_should_evaluate_fresh_token(self, sniper):
        curve = MagicMock()
        curve.complete = False
        curve.get_migration_progress.return_value = 10.0
        token_data = {"bonding_curve_state": curve}
        assert await sniper.should_evaluate(token_data) is True

    @pytest.mark.asyncio
    async def test_skips_migrated_token(self, sniper):
        curve = MagicMock()
        curve.complete = True
        token_data = {"bonding_curve_state": curve}
        assert await sniper.should_evaluate(token_data) is False

    @pytest.mark.asyncio
    async def test_skips_token_over_50pct_migrated(self, sniper):
        curve = MagicMock()
        curve.complete = False
        curve.get_migration_progress.return_value = 60.0
        token_data = {"bonding_curve_state": curve}
        assert await sniper.should_evaluate(token_data) is False

    def test_get_ai_context_contains_strategy_name(self, sniper):
        ctx = sniper.get_ai_context()
        assert "LAUNCH SNIPER" in ctx

    def test_get_trade_params_matches_config(self, config, sniper):
        params = sniper.get_trade_params()
        assert params.buy_amount_sol == config.buy_amount_sol
        assert params.stop_loss_pct == config.stop_loss_pct

    def test_record_spend_and_close(self, sniper):
        sniper.record_spend(0.1)
        assert sniper.state.positions_open == 1
        assert sniper.state.sol_spent_today == pytest.approx(0.1)
        sniper.record_close(pnl_pct=50.0)
        assert sniper.state.positions_open == 0
        assert sniper.state.wins_today == 1

    def test_can_open_position_within_limits(self, sniper):
        assert sniper.can_open_position() is True

    def test_cannot_open_position_when_paused(self, sniper):
        sniper.pause()
        assert sniper.can_open_position() is False

    def test_cannot_open_position_when_at_limit(self, sniper):
        for _ in range(sniper.max_concurrent_positions):
            sniper.record_spend(0.1)
        assert sniper.can_open_position() is False

    def test_get_status_structure(self, sniper):
        status = sniper.get_status()
        assert status["strategy_id"] == "sniper"
        assert "budget_sol" in status
        assert "positions_open" in status
        assert "win_rate_today" in status


class TestConservativeSniperStrategy:
    @pytest.mark.asyncio
    async def test_requires_higher_liquidity(self):
        strat = ConservativeSniperStrategy(BotConfig())
        curve = MagicMock()
        curve.complete = False
        curve.get_migration_progress.return_value = 5.0
        # Below 5.0 SOL minimum
        token_data = {"bonding_curve_state": curve, "initial_liquidity_sol": 3.0}
        assert await strat.should_evaluate(token_data) is False

    @pytest.mark.asyncio
    async def test_passes_with_sufficient_liquidity(self):
        strat = ConservativeSniperStrategy(BotConfig())
        curve = MagicMock()
        curve.complete = False
        curve.get_migration_progress.return_value = 5.0
        token_data = {"bonding_curve_state": curve, "initial_liquidity_sol": 6.0}
        assert await strat.should_evaluate(token_data) is True


class TestGraduationStrategy:
    @pytest.fixture
    def grad(self, config):
        return GraduationStrategy(config)

    @pytest.mark.asyncio
    async def test_accepts_token_in_sweet_spot(self, grad):
        curve = MagicMock()
        curve.complete = False
        curve.get_migration_progress.return_value = 70.0
        token_data = {"bonding_curve_state": curve, "initial_liquidity_sol": 10.0}
        assert await grad.should_evaluate(token_data) is True

    @pytest.mark.asyncio
    async def test_rejects_token_below_50pct(self, grad):
        curve = MagicMock()
        curve.complete = False
        curve.get_migration_progress.return_value = 30.0
        token_data = {"bonding_curve_state": curve, "initial_liquidity_sol": 10.0}
        assert await grad.should_evaluate(token_data) is False

    @pytest.mark.asyncio
    async def test_rejects_token_above_90pct(self, grad):
        curve = MagicMock()
        curve.complete = False
        curve.get_migration_progress.return_value = 95.0
        token_data = {"bonding_curve_state": curve, "initial_liquidity_sol": 10.0}
        assert await grad.should_evaluate(token_data) is False

    @pytest.mark.asyncio
    async def test_rejects_migrated_token(self, grad):
        curve = MagicMock()
        curve.complete = True
        token_data = {"bonding_curve_state": curve, "initial_liquidity_sol": 10.0}
        assert await grad.should_evaluate(token_data) is False

    @pytest.mark.asyncio
    async def test_rejects_low_liquidity(self, grad):
        curve = MagicMock()
        curve.complete = False
        curve.get_migration_progress.return_value = 70.0
        token_data = {"bonding_curve_state": curve, "initial_liquidity_sol": 2.0}
        assert await grad.should_evaluate(token_data) is False

    def test_ai_context_mentions_graduation(self, grad):
        ctx = grad.get_ai_context()
        assert "GRADUATION" in ctx

    def test_take_profit_higher_than_sniper(self, config):
        sniper = SniperStrategy(config)
        grad = GraduationStrategy(config)
        assert grad.get_trade_params().take_profit_pct > sniper.get_trade_params().take_profit_pct


# ═══════════════════════════════════════════════════════════════════
#  AI HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════


class TestAIHealthMonitor:
    @pytest.fixture
    def monitor(self):
        cfg = HealthMonitorConfig(
            confidence_window=30,
            confidence_cluster_min_samples=5,
            confidence_stddev_floor=0.06,
            reasoning_collapse_min_samples=5,
            reasoning_similarity_threshold=0.80,
            skip_streak_threshold=5,
            response_time_min_samples=5,
            response_time_drift_factor=2.0,
            loss_cascade_threshold=3,
            trade_outcome_window=8,   # Small window so decay shows quickly
            win_rate_min_trades=4,
            win_rate_decay_threshold=0.15,
            alert_cooldown_seconds=300.0,  # Default; _overall_status looks back 600s
        )
        return AIHealthMonitor(config=cfg)

    def _make_ai_event(self, confidence=0.65, decision="BUY", reasoning="looks good", elapsed_ms=500.0, strategy_id=None):
        return TradeEvent(
            event_type="AI_DECISION",
            category=EventCategory.AI,
            strategy_id=strategy_id,
            data={
                "confidence": confidence,
                "decision": decision,
                "reasoning": reasoning,
                "elapsed_ms": elapsed_ms,
            },
            message="",
        )

    def _make_sell_event(self, pnl_pct: float, strategy_id=None):
        return TradeEvent(
            event_type="SELL_EXECUTED",
            category=EventCategory.TRADING,
            strategy_id=strategy_id,
            data={"pnl_pct": pnl_pct},
            message="",
        )

    @pytest.mark.asyncio
    async def test_no_alerts_on_healthy_data(self, monitor):
        # Feed varied confidences and unique reasoning
        for i in range(10):
            await monitor.on_event(self._make_ai_event(
                confidence=0.4 + i * 0.06,
                reasoning=f"unique reasoning {i} about this specific token",
            ))
        assert monitor._alerts == []

    @pytest.mark.asyncio
    async def test_confidence_clustering_detected(self, monitor):
        # All confidence scores very close to each other (low stddev)
        for _ in range(10):
            await monitor.on_event(self._make_ai_event(confidence=0.61))
        alerts = [a for a in monitor._alerts if a.drift_type == DriftType.CONFIDENCE_CLUSTERING]
        assert len(alerts) >= 1

    @pytest.mark.asyncio
    async def test_reasoning_collapse_detected(self, monitor):
        # All reasoning identical
        for _ in range(10):
            await monitor.on_event(self._make_ai_event(reasoning="template response"))
        alerts = [a for a in monitor._alerts if a.drift_type == DriftType.REASONING_COLLAPSE]
        assert len(alerts) >= 1

    @pytest.mark.asyncio
    async def test_skip_streak_detected(self, monitor):
        # Lots of tokens + lots of skips
        for _ in range(6):
            await monitor.on_event(TradeEvent(
                event_type="TOKEN_DETECTED",
                category=EventCategory.DETECTION,
                message="",
            ))
        for _ in range(6):
            await monitor.on_event(self._make_ai_event(decision="SKIP", confidence=0.3))
        alerts = [a for a in monitor._alerts if a.drift_type == DriftType.SKIP_STREAK]
        assert len(alerts) >= 1

    @pytest.mark.asyncio
    async def test_skip_streak_not_flagged_without_token_flow(self, monitor):
        # Skips without any token detection events
        for _ in range(10):
            await monitor.on_event(self._make_ai_event(decision="SKIP"))
        alerts = [a for a in monitor._alerts if a.drift_type == DriftType.SKIP_STREAK]
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_skip_streak_resets_on_buy(self, monitor):
        # Accumulate skips
        for _ in range(4):
            await monitor.on_event(TradeEvent(
                event_type="TOKEN_DETECTED", category=EventCategory.DETECTION, message=""
            ))
            await monitor.on_event(self._make_ai_event(decision="SKIP"))

        # Buy resets the streak
        await monitor.on_event(TradeEvent(
            event_type="BUY_EXECUTED", category=EventCategory.TRADING, message="",
            data={},
        ))
        state = monitor._get_state(None)
        assert state.consecutive_skips == 0
        assert state.tokens_seen_since_last_buy == 0

    @pytest.mark.asyncio
    async def test_loss_cascade_detected(self, monitor):
        for _ in range(4):
            await monitor.on_event(self._make_sell_event(pnl_pct=-25.0))
        alerts = [a for a in monitor._alerts if a.drift_type == DriftType.LOSS_CASCADE]
        assert len(alerts) >= 1
        assert alerts[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_loss_cascade_resets_on_win(self, monitor):
        for _ in range(2):
            await monitor.on_event(self._make_sell_event(pnl_pct=-25.0))
        await monitor.on_event(self._make_sell_event(pnl_pct=50.0))
        state = monitor._get_state(None)
        assert state.consecutive_losses == 0

    @pytest.mark.asyncio
    async def test_response_time_drift_detected(self, monitor):
        # Build baseline with fast responses
        for _ in range(6):
            await monitor.on_event(self._make_ai_event(elapsed_ms=300.0))

        # Then drift to very slow responses
        for _ in range(6):
            await monitor.on_event(self._make_ai_event(elapsed_ms=1500.0))

        alerts = [a for a in monitor._alerts if a.drift_type == DriftType.RESPONSE_TIME_DRIFT]
        assert len(alerts) >= 1

    @pytest.mark.asyncio
    async def test_win_rate_decay_detected(self, monitor):
        # Build strong baseline (8 wins → session win_rate=1.0, rolling full of wins)
        for _ in range(8):
            await monitor.on_event(self._make_sell_event(pnl_pct=50.0))

        # Tank the rolling window (5 losses → rolling has 3 wins + 5 losses = 37.5%,
        # session still 8/13 = 61.5%, decay = 0.24 > threshold 0.15)
        for _ in range(5):
            await monitor.on_event(self._make_sell_event(pnl_pct=-30.0))

        alerts = [a for a in monitor._alerts if a.drift_type == DriftType.WIN_RATE_DECAY]
        assert len(alerts) >= 1

    @pytest.mark.asyncio
    async def test_per_strategy_state_independent(self, monitor):
        # Strategy A has loss cascade
        for _ in range(4):
            await monitor.on_event(self._make_sell_event(-30.0, strategy_id="A"))

        # Strategy B is clean
        for _ in range(4):
            await monitor.on_event(self._make_sell_event(50.0, strategy_id="B"))

        state_a = monitor._get_state("A")
        state_b = monitor._get_state("B")
        assert state_a.consecutive_losses >= 3
        assert state_b.consecutive_losses == 0

    @pytest.mark.asyncio
    async def test_alert_cooldown_respected(self, monitor):
        # Fixture uses 300s cooldown — second identical alert should be suppressed
        # Trip confidence clustering
        for _ in range(20):
            await monitor.on_event(self._make_ai_event(confidence=0.61))
        first_count = len([a for a in monitor._alerts if a.drift_type == DriftType.CONFIDENCE_CLUSTERING])
        # Try to trip it again immediately
        for _ in range(20):
            await monitor.on_event(self._make_ai_event(confidence=0.61))
        second_count = len([a for a in monitor._alerts if a.drift_type == DriftType.CONFIDENCE_CLUSTERING])
        assert first_count == second_count  # No new alert during cooldown

    @pytest.mark.asyncio
    async def test_get_health_report_structure(self, monitor):
        report = monitor.get_health_report()
        assert "status" in report
        assert "uptime_seconds" in report
        assert "total_alerts" in report
        assert "recent_alerts" in report
        assert report["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_status_degrades_on_warning(self, monitor):
        for _ in range(10):
            await monitor.on_event(self._make_ai_event(confidence=0.61))
        report = monitor.get_health_report()
        assert report["status"] in ("degraded", "critical")

    @pytest.mark.asyncio
    async def test_health_status_critical_on_loss_cascade(self, monitor):
        for _ in range(4):
            await monitor.on_event(self._make_sell_event(-50.0))
        report = monitor.get_health_report()
        assert report["status"] == "critical"

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, monitor):
        for _ in range(4):
            await monitor.on_event(self._make_sell_event(-30.0))
        assert len(monitor._alerts) > 0
        monitor.reset()
        assert len(monitor._alerts) == 0

    @pytest.mark.asyncio
    async def test_emits_event_on_bus_when_alert_fires(self, monitor):
        bus = EventBus()
        monitor._bus = bus
        emitted = []

        class Collector(EventListener):
            min_severity = EventSeverity.DEBUG

            async def on_event(self, event):
                emitted.append(event)

        bus.register(Collector())

        for _ in range(4):
            await monitor.on_event(self._make_sell_event(-50.0))

        health_warnings = [e for e in emitted if e.event_type == "AI_HEALTH_WARNING"]
        assert len(health_warnings) >= 1
        assert health_warnings[0].data["drift_type"] == DriftType.LOSS_CASCADE.value
