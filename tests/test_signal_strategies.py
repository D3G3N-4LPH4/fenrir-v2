#!/usr/bin/env python3
"""
FENRIR - Signal Strategy Test Suite

Covers the four signal-oriented strategies (migration_snipe, reversal,
volume_anomaly, narrative_tracker) after their refactor onto the
TradingStrategy ABC: registry wiring, ABC conformance, and the retained
evaluate_token / build_ai_context machinery gated on MarketData.

Run with: pytest tests/test_signal_strategies.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fenrir.config import BotConfig
from fenrir.filters import MarketData
from fenrir.strategies import (
    DEFAULT_DISABLED_STRATEGIES,
    STRATEGY_REGISTRY,
    MigrationSniperStrategy,
    NarrativeTrackerStrategy,
    ReversalStrategy,
    TradeParams,
    TradingStrategy,
    VolumeAnomalyStrategy,
    get_strategy_class,
    is_enabled_by_default,
    list_strategies,
)
from fenrir.strategies.narrative import detect_narrative

TOKEN = "So11111111111111111111111111111111111111112"

NEW_IDS = ["migration_snipe", "reversal", "volume_anomaly", "narrative_tracker"]

# Concrete classes (typed as ``type`` so mypy allows the ``cls(cfg)`` call — the
# ABC's __init__ takes no args, but every concrete strategy takes a BotConfig).
NEW_CLASSES: list[type] = [
    MigrationSniperStrategy,
    ReversalStrategy,
    VolumeAnomalyStrategy,
    NarrativeTrackerStrategy,
]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    @pytest.mark.parametrize("sid", NEW_IDS)
    def test_registered(self, sid: str) -> None:
        assert sid in STRATEGY_REGISTRY
        cls = get_strategy_class(sid)
        assert cls is not None
        assert issubclass(cls, TradingStrategy)
        assert sid in list_strategies()

    @pytest.mark.parametrize("sid", NEW_IDS)
    def test_off_by_default(self, sid: str) -> None:
        assert sid in DEFAULT_DISABLED_STRATEGIES
        assert is_enabled_by_default(sid) is False

    def test_existing_strategies_still_default_on(self) -> None:
        assert is_enabled_by_default("sniper") is True
        assert is_enabled_by_default("graduation") is True

    def test_registry_id_matches_class_attr(self) -> None:
        for sid in NEW_IDS:
            cls = get_strategy_class(sid)
            assert cls is not None
            assert cls.strategy_id == sid


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> BotConfig:
    return BotConfig()


class TestABCConformance:
    @pytest.mark.parametrize("cls", NEW_CLASSES)
    def test_instantiates_and_exposes_identity(self, cls: type, cfg: BotConfig) -> None:
        strat = cls(cfg)
        assert strat.strategy_id in NEW_IDS
        assert strat.display_name
        assert strat.description
        assert strat.budget_sol > 0
        assert strat.max_concurrent_positions >= 1

    @pytest.mark.parametrize("cls", NEW_CLASSES)
    def test_get_ai_context_static(self, cls: type, cfg: BotConfig) -> None:
        ctx = cls(cfg).get_ai_context()
        assert isinstance(ctx, str)
        assert "STRATEGY CONTEXT" in ctx

    @pytest.mark.parametrize("cls", NEW_CLASSES)
    def test_get_trade_params(self, cls: type, cfg: BotConfig) -> None:
        params = cls(cfg).get_trade_params()
        assert isinstance(params, TradeParams)
        assert params.stop_loss_pct > 0
        assert params.take_profit_pct > 0
        assert 0.0 < params.ai_min_confidence <= 1.0
        assert params.max_position_age_minutes > 0

    def test_trade_param_values(self, cfg: BotConfig) -> None:
        mig = MigrationSniperStrategy(cfg).get_trade_params()
        assert mig.stop_loss_pct == 35.0
        assert mig.take_profit_pct == 100.0
        assert mig.ai_min_confidence == 0.65
        assert mig.max_position_age_minutes == 60

        vol = VolumeAnomalyStrategy(cfg).get_trade_params()
        assert vol.stop_loss_pct == 15.0
        assert vol.take_profit_pct == 40.0
        assert vol.trailing_stop_pct == 12.0
        assert vol.max_position_age_minutes == 240

        nar = NarrativeTrackerStrategy(cfg).get_trade_params()
        assert nar.take_profit_pct == 200.0
        assert nar.max_position_age_minutes == 720


# ---------------------------------------------------------------------------
# should_evaluate (cheap token_data pre-filter)
# ---------------------------------------------------------------------------


class TestShouldEvaluate:
    async def test_migration_skips_incomplete_curve(self, cfg: BotConfig) -> None:
        strat = MigrationSniperStrategy(cfg)
        td = {"token_address": TOKEN, "bonding_curve_state": SimpleNamespace(complete=False)}
        assert await strat.should_evaluate(td) is False

    async def test_migration_allows_complete_curve(self, cfg: BotConfig) -> None:
        strat = MigrationSniperStrategy(cfg)
        td = {"token_address": TOKEN, "bonding_curve_state": SimpleNamespace(complete=True)}
        assert await strat.should_evaluate(td) is True

    async def test_migration_allows_when_no_curve(self, cfg: BotConfig) -> None:
        strat = MigrationSniperStrategy(cfg)
        assert await strat.should_evaluate({"token_address": TOKEN}) is True

    async def test_narrative_requires_keyword_match(self, cfg: BotConfig) -> None:
        strat = NarrativeTrackerStrategy(cfg)
        assert await strat.should_evaluate({"name": "Claude AI", "symbol": "AGI"}) is True
        assert await strat.should_evaluate({"name": "Random Thing", "symbol": "RND"}) is False

    @pytest.mark.parametrize("cls", [ReversalStrategy, VolumeAnomalyStrategy])
    async def test_market_gated_strategies_passthrough(self, cls: type, cfg: BotConfig) -> None:
        # These gate on market data later; the cheap pre-filter admits everything.
        assert await cls(cfg).should_evaluate({"token_address": TOKEN}) is True


# ---------------------------------------------------------------------------
# evaluate_token — MigrationSniper
# ---------------------------------------------------------------------------


def _migration_md(**over: object) -> MarketData:
    base: dict[str, Any] = dict(
        token_address=TOKEN,
        pair_address="PAIR",
        dex_id="raydium",
        age_minutes=1.0,
        liquidity_usd=50_000.0,
        market_cap_usd=80_000.0,
        volume_5m_usd=20_000.0,
        txns_5m_buys=80,
        txns_5m_sells=40,
    )
    base.update(over)
    return MarketData(**base)  # type: ignore[arg-type]


class TestMigrationEvaluate:
    def test_pass(self, cfg: BotConfig) -> None:
        sig = MigrationSniperStrategy(cfg).evaluate_token({"token_address": TOKEN}, _migration_md())
        assert sig is not None
        assert 0.0 <= sig.urgency_score <= 1.0
        assert sig.metadata["strategy"] == "migration_snipe"

    def test_reject_pool_too_old(self, cfg: BotConfig) -> None:
        md = _migration_md(age_minutes=10.0)
        assert MigrationSniperStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_accepts_pumpswap(self, cfg: BotConfig) -> None:
        # Modern graduations land on PumpSwap — must be accepted.
        md = _migration_md(dex_id="pumpswap")
        sig = MigrationSniperStrategy(cfg).evaluate_token({"token_address": TOKEN}, md)
        assert sig is not None

    def test_reject_unsupported_dex(self, cfg: BotConfig) -> None:
        # pre-migration bonding curve / other AMMs are not migration targets.
        for dex in ("pumpfun", "orca", ""):
            md = _migration_md(dex_id=dex)
            assert MigrationSniperStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_none_market_data(self, cfg: BotConfig) -> None:
        assert MigrationSniperStrategy(cfg).evaluate_token({"token_address": TOKEN}, None) is None

    def test_inactive_returns_none(self, cfg: BotConfig) -> None:
        strat = MigrationSniperStrategy(cfg)
        strat.deactivate()
        assert strat.evaluate_token({"token_address": TOKEN}, _migration_md()) is None

    def test_build_ai_context_with_security(self, cfg: BotConfig) -> None:
        strat = MigrationSniperStrategy(cfg)
        sig = strat.evaluate_token({"token_address": TOKEN}, _migration_md())
        assert sig is not None
        sec = SimpleNamespace(details={"top10_holder_pct": 12.0, "lp_burned_pct": 99.0})
        ctx = strat.build_ai_context(sig, security_result=sec)
        assert "MIGRATION SNIPE" in ctx
        assert "Top-10 holders" in ctx
        assert "LP burned" in ctx


# ---------------------------------------------------------------------------
# evaluate_token — Reversal
# ---------------------------------------------------------------------------


def _reversal_md(**over: object) -> MarketData:
    base: dict[str, Any] = dict(
        token_address=TOKEN,
        pair_address="PAIR",
        age_minutes=60.0,
        liquidity_usd=8_000.0,
        price_change_1h_pct=-70.0,  # 70% drawdown proxy
        price_change_5m_pct=-5.0,  # consolidating
        txns_1h_buys=40,
        txns_1h_sells=20,
    )
    base.update(over)
    return MarketData(**base)  # type: ignore[arg-type]


class TestReversalEvaluate:
    def test_pass(self, cfg: BotConfig) -> None:
        sig = ReversalStrategy(cfg).evaluate_token({"token_address": TOKEN}, _reversal_md())
        assert sig is not None
        assert sig.estimated_drawdown_pct == 70.0
        assert 0.0 <= sig.recovery_strength <= 1.0

    def test_reject_drawdown_too_shallow(self, cfg: BotConfig) -> None:
        md = _reversal_md(price_change_1h_pct=-30.0)
        assert ReversalStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_outside_age_window(self, cfg: BotConfig) -> None:
        md = _reversal_md(age_minutes=5.0)
        assert ReversalStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_still_volatile(self, cfg: BotConfig) -> None:
        md = _reversal_md(price_change_5m_pct=-40.0)
        assert ReversalStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_build_ai_context(self, cfg: BotConfig) -> None:
        strat = ReversalStrategy(cfg)
        sig = strat.evaluate_token({"token_address": TOKEN}, _reversal_md())
        assert sig is not None
        ctx = strat.build_ai_context(sig)
        assert "REVERSAL" in ctx
        assert "→" in ctx  # incremental TP ladder rendered


# ---------------------------------------------------------------------------
# evaluate_token — VolumeAnomaly
# ---------------------------------------------------------------------------


def _volume_md(**over: object) -> MarketData:
    base: dict[str, Any] = dict(
        token_address=TOKEN,
        pair_address="PAIR",
        age_minutes=600.0,  # 10h
        market_cap_usd=1_000_000.0,
        volume_24h_usd=2_000_000.0,  # 2.0x ratio
        liquidity_usd=150_000.0,
        price_change_5m_pct=-3.0,  # dip entry
    )
    base.update(over)
    return MarketData(**base)  # type: ignore[arg-type]


class TestVolumeAnomalyEvaluate:
    def test_pass(self, cfg: BotConfig) -> None:
        sig = VolumeAnomalyStrategy(cfg).evaluate_token({"token_address": TOKEN}, _volume_md())
        assert sig is not None
        assert sig.volume_to_mcap_ratio == pytest.approx(2.0)
        assert 0.0 <= sig.anomaly_score <= 1.0

    def test_reject_mcap_out_of_range(self, cfg: BotConfig) -> None:
        md = _volume_md(market_cap_usd=100_000.0)  # below $500k floor
        assert VolumeAnomalyStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_ratio_too_low(self, cfg: BotConfig) -> None:
        md = _volume_md(volume_24h_usd=600_000.0)  # 0.6x ratio
        assert VolumeAnomalyStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_not_a_dip(self, cfg: BotConfig) -> None:
        md = _volume_md(price_change_5m_pct=10.0)  # pumping, not a dip
        assert VolumeAnomalyStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_outside_age_window(self, cfg: BotConfig) -> None:
        md = _volume_md(age_minutes=60.0)  # 1h < 6h floor
        assert VolumeAnomalyStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None


# ---------------------------------------------------------------------------
# evaluate_token — NarrativeTracker
# ---------------------------------------------------------------------------


def _narrative_md(**over: object) -> MarketData:
    base: dict[str, Any] = dict(
        token_address=TOKEN,
        pair_address="PAIR",
        age_minutes=120.0,  # 2h
        market_cap_usd=100_000.0,
        liquidity_usd=30_000.0,
        volume_1h_usd=60_000.0,
        price_change_1h_pct=25.0,
        txns_1h_buys=50,
        txns_1h_sells=30,
    )
    base.update(over)
    return MarketData(**base)  # type: ignore[arg-type]


class TestNarrativeEvaluate:
    def test_pass(self, cfg: BotConfig) -> None:
        td = {"token_address": TOKEN, "name": "Claude AI Agent", "symbol": "AGI"}
        sig = NarrativeTrackerStrategy(cfg).evaluate_token(td, _narrative_md())
        assert sig is not None
        assert sig.narrative == "ai_agents"
        assert 0.0 <= sig.narrative_momentum_score <= 1.0

    def test_reject_no_narrative_match(self, cfg: BotConfig) -> None:
        td = {"token_address": TOKEN, "name": "Nondescript", "symbol": "ND"}
        assert NarrativeTrackerStrategy(cfg).evaluate_token(td, _narrative_md()) is None

    def test_reject_mcap_too_low(self, cfg: BotConfig) -> None:
        td = {"token_address": TOKEN, "name": "Doge Beta", "symbol": "DOGE"}
        md = _narrative_md(market_cap_usd=10_000.0)
        assert NarrativeTrackerStrategy(cfg).evaluate_token(td, md) is None

    def test_leader_pump_tracking(self, cfg: BotConfig) -> None:
        strat = NarrativeTrackerStrategy(cfg)
        assert strat._is_leader_pumping("ai_agents") is False
        strat.update_narrative_leader("ai_agents", {"price_change_1h_pct": 80.0})
        assert strat._is_leader_pumping("ai_agents") is True

    def test_build_ai_context(self, cfg: BotConfig) -> None:
        strat = NarrativeTrackerStrategy(cfg)
        td = {"token_address": TOKEN, "name": "Popcat", "symbol": "POP"}
        sig = strat.evaluate_token(td, _narrative_md())
        assert sig is not None
        ctx = strat.build_ai_context(sig)
        assert "NARRATIVE BETA PLAY" in ctx
        assert "CATS" in ctx


# ---------------------------------------------------------------------------
# detect_narrative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "symbol", "expected"),
    [
        ("Claude AI", "AGI", "ai_agents"),
        ("DogeCoin", "DOGE", "dog_breeds"),
        ("Popcat", "POP", "cats"),
        ("Trump 2024", "MAGA", "political"),
        ("Nondescript Token", "XYZ", None),
    ],
)
def test_detect_narrative(name: str, symbol: str, expected: str | None) -> None:
    assert detect_narrative(name, symbol) == expected
