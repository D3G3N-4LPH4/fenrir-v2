#!/usr/bin/env python3
"""
FENRIR - Momentum Continuation strategy tests (Phase 4)

Registry wiring, ABC conformance, the cheap should_evaluate pre-filter, and the
MarketData-gated evaluate_token / momentum_score / build_ai_context machinery.
Each entry gate has a boundary reject case. No network.
"""

from __future__ import annotations

from typing import Any

import pytest

from fenrir.config import BotConfig
from fenrir.filters import MarketData
from fenrir.strategies import (
    DEFAULT_DISABLED_STRATEGIES,
    STRATEGY_REGISTRY,
    MomentumStrategy,
    TradeParams,
    TradingStrategy,
    get_strategy_class,
    is_enabled_by_default,
    list_strategies,
)
from fenrir.strategies.momentum import MomentumSignal

TOKEN = "So11111111111111111111111111111111111111112"


@pytest.fixture
def cfg() -> BotConfig:
    return BotConfig()


def _md(**over: object) -> MarketData:
    """A MarketData snapshot that PASSES every momentum gate by default.

    accel = (volume_5m/5) / (volume_1h/60) = (30_000/5) / (200_000/60) = 6000/3333 ≈ 1.8x
    buy pressure = 70 / (70+30) = 0.70
    """
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


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered(self) -> None:
        assert "momentum" in STRATEGY_REGISTRY
        cls = get_strategy_class("momentum")
        assert cls is MomentumStrategy
        assert issubclass(cls, TradingStrategy)
        assert "momentum" in list_strategies()

    def test_off_by_default(self) -> None:
        assert "momentum" in DEFAULT_DISABLED_STRATEGIES
        assert is_enabled_by_default("momentum") is False

    def test_registry_id_matches_class_attr(self) -> None:
        assert MomentumStrategy.strategy_id == "momentum"


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------


class TestABCConformance:
    def test_identity(self, cfg: BotConfig) -> None:
        strat = MomentumStrategy(cfg)
        assert strat.strategy_id == "momentum"
        assert strat.display_name
        assert strat.description
        assert strat.budget_sol > 0
        assert strat.max_concurrent_positions >= 1
        assert strat.uses_market_data is True

    def test_ai_context_static(self, cfg: BotConfig) -> None:
        ctx = MomentumStrategy(cfg).get_ai_context()
        assert "STRATEGY CONTEXT: MOMENTUM CONTINUATION" in ctx

    def test_trade_params(self, cfg: BotConfig) -> None:
        params = MomentumStrategy(cfg).get_trade_params()
        assert isinstance(params, TradeParams)
        assert params.take_profit_pct == 60.0
        assert params.trailing_stop_pct == 15.0
        assert params.stop_loss_pct == 15.0
        assert params.max_position_age_minutes == 180
        assert 0.0 < params.ai_min_confidence <= 1.0


# ---------------------------------------------------------------------------
# should_evaluate (cheap token_data pre-filter)
# ---------------------------------------------------------------------------


class TestShouldEvaluate:
    async def test_passthrough(self, cfg: BotConfig) -> None:
        # Gated on market data later; the cheap pre-filter admits everything.
        assert await MomentumStrategy(cfg).should_evaluate({"token_address": TOKEN}) is True


# ---------------------------------------------------------------------------
# evaluate_token — entry gates
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_pass(self, cfg: BotConfig) -> None:
        sig = MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md())
        assert sig is not None
        assert isinstance(sig, MomentumSignal)
        assert sig.metadata["strategy"] == "momentum"
        assert sig.volume_acceleration == pytest.approx(1.8, abs=0.05)
        assert sig.buy_pressure_5m == pytest.approx(0.70, abs=0.01)
        assert 0.0 < sig.momentum_score <= 1.0

    def test_none_market_data(self, cfg: BotConfig) -> None:
        assert MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, None) is None

    def test_inactive_strategy(self, cfg: BotConfig) -> None:
        strat = MomentumStrategy(cfg)
        strat.deactivate()
        assert strat.evaluate_token({"token_address": TOKEN}, _md()) is None

    @pytest.mark.parametrize("age", [10.0, 800.0])
    def test_reject_age_outside_window(self, cfg: BotConfig, age: float) -> None:
        assert (
            MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md(age_minutes=age))
            is None
        )

    def test_reject_weak_1h_trend(self, cfg: BotConfig) -> None:
        assert (
            MomentumStrategy(cfg).evaluate_token(
                {"token_address": TOKEN}, _md(price_change_1h_pct=5.0)
            )
            is None
        )

    def test_reject_stalling_5m(self, cfg: BotConfig) -> None:
        assert (
            MomentumStrategy(cfg).evaluate_token(
                {"token_address": TOKEN}, _md(price_change_5m_pct=-1.0)
            )
            is None
        )

    def test_reject_parabolic_24h(self, cfg: BotConfig) -> None:
        assert (
            MomentumStrategy(cfg).evaluate_token(
                {"token_address": TOKEN}, _md(price_change_24h_pct=500.0)
            )
            is None
        )

    def test_reject_volume_fading(self, cfg: BotConfig) -> None:
        # recent pace (10_000/5=2000) < hourly pace (200_000/60≈3333) × 1.2 → accel ~0.6.
        assert (
            MomentumStrategy(cfg).evaluate_token(
                {"token_address": TOKEN}, _md(volume_5m_usd=10_000.0)
            )
            is None
        )

    def test_reject_low_1h_volume(self, cfg: BotConfig) -> None:
        # Keep acceleration high but drop 1h volume below the absolute floor.
        md = _md(volume_1h_usd=10_000.0)
        assert MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_weak_buy_pressure(self, cfg: BotConfig) -> None:
        md = _md(txns_5m_buys=40, txns_5m_sells=60)  # 0.40 < 0.55
        assert MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_thin_liquidity(self, cfg: BotConfig) -> None:
        assert (
            MomentumStrategy(cfg).evaluate_token(
                {"token_address": TOKEN}, _md(liquidity_usd=10_000.0)
            )
            is None
        )


# ---------------------------------------------------------------------------
# momentum_score + build_ai_context
# ---------------------------------------------------------------------------


class TestScoreAndContext:
    def test_score_bounds(self, cfg: BotConfig) -> None:
        sig = MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md())
        assert sig is not None
        assert 0.0 <= sig.momentum_score <= 1.0

    def test_stronger_trend_scores_higher(self, cfg: BotConfig) -> None:
        strat = MomentumStrategy(cfg)
        weak = strat.evaluate_token({"token_address": TOKEN}, _md(price_change_1h_pct=13.0))
        strong = strat.evaluate_token({"token_address": TOKEN}, _md(price_change_1h_pct=45.0))
        assert weak is not None and strong is not None
        assert strong.momentum_score > weak.momentum_score

    def test_build_ai_context_has_key_fields(self, cfg: BotConfig) -> None:
        sig = MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md())
        assert sig is not None
        ctx = MomentumStrategy(cfg).build_ai_context(sig)
        assert "MOMENTUM CONTINUATION EVALUATION" in ctx
        assert "Volume acceleration" in ctx
        assert "Buy pressure" in ctx
        assert "Momentum score" in ctx
        assert "EXIT PLAN" in ctx
