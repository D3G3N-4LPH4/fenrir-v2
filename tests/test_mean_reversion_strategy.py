#!/usr/bin/env python3
"""
FENRIR - Mean Reversion strategy tests (Phase 4)

Registry wiring, ABC conformance, the cheap should_evaluate pre-filter, and the
MarketData-gated evaluate_token / reversion_score / build_ai_context machinery.
Each entry gate has a boundary reject case (including both the oversold trigger
and the collapse floor, and the falling-knife guard). No network.
"""

from __future__ import annotations

from typing import Any

import pytest

from fenrir.config import BotConfig
from fenrir.filters import MarketData
from fenrir.strategies import (
    DEFAULT_DISABLED_STRATEGIES,
    STRATEGY_REGISTRY,
    MeanReversionStrategy,
    TradeParams,
    TradingStrategy,
    get_strategy_class,
    is_enabled_by_default,
    list_strategies,
)
from fenrir.strategies.mean_reversion import MeanReversionSignal

TOKEN = "So11111111111111111111111111111111111111112"


@pytest.fixture
def cfg() -> BotConfig:
    return BotConfig()


def _md(**over: object) -> MarketData:
    """A MarketData snapshot that PASSES every mean-reversion gate by default.

    Oversold on 1h (-25%) but stabilizing on 5m (+1%), 24h only mildly down,
    buyers returning (55/45 = 0.55 pressure), ample volume + liquidity.
    """
    base: dict[str, Any] = dict(
        token_address=TOKEN,
        pair_address="PAIR",
        dex_id="raydium",
        age_minutes=180.0,
        market_cap_usd=500_000.0,
        price_usd=0.001,
        liquidity_usd=100_000.0,
        volume_1h_usd=200_000.0,
        txns_5m_buys=55,
        txns_5m_sells=45,
        price_change_5m_pct=1.0,
        price_change_1h_pct=-25.0,
        price_change_24h_pct=-10.0,
    )
    base.update(over)
    return MarketData(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered(self) -> None:
        assert "mean_reversion" in STRATEGY_REGISTRY
        cls = get_strategy_class("mean_reversion")
        assert cls is MeanReversionStrategy
        assert issubclass(cls, TradingStrategy)
        assert "mean_reversion" in list_strategies()

    def test_off_by_default(self) -> None:
        assert "mean_reversion" in DEFAULT_DISABLED_STRATEGIES
        assert is_enabled_by_default("mean_reversion") is False

    def test_registry_id_matches_class_attr(self) -> None:
        assert MeanReversionStrategy.strategy_id == "mean_reversion"


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------


class TestABCConformance:
    def test_identity(self, cfg: BotConfig) -> None:
        strat = MeanReversionStrategy(cfg)
        assert strat.strategy_id == "mean_reversion"
        assert strat.display_name
        assert strat.description
        assert strat.budget_sol > 0
        assert strat.max_concurrent_positions >= 1
        assert strat.uses_market_data is True

    def test_ai_context_static(self, cfg: BotConfig) -> None:
        ctx = MeanReversionStrategy(cfg).get_ai_context()
        assert "STRATEGY CONTEXT: MEAN REVERSION" in ctx

    def test_trade_params(self, cfg: BotConfig) -> None:
        params = MeanReversionStrategy(cfg).get_trade_params()
        assert isinstance(params, TradeParams)
        assert params.take_profit_pct == 30.0
        assert params.trailing_stop_pct == 10.0
        assert params.stop_loss_pct == 12.0
        assert params.max_position_age_minutes == 120
        assert 0.0 < params.ai_min_confidence <= 1.0


# ---------------------------------------------------------------------------
# should_evaluate (cheap token_data pre-filter)
# ---------------------------------------------------------------------------


class TestShouldEvaluate:
    async def test_passthrough(self, cfg: BotConfig) -> None:
        assert await MeanReversionStrategy(cfg).should_evaluate({"token_address": TOKEN}) is True


# ---------------------------------------------------------------------------
# evaluate_token — entry gates
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_pass(self, cfg: BotConfig) -> None:
        sig = MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md())
        assert sig is not None
        assert isinstance(sig, MeanReversionSignal)
        assert sig.metadata["strategy"] == "mean_reversion"
        assert sig.buy_pressure_5m == pytest.approx(0.55, abs=0.01)
        assert 0.0 < sig.reversion_score <= 1.0

    def test_none_market_data(self, cfg: BotConfig) -> None:
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, None) is None

    def test_inactive_strategy(self, cfg: BotConfig) -> None:
        strat = MeanReversionStrategy(cfg)
        strat.deactivate()
        assert strat.evaluate_token({"token_address": TOKEN}, _md()) is None

    @pytest.mark.parametrize("age", [30.0, 3000.0])
    def test_reject_age_outside_window(self, cfg: BotConfig, age: float) -> None:
        md = _md(age_minutes=age)
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_not_oversold(self, cfg: BotConfig) -> None:
        # 1h only -5% → not a real dislocation (want <= -15%).
        md = _md(price_change_1h_pct=-5.0)
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_collapse_floor(self, cfg: BotConfig) -> None:
        # 1h -70% is below the collapse floor (-60%): a dead token, not a dislocation.
        md = _md(price_change_1h_pct=-70.0)
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_dead_token_24h(self, cfg: BotConfig) -> None:
        md = _md(price_change_24h_pct=-90.0)
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_falling_knife(self, cfg: BotConfig) -> None:
        # 5m still crashing (-10%) → not stabilizing.
        md = _md(price_change_5m_pct=-10.0)
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_weak_buy_pressure(self, cfg: BotConfig) -> None:
        md = _md(txns_5m_buys=30, txns_5m_sells=70)  # 0.30 < 0.45
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_low_1h_volume(self, cfg: BotConfig) -> None:
        md = _md(volume_1h_usd=10_000.0)
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

    def test_reject_thin_liquidity(self, cfg: BotConfig) -> None:
        md = _md(liquidity_usd=10_000.0)
        assert MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None


# ---------------------------------------------------------------------------
# reversion_score + build_ai_context
# ---------------------------------------------------------------------------


class TestScoreAndContext:
    def test_score_bounds(self, cfg: BotConfig) -> None:
        sig = MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md())
        assert sig is not None
        assert 0.0 <= sig.reversion_score <= 1.0

    def test_deeper_dislocation_scores_higher(self, cfg: BotConfig) -> None:
        strat = MeanReversionStrategy(cfg)
        shallow = strat.evaluate_token({"token_address": TOKEN}, _md(price_change_1h_pct=-16.0))
        deep = strat.evaluate_token({"token_address": TOKEN}, _md(price_change_1h_pct=-45.0))
        assert shallow is not None and deep is not None
        assert deep.reversion_score > shallow.reversion_score

    def test_build_ai_context_has_key_fields(self, cfg: BotConfig) -> None:
        sig = MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md())
        assert sig is not None
        ctx = MeanReversionStrategy(cfg).build_ai_context(sig)
        assert "MEAN REVERSION (OVERSOLD BOUNCE) EVALUATION" in ctx
        assert "dislocation" in ctx
        assert "Buy pressure" in ctx
        assert "Reversion score" in ctx
        assert "EXIT PLAN" in ctx
