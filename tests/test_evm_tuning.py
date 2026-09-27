#!/usr/bin/env python3
"""
FENRIR - EVM strategy cadence-tuning tests (Phase 7)

EVM/L2 tokens live hours→days and carry less volume than Solana pump.fun launches, so the
EVM evaluator uses widened age windows + lower $ floors while keeping the ratio gates.
These pin the strategy params override and config.build_evm_strategies, and prove an aged
uptrend token that the Solana-tuned momentum rejects (too old) fires the EVM-tuned one.
No network.
"""

from __future__ import annotations

from typing import Any

from fenrir.config import BotConfig
from fenrir.filters import MarketData
from fenrir.strategies.momentum import MomentumConfig, MomentumStrategy

TOKEN = "0x32dae312abe8f6fdb782907b85edbc90d2e74b02"


def _md(age_minutes: float) -> MarketData:
    # An uptrend/accelerating/buyer-dominant snapshot at a given age.
    return MarketData(
        token_address=TOKEN,
        pair_address="P",
        dex_id="uniswap",
        age_minutes=age_minutes,
        market_cap_usd=359_000.0,
        price_usd=0.00036,
        liquidity_usd=30_000.0,  # below Solana 50k floor, above EVM 25k floor
        volume_5m_usd=6_000.0,
        volume_1h_usd=30_000.0,  # below Solana 50k floor, above EVM 20k floor
        txns_5m_buys=70,
        txns_5m_sells=30,
        price_change_5m_pct=2.0,
        price_change_1h_pct=25.0,
        price_change_24h_pct=150.0,
    )


class TestParamsOverride:
    def test_momentum_accepts_params(self) -> None:
        strat = MomentumStrategy(BotConfig(), MomentumConfig(max_age_minutes=99_999.0))
        assert strat.params.max_age_minutes == 99_999.0

    def test_default_params_unchanged(self) -> None:
        assert MomentumStrategy(BotConfig()).params.max_age_minutes == 720.0


class TestBuildEvmStrategies:
    def test_widened_defaults(self) -> None:
        strategies = {s.strategy_id: s for s in BotConfig().build_evm_strategies()}
        assert set(strategies) == {"momentum", "mean_reversion"}
        mom = strategies["momentum"]
        assert mom.params.max_age_minutes == 10_080.0
        assert mom.params.min_liquidity_usd == 25_000.0
        assert mom.params.min_volume_1h_usd == 20_000.0
        rev = strategies["mean_reversion"]
        assert rev.params.max_age_minutes == 20_160.0
        # Ratio gates are untouched (calibrated later from data, not guessed).
        assert mom.params.min_price_change_1h_pct == 12.0

    def test_env_override(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("EVM_MOMENTUM_MAX_AGE_MINUTES", "4000")
        monkeypatch.setenv("EVM_MIN_VOLUME_1H_USD", "5000")
        strategies = {s.strategy_id: s for s in BotConfig().build_evm_strategies()}
        assert strategies["momentum"].params.max_age_minutes == 4000.0
        assert strategies["momentum"].params.min_volume_1h_usd == 5000.0


class TestCadenceEffect:
    def test_aged_uptrend_fires_evm_not_solana(self) -> None:
        # A 3-day-old (4320 min) uptrend token on a small L2.
        md = _md(age_minutes=4320.0)
        cfg = BotConfig()

        # Solana-tuned momentum: rejects (age > 720 min AND liq/vol below 50k floors).
        assert MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, md) is None

        # EVM-tuned momentum: age within 7d, floors lowered → fires.
        evm_mom = next(s for s in cfg.build_evm_strategies() if s.strategy_id == "momentum")
        sig = evm_mom.evaluate_token({"token_address": TOKEN}, md)
        assert sig is not None
        assert sig.momentum_score > 0
