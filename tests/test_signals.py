#!/usr/bin/env python3
"""
FENRIR - Unified signal normalization tests (Phase 5)

Normalizes REAL signals produced by the actual strategies (via their evaluate_token on
passing MarketData) and a real ArbOpportunity onto the common Signal, plus the model's
clamp/direction semantics and the dispatcher. No network.
"""

from __future__ import annotations

from typing import Any

import pytest

from fenrir.config import BotConfig
from fenrir.filters import MarketData
from fenrir.signals import (
    Signal,
    SignalDirection,
    normalize_arbitrage,
    normalize_signal,
    normalize_strategy_signal,
)
from fenrir.strategies.mean_reversion import MeanReversionStrategy
from fenrir.strategies.momentum import MomentumStrategy
from fenrir.strategies.volume_anomaly import VolumeAnomalyStrategy
from fenrir.trading.arbitrage import ArbitrageDetector, VenueQuote

TOKEN = "So11111111111111111111111111111111111111112"


@pytest.fixture
def cfg() -> BotConfig:
    return BotConfig()


def _md(**over: Any) -> MarketData:
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


class TestModel:
    def test_strength_clamped(self) -> None:
        assert Signal("s", TOKEN, SignalDirection.LONG, strength=2.0).strength == 1.0
        assert Signal("s", TOKEN, SignalDirection.LONG, strength=-1.0).strength == 0.0

    def test_is_actionable(self) -> None:
        assert Signal("s", TOKEN, SignalDirection.LONG, 0.1).is_actionable is True
        assert Signal("s", TOKEN, SignalDirection.LONG, 0.0).is_actionable is False

    def test_to_dict(self) -> None:
        d = Signal("momentum", TOKEN, SignalDirection.LONG, 0.5, rationale="r").to_dict()
        assert d["source"] == "momentum"
        assert d["direction"] == "long"
        assert d["strength"] == 0.5


class TestStrategyNormalization:
    def test_momentum(self, cfg: BotConfig) -> None:
        sig = MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md())
        assert sig is not None
        norm = normalize_signal(sig)
        assert norm.source == "momentum"
        assert norm.direction is SignalDirection.LONG
        assert norm.strength == pytest.approx(sig.momentum_score)
        assert norm.token_address == TOKEN

    def test_mean_reversion(self, cfg: BotConfig) -> None:
        md = _md(price_change_1h_pct=-25.0, price_change_5m_pct=1.0, price_change_24h_pct=-10.0)
        sig = MeanReversionStrategy(cfg).evaluate_token({"token_address": TOKEN}, md)
        assert sig is not None
        norm = normalize_signal(sig)
        assert norm.source == "mean_reversion"
        assert norm.direction is SignalDirection.LONG
        assert norm.strength == pytest.approx(sig.reversion_score)

    def test_volume_anomaly(self, cfg: BotConfig) -> None:
        md = _md(
            age_minutes=600.0,
            market_cap_usd=1_000_000.0,
            volume_24h_usd=2_000_000.0,
            liquidity_usd=200_000.0,
            price_change_5m_pct=-2.0,
        )
        sig = VolumeAnomalyStrategy(cfg).evaluate_token({"token_address": TOKEN}, md)
        assert sig is not None
        norm = normalize_signal(sig)
        assert norm.source == "volume_anomaly"
        assert norm.strength == pytest.approx(sig.anomaly_score)

    def test_source_override(self, cfg: BotConfig) -> None:
        sig = MomentumStrategy(cfg).evaluate_token({"token_address": TOKEN}, _md())
        assert sig is not None
        norm = normalize_strategy_signal(sig, source="custom")
        assert norm.source == "custom"

    def test_missing_source_raises(self) -> None:
        from types import SimpleNamespace

        with pytest.raises(ValueError):
            normalize_strategy_signal(SimpleNamespace(metadata={}, token_address=TOKEN))

    def test_unmapped_strategy_falls_back_to_score_property(self) -> None:
        # A signal from an unmapped strategy still normalizes via its *_score property.
        from types import SimpleNamespace

        sig = SimpleNamespace(
            token_address=TOKEN,
            metadata={"strategy": "brand_new_strat"},
            conviction_score=0.42,
        )
        norm = normalize_strategy_signal(sig)
        assert norm.source == "brand_new_strat"
        assert norm.strength == pytest.approx(0.42)


class TestArbitrageNormalization:
    def test_arbitrage(self) -> None:
        opp = ArbitrageDetector().evaluate(
            TOKEN,
            [
                VenueQuote("raydium", price=1.00, liquidity_sol=100_000.0, fee_bps=10),
                VenueQuote("pumpswap", price=1.05, liquidity_sol=100_000.0, fee_bps=10),
            ],
            size_sol=1.0,
        )
        assert opp is not None
        norm = normalize_signal(opp)
        assert norm.source == "arbitrage"
        assert norm.direction is SignalDirection.NEUTRAL
        assert 0.0 < norm.strength <= 1.0
        assert norm.metadata["buy_venue"] == "raydium"
        assert norm.metadata["sell_venue"] == "pumpswap"

    def test_arbitrage_strength_saturates(self) -> None:
        # A huge net edge saturates strength at 1.0.
        opp = ArbitrageDetector().evaluate(
            TOKEN,
            [
                VenueQuote("a", price=1.00, liquidity_sol=1_000_000.0, fee_bps=1),
                VenueQuote("b", price=1.50, liquidity_sol=1_000_000.0, fee_bps=1),
            ],
            size_sol=1.0,
        )
        assert opp is not None
        assert normalize_arbitrage(opp).strength == 1.0


class TestDispatcher:
    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError):
            normalize_signal(object())
