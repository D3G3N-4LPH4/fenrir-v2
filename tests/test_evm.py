#!/usr/bin/env python3
"""
FENRIR - EVM read-only evaluation tests (on-chain EVM)

The bridge maps a discovery TokenSnapshot onto MarketData/token_data, and the evaluator
runs an EVM token through the REAL momentum/mean_reversion strategies + unified Signal
(+ optional AI brain), read-only. Verified with a real ETHEREUM snapshot and injected
fetch/brain — no network, no execution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fenrir.config import BotConfig
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.evm import (
    EvmTokenEvaluator,
    snapshot_to_market_data,
    snapshot_to_token_data,
)
from fenrir.signals import SignalAggregator
from fenrir.strategies.mean_reversion import MeanReversionStrategy
from fenrir.strategies.momentum import MomentumStrategy

TOKEN = "0x1234567890abcdef1234567890abcdef12345678"


def _uptrend_snapshot(**over: Any) -> TokenSnapshot:
    """An ETHEREUM snapshot momentum fires on."""
    base: dict[str, Any] = dict(
        chain=Chain.ETHEREUM,
        token_address=TOKEN,
        symbol="PEPE",
        name="Pepe",
        pair_address="0xpair",
        dex_id="uniswap",
        price_usd=0.0000012,
        market_cap_usd=5_000_000.0,
        fdv_usd=6_000_000.0,
        liquidity_usd=500_000.0,
        volume_5m_usd=60_000.0,
        volume_1h_usd=400_000.0,
        txns_5m_buys=70,
        txns_5m_sells=30,
        price_change_5m_pct=2.0,
        price_change_1h_pct=25.0,
        price_change_24h_pct=150.0,
        age_minutes=120.0,
        holder_count=1500,
        top_holder_pct=8.0,
        safety=SafetySignals(honeypot=False, buy_tax_pct=0.0, sell_tax_pct=0.0),
    )
    base.update(over)
    return TokenSnapshot(**base)


def _fetch(snapshot: TokenSnapshot | None) -> Any:
    async def _get(addr: str, chain: Any) -> TokenSnapshot | None:
        return snapshot

    return _get


class TestAdapters:
    def test_snapshot_to_market_data(self) -> None:
        md = snapshot_to_market_data(_uptrend_snapshot())
        assert md.token_address == TOKEN
        assert md.dex_id == "uniswap"
        assert md.liquidity_usd == 500_000.0
        assert md.price_change_1h_pct == 25.0
        assert md.buy_pressure_5m == pytest.approx(0.70, abs=0.01)  # computed property
        assert md.price_sol == 0.0  # EVM is not SOL-denominated

    def test_snapshot_to_token_data(self) -> None:
        td = snapshot_to_token_data(_uptrend_snapshot())
        assert td["chain"] == "ethereum"
        assert td["symbol"] == "PEPE"
        assert td["dex_price_change_1h_pct"] == 25.0
        assert td["dex_buy_pressure_5m"] == pytest.approx(0.70, abs=0.01)
        assert td["honeypot"] is False
        assert td["holder_count"] == 1500


class TestEvaluator:
    def _strategies(self) -> list[Any]:
        cfg = BotConfig()
        return [MomentumStrategy(cfg), MeanReversionStrategy(cfg)]

    async def test_momentum_fires_on_evm_token(self) -> None:
        ev = EvmTokenEvaluator(self._strategies(), _fetch(_uptrend_snapshot()))
        result = await ev.evaluate(TOKEN, Chain.ETHEREUM)
        assert result is not None
        assert result.chain == "ethereum"
        assert result.symbol == "PEPE"
        # Momentum fires (uptrend); mean_reversion does not (not oversold).
        assert result.strategy_ids == ["momentum"]
        assert 0.0 < result.signals[0].strength <= 1.0

    async def test_no_snapshot_returns_none(self) -> None:
        ev = EvmTokenEvaluator(self._strategies(), _fetch(None))
        assert await ev.evaluate(TOKEN, Chain.ETHEREUM) is None

    async def test_fetch_error_returns_none(self) -> None:
        async def boom(addr: str, chain: Any) -> Any:
            raise RuntimeError("provider down")

        ev = EvmTokenEvaluator(self._strategies(), boom)
        assert await ev.evaluate(TOKEN, Chain.ETHEREUM) is None

    async def test_no_signal_when_nothing_fires(self) -> None:
        # Flat snapshot: neither strategy claims it.
        flat = _uptrend_snapshot(price_change_1h_pct=0.0, price_change_5m_pct=0.0)
        ev = EvmTokenEvaluator(self._strategies(), _fetch(flat))
        result = await ev.evaluate(TOKEN, Chain.ETHEREUM)
        assert result is not None
        assert result.signals == []
        assert result.ai_decision is None

    async def test_brain_decision_populated(self) -> None:
        analysis = SimpleNamespace(
            decision=SimpleNamespace(value="BUY"), confidence=0.82, risk_score=3.0
        )
        brain = SimpleNamespace(evaluate_entry=AsyncMock(return_value=(True, analysis, None)))
        ev = EvmTokenEvaluator(self._strategies(), _fetch(_uptrend_snapshot()), brain=brain)
        result = await ev.evaluate(TOKEN, Chain.ETHEREUM)
        assert result is not None
        assert result.ai_decision == {
            "should_buy": True,
            "decision": "BUY",
            "confidence": 0.82,
            "risk_score": 3.0,
        }
        brain.evaluate_entry.assert_awaited_once()

    async def test_brain_error_is_isolated(self) -> None:
        brain = SimpleNamespace(evaluate_entry=AsyncMock(side_effect=RuntimeError("AI down")))
        ev = EvmTokenEvaluator(self._strategies(), _fetch(_uptrend_snapshot()), brain=brain)
        result = await ev.evaluate(TOKEN, Chain.ETHEREUM)
        assert result is not None
        assert result.ai_decision is None  # AI failed, but the read still returns
        assert result.strategy_ids == ["momentum"]  # strategies unaffected

    async def test_confluence_via_aggregator(self) -> None:
        # A second momentum-like strategy makes the token confluent.
        class _Momentum2(MomentumStrategy):
            strategy_id = "momentum_2"

        cfg = BotConfig()
        agg = SignalAggregator()
        ev = EvmTokenEvaluator(
            [MomentumStrategy(cfg), _Momentum2(cfg)], _fetch(_uptrend_snapshot()), aggregator=agg
        )
        result = await ev.evaluate(TOKEN, Chain.ETHEREUM)
        assert result is not None
        assert result.confluence is not None
        assert result.confluence.source_count == 2
        assert result.confluence.is_confluent() is True

    async def test_to_dict(self) -> None:
        ev = EvmTokenEvaluator(self._strategies(), _fetch(_uptrend_snapshot()))
        result = await ev.evaluate(TOKEN, Chain.ETHEREUM)
        assert result is not None
        d = result.to_dict()
        assert d["chain"] == "ethereum"
        assert d["token_address"] == TOKEN
        assert len(d["signals"]) == 1
