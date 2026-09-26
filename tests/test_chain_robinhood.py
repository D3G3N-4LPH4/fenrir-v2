#!/usr/bin/env python3
"""
FENRIR - Robinhood EVM chain support tests

DexScreener indexes tokens on Robinhood's EVM L2 under chainId "robinhood". Before this
was mapped, such tokens were silently dropped (Chain.from_dexscreener → None). These pin
the mapping, the is_evm helper, and that a robinhood snapshot flows through the EVM
adapters + evaluator like any other EVM chain. No network.
"""

from __future__ import annotations

from typing import Any

from fenrir.config import BotConfig
from fenrir.discovery.models import Chain, TokenSnapshot
from fenrir.evm import EvmTokenEvaluator, snapshot_to_token_data
from fenrir.strategies.momentum import MomentumStrategy

CA = "0x32dae312abe8f6fdb782907b85edbc90d2e74b02"


class TestChainMapping:
    def test_robinhood_maps(self) -> None:
        assert Chain.from_dexscreener("robinhood") is Chain.ROBINHOOD
        assert Chain.from_dexscreener("Robinhood") is Chain.ROBINHOOD  # case-insensitive

    def test_is_evm(self) -> None:
        assert Chain.ROBINHOOD.is_evm is True
        assert Chain.ETHEREUM.is_evm is True
        assert Chain.SOLANA.is_evm is False

    def test_unknown_chain_still_none(self) -> None:
        assert Chain.from_dexscreener("some-other-l2") is None
        assert Chain.from_dexscreener(None) is None


def _robinhood_snapshot(**over: Any) -> TokenSnapshot:
    base: dict[str, Any] = dict(
        chain=Chain.ROBINHOOD,
        token_address=CA,
        symbol="0XP",
        dex_id="uniswap",
        price_usd=0.00036,
        market_cap_usd=359_000.0,
        liquidity_usd=57_500.0,
        # A momentum-firing regime (young, uptrend, buyers) to prove the pipeline runs.
        age_minutes=120.0,
        volume_5m_usd=30_000.0,
        volume_1h_usd=200_000.0,
        txns_5m_buys=70,
        txns_5m_sells=30,
        price_change_5m_pct=2.0,
        price_change_1h_pct=25.0,
        price_change_24h_pct=150.0,
    )
    base.update(over)
    return TokenSnapshot(**base)


class TestRobinhoodThroughEvm:
    def test_token_data_carries_chain(self) -> None:
        td = snapshot_to_token_data(_robinhood_snapshot())
        assert td["chain"] == "robinhood"
        assert td["symbol"] == "0XP"

    async def test_evaluator_resolves_and_fires(self) -> None:
        snap = _robinhood_snapshot()

        async def fetch(token: str, chain: Any = None) -> TokenSnapshot:
            return snap

        ev = EvmTokenEvaluator([MomentumStrategy(BotConfig())], fetch)
        result = await ev.evaluate(CA)
        assert result is not None
        assert result.chain == "robinhood"  # chain comes from the snapshot, authoritative
        assert result.strategy_ids == ["momentum"]  # a real Robinhood-chain setup evaluates
