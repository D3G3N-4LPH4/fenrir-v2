#!/usr/bin/env python3
"""
FENRIR - Market Scanner Test Suite

Covers tiering, liquidity/quality/cooldown filters, candidate shaping, and the
per-cycle emit cap + cross-category dedup of the multi-tier discovery scanner.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from fenrir.config import BotConfig
from fenrir.trading.scanner import MarketScanner

T0 = datetime(2020, 1, 1, 12, 0, 0)


def _tok(mint="Mint1", mcap=2_000_000.0, liq=100_000.0, verified=True, score=90.0, graduated=None):
    return {
        "id": mint,
        "symbol": "AAA",
        "name": "Token",
        "mcap": mcap,
        "liquidity": liq,
        "isVerified": verified,
        "organicScore": score,
        "usdPrice": 1.5,
        "holderCount": 1000,
        "decimals": 6,
        "graduatedAt": graduated,
    }


@pytest.fixture
def scanner():
    c = BotConfig()
    c.mid_cap_min_usd = 200_000.0
    c.large_cap_min_usd = 1_000_000.0
    c.scanner_min_liquidity_usd = 50_000.0
    c.scanner_require_verified = False
    c.scanner_min_organic_score = 0.0
    c.scanner_max_candidates_per_cycle = 5
    c.scanner_cooldown_minutes = 30.0
    c.scanner_categories = ["toptraded"]
    return MarketScanner(c, AsyncMock(), MagicMock())


class TestTier:
    def test_buckets(self, scanner):
        assert scanner._tier(2_000_000) == "large"
        assert scanner._tier(1_000_000) == "large"
        assert scanner._tier(500_000) == "mid"
        assert scanner._tier(199_999) is None


class TestEvaluate:
    def test_passes_large_cap(self, scanner):
        c = scanner._evaluate(_tok(), T0)
        assert c is not None
        assert c["tier"] == "large"
        assert c["token_address"] == "Mint1"
        assert c["bonding_curve_state"] is None
        assert c["decimals"] == 6

    def test_rejects_below_mid(self, scanner):
        assert scanner._evaluate(_tok(mcap=100_000), T0) is None

    def test_rejects_low_liquidity(self, scanner):
        assert scanner._evaluate(_tok(liq=1_000), T0) is None

    def test_require_verified(self, scanner):
        scanner.config.scanner_require_verified = True
        assert scanner._evaluate(_tok(verified=False), T0) is None

    def test_min_organic_score(self, scanner):
        scanner.config.scanner_min_organic_score = 95.0
        assert scanner._evaluate(_tok(score=50), T0) is None

    def test_cooldown_blocks_then_allows(self, scanner):
        scanner._cooldown["Mint1"] = T0
        assert scanner._evaluate(_tok(), T0 + timedelta(minutes=5)) is None
        assert scanner._evaluate(_tok(), T0 + timedelta(minutes=31)) is not None

    def test_mid_tier_migrated_flag(self, scanner):
        c = scanner._evaluate(_tok(mcap=500_000, graduated="2024-01-01"), T0)
        assert c is not None
        assert c["tier"] == "mid"
        assert c["migrated"] is True


class TestScanOnce:
    @pytest.mark.asyncio
    async def test_emits_up_to_cap(self, scanner):
        scanner.config.scanner_max_candidates_per_cycle = 2
        scanner.jupiter.get_trending_tokens.return_value = [_tok(mint=f"m{i}") for i in range(5)]
        got: list = []

        async def cb(c):
            got.append(c)

        await scanner._scan_once(cb)
        assert len(got) == 2

    @pytest.mark.asyncio
    async def test_dedup_across_categories(self, scanner):
        scanner.config.scanner_categories = ["toptraded", "toporganicscore"]
        scanner.jupiter.get_trending_tokens.return_value = [_tok(mint="same")]
        got: list = []

        async def cb(c):
            got.append(c)

        await scanner._scan_once(cb)
        assert len(got) == 1  # same mint surfaced by both categories -> emitted once


def _boost(mint="Boost1", amount=500):
    return {
        "chainId": "solana",
        "tokenAddress": mint,
        "totalAmount": amount,
        "description": "meme",
        "links": [],
    }


def _pair(mcap=2_000_000.0, liq=100_000.0, buys=80, sells=20):
    return {
        "baseToken": {"address": "Boost1", "name": "BoostTok", "symbol": "BST"},
        "priceUsd": "0.0025",
        "marketCap": mcap,
        "fdv": mcap,
        "liquidity": {"usd": liq},
        "volume": {"m5": 5000.0, "h1": 20000.0},
        "txns": {"m5": {"buys": buys, "sells": sells}},
        "priceChange": {"h1": 12.0},
    }


class TestDexBoosts:
    def test_build_candidate_shape(self, scanner):
        c = scanner._build_dex_candidate(_boost(), _pair(), "large")
        assert c["token_address"] == "Boost1"
        assert c["symbol"] == "BST"
        assert c["tier"] == "large"
        assert c["source"] == "dexscreener_boost"
        assert c["market_cap_usd"] == 2_000_000.0
        assert c["decimals"] == 6
        assert c["bonding_curve_state"] is None
        # momentum → AI context
        assert c["dex_txns_5m_buys"] == 80
        assert c["dex_buy_pressure_5m"] == 0.8
        assert c["dex_price_change_1h_pct"] == 12.0

    def test_evaluate_below_mid_rejected(self, scanner):
        assert scanner._evaluate_dex(_boost(), _pair(mcap=100_000), T0) is None

    def test_evaluate_low_liquidity_rejected(self, scanner):
        assert scanner._evaluate_dex(_boost(), _pair(liq=1_000), T0) is None

    def test_evaluate_large_cap_passes(self, scanner):
        c = scanner._evaluate_dex(_boost(), _pair(), T0)
        assert c is not None
        assert c["tier"] == "large"

    def test_evaluate_cooldown_blocks_then_allows(self, scanner):
        assert scanner._evaluate_dex(_boost(), _pair(), T0) is not None
        scanner._cooldown["Boost1"] = T0
        assert scanner._evaluate_dex(_boost(), _pair(), T0 + timedelta(minutes=5)) is None
        assert scanner._evaluate_dex(_boost(), _pair(), T0 + timedelta(minutes=31)) is not None

    @pytest.mark.asyncio
    async def test_scan_dex_boosts_emits(self, scanner):
        scanner._fetch_dex_boosts = AsyncMock(return_value=[_boost("b1"), _boost("b2")])
        scanner._fetch_dex_pair = AsyncMock(return_value=_pair())
        got: list = []

        async def cb(c):
            got.append(c)

        n = await scanner._scan_dex_boosts(cb, T0, set(), 0)
        assert n == 2
        assert {c["token_address"] for c in got} == {"b1", "b2"}

    @pytest.mark.asyncio
    async def test_scan_dex_respects_remaining_cap(self, scanner):
        scanner.config.scanner_max_candidates_per_cycle = 3
        scanner._fetch_dex_boosts = AsyncMock(return_value=[_boost(f"b{i}") for i in range(5)])
        scanner._fetch_dex_pair = AsyncMock(return_value=_pair())
        got: list = []

        async def cb(c):
            got.append(c)

        # 1 already emitted by the Jupiter pass -> only 2 more allowed.
        n = await scanner._scan_dex_boosts(cb, T0, set(), 1)
        assert n == 2

    @pytest.mark.asyncio
    async def test_scan_dex_skips_seen(self, scanner):
        scanner._fetch_dex_boosts = AsyncMock(return_value=[_boost("dup")])
        scanner._fetch_dex_pair = AsyncMock(return_value=_pair())
        got: list = []

        async def cb(c):
            got.append(c)

        n = await scanner._scan_dex_boosts(cb, T0, {"dup"}, 0)
        assert n == 0

    @pytest.mark.asyncio
    async def test_scan_once_includes_dex_when_enabled(self, scanner):
        scanner.config.scanner_dex_boosts_enabled = True
        scanner.jupiter.get_trending_tokens.return_value = []
        scanner._fetch_dex_boosts = AsyncMock(return_value=[_boost("bx")])
        scanner._fetch_dex_pair = AsyncMock(return_value=_pair())
        got: list = []

        async def cb(c):
            got.append(c)

        await scanner._scan_once(cb)
        assert [c["token_address"] for c in got] == ["bx"]

    @pytest.mark.asyncio
    async def test_scan_once_skips_dex_when_disabled(self, scanner):
        scanner.config.scanner_dex_boosts_enabled = False
        scanner.jupiter.get_trending_tokens.return_value = []
        scanner._fetch_dex_boosts = AsyncMock(return_value=[_boost("bx")])
        scanner._fetch_dex_pair = AsyncMock(return_value=_pair())
        got: list = []

        async def cb(c):
            got.append(c)

        await scanner._scan_once(cb)
        assert got == []
        scanner._fetch_dex_boosts.assert_not_called()
