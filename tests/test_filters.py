#!/usr/bin/env python3
"""
FENRIR - Filters Test Suite

Covers the security hard-filter and the two-tier market-condition filter.
Network I/O is fully mocked — no RPC or DexScreener calls are made.

Run with: pytest tests/test_filters.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fenrir.filters import (
    LiveLaunchFilterConfig,
    MarketData,
    MarketFilter,
    MarketFilterConfig,
    MomentumBreakoutFilterConfig,
    SecurityCheckResult,
    SecurityFilter,
    SecurityFilterConfig,
)

TOKEN = "So11111111111111111111111111111111111111112"
LP_MINT = "LPmint1111111111111111111111111111111111111"


# ---------------------------------------------------------------------------
# MarketData model
# ---------------------------------------------------------------------------


class TestMarketData:
    def test_txn_totals(self) -> None:
        md = MarketData(
            token_address=TOKEN,
            txns_5m_buys=60,
            txns_5m_sells=40,
            txns_1h_buys=300,
            txns_1h_sells=200,
        )
        assert md.txns_5m_total == 100
        assert md.txns_1h_total == 500

    def test_buy_pressure(self) -> None:
        md = MarketData(token_address=TOKEN, txns_5m_buys=75, txns_5m_sells=25)
        assert md.buy_pressure_5m == 0.75

    def test_buy_pressure_no_txns_defaults_to_neutral(self) -> None:
        md = MarketData(token_address=TOKEN)
        assert md.buy_pressure_5m == 0.5


# ---------------------------------------------------------------------------
# MarketFilter
# ---------------------------------------------------------------------------


def _passing_live_launch_data() -> MarketData:
    return MarketData(
        token_address=TOKEN,
        age_minutes=20.0,
        liquidity_usd=25_000.0,
        market_cap_usd=80_000.0,
        volume_5m_usd=30_000.0,
        txns_5m_buys=120,
        txns_5m_sells=40,
    )


def _passing_momentum_data() -> MarketData:
    return MarketData(
        token_address=TOKEN,
        age_minutes=180.0,
        liquidity_usd=80_000.0,
        market_cap_usd=500_000.0,
        volume_1h_usd=150_000.0,
        unique_buyers_1h=400,
    )


class TestMarketFilter:
    async def test_disabled_config_passes_through(self) -> None:
        mf = MarketFilter(MarketFilterConfig(enabled=False))
        result = await mf.check(TOKEN)
        assert result.passed is True
        assert result.tier == "disabled"

    async def test_fetch_failure_fail_open(self) -> None:
        mf = MarketFilter(MarketFilterConfig(fail_open_on_fetch_error=True))
        mf._fetch_market_data = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.passed is True
        assert result.warnings

    async def test_fetch_failure_fail_closed(self) -> None:
        mf = MarketFilter(MarketFilterConfig(fail_open_on_fetch_error=False))
        mf._fetch_market_data = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.passed is False
        assert result.failures

    async def test_live_launch_pass(self) -> None:
        mf = MarketFilter(MarketFilterConfig())
        mf._fetch_market_data = AsyncMock(return_value=_passing_live_launch_data())  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.tier == "live_launch"
        assert result.passed is True
        assert result.failures == []

    async def test_live_launch_reject_low_liquidity(self) -> None:
        md = _passing_live_launch_data()
        md.liquidity_usd = 1_000.0
        mf = MarketFilter(MarketFilterConfig())
        mf._fetch_market_data = AsyncMock(return_value=md)  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.tier == "live_launch"
        assert result.passed is False
        assert any("LP" in f for f in result.failures)

    async def test_live_launch_reject_mcap_too_high(self) -> None:
        md = _passing_live_launch_data()
        md.market_cap_usd = 500_000.0
        mf = MarketFilter(MarketFilterConfig())
        mf._fetch_market_data = AsyncMock(return_value=md)  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.passed is False
        assert any("MCap" in f and ">" in f for f in result.failures)

    async def test_momentum_pass(self) -> None:
        mf = MarketFilter(MarketFilterConfig())
        mf._fetch_market_data = AsyncMock(return_value=_passing_momentum_data())  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.tier == "momentum_breakout"
        assert result.passed is True

    async def test_momentum_unique_buyers_proxy_warning(self) -> None:
        md = _passing_momentum_data()
        md.unique_buyers_1h = 0  # unavailable
        md.txns_1h_buys = 400  # proxy exceeds threshold
        mf = MarketFilter(MarketFilterConfig())
        mf._fetch_market_data = AsyncMock(return_value=md)  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.passed is True
        assert any("proxy" in w for w in result.warnings)

    async def test_momentum_reject_insufficient_buyers(self) -> None:
        md = _passing_momentum_data()
        md.unique_buyers_1h = 10
        md.txns_1h_buys = 10
        mf = MarketFilter(MarketFilterConfig())
        mf._fetch_market_data = AsyncMock(return_value=md)  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.passed is False
        assert any("Buyers" in f for f in result.failures)

    async def test_age_outside_windows_rejected(self) -> None:
        md = MarketData(token_address=TOKEN, age_minutes=5_000.0, liquidity_usd=80_000.0)
        mf = MarketFilter(MarketFilterConfig())
        mf._fetch_market_data = AsyncMock(return_value=md)  # type: ignore[method-assign]
        result = await mf.check(TOKEN)
        assert result.tier == "none"
        assert result.passed is False
        assert any("outside filter windows" in f for f in result.failures)

    def test_parse_pair(self) -> None:
        mf = MarketFilter(MarketFilterConfig())
        pair = {
            "pairAddress": "PAIR",
            "dexId": "raydium",
            "priceUsd": "0.0025",
            "liquidity": {"usd": 42_000.0},
            "marketCap": 120_000.0,
            "fdv": 130_000.0,
            "volume": {"m5": 5_000.0, "h1": 20_000.0, "h6": 50_000.0, "h24": 90_000.0},
            "txns": {"m5": {"buys": 80, "sells": 20}, "h1": {"buys": 300, "sells": 150}},
            "priceChange": {"m5": 3.2, "h1": -1.1, "h6": 12.0, "h24": 40.0},
            "pairCreatedAt": None,
        }
        md = mf._parse_pair(TOKEN, pair)
        assert md.pair_address == "PAIR"
        assert md.dex_id == "raydium"
        assert md.liquidity_usd == 42_000.0
        assert md.txns_5m_total == 100
        assert md.txns_1h_buys == 300
        assert md.price_change_24h_pct == 40.0
        assert md.raw is pair

    async def test_close_is_safe_without_session(self) -> None:
        mf = MarketFilter(MarketFilterConfig())
        await mf.close()  # no session created — must not raise


# ---------------------------------------------------------------------------
# SecurityFilter
# ---------------------------------------------------------------------------

RPC_URL = "https://rpc.example.com"


def _clean_mint_info() -> dict[str, object]:
    return {
        "mint_authority": None,
        "freeze_authority": None,
        "decimals": 6,
        "supply": "1000000000",
        "is_initialized": True,
    }


class TestSecurityFilter:
    async def test_mint_info_unavailable_fails(self) -> None:
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await sf.check(TOKEN)
        assert result.passed is False
        assert any("mint account info" in f for f in result.failures)

    async def test_mint_authority_not_revoked_fails(self) -> None:
        info = _clean_mint_info()
        info["mint_authority"] = "SomeAuthorityPubkey1111111111111111111111111"
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=info)  # type: ignore[method-assign]
        result = await sf.check(TOKEN)
        assert result.passed is False
        assert any("Mint authority" in f for f in result.failures)

    async def test_freeze_authority_not_revoked_fails(self) -> None:
        info = _clean_mint_info()
        info["freeze_authority"] = "FreezeAuthorityPubkey11111111111111111111111"
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=info)  # type: ignore[method-assign]
        result = await sf.check(TOKEN)
        assert result.passed is False
        assert any("Freeze authority" in f for f in result.failures)

    async def test_full_pass(self) -> None:
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=_clean_mint_info())  # type: ignore[method-assign]
        sf._fetch_lp_burned_pct = AsyncMock(return_value=99.0)  # type: ignore[method-assign]
        sf._fetch_top10_holder_pct = AsyncMock(return_value=12.0)  # type: ignore[method-assign]
        result = await sf.check(TOKEN, LP_MINT)
        assert result.passed is True
        assert result.failures == []
        assert result.details["lp_burned_pct"] == 99.0
        assert result.details["top10_holder_pct"] == 12.0

    async def test_lp_burn_below_threshold_fails(self) -> None:
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=_clean_mint_info())  # type: ignore[method-assign]
        sf._fetch_lp_burned_pct = AsyncMock(return_value=50.0)  # type: ignore[method-assign]
        sf._fetch_top10_holder_pct = AsyncMock(return_value=12.0)  # type: ignore[method-assign]
        result = await sf.check(TOKEN, LP_MINT)
        assert result.passed is False
        assert any("LP burned" in f for f in result.failures)

    async def test_lp_burn_unverifiable_warns_but_passes(self) -> None:
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=_clean_mint_info())  # type: ignore[method-assign]
        sf._fetch_lp_burned_pct = AsyncMock(return_value=None)  # type: ignore[method-assign]
        sf._fetch_top10_holder_pct = AsyncMock(return_value=12.0)  # type: ignore[method-assign]
        result = await sf.check(TOKEN, LP_MINT)
        assert result.passed is True
        assert any("LP burn" in w for w in result.warnings)

    async def test_top10_concentration_too_high_fails(self) -> None:
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=_clean_mint_info())  # type: ignore[method-assign]
        sf._fetch_lp_burned_pct = AsyncMock(return_value=99.0)  # type: ignore[method-assign]
        sf._fetch_top10_holder_pct = AsyncMock(return_value=55.0)  # type: ignore[method-assign]
        result = await sf.check(TOKEN, LP_MINT)
        assert result.passed is False
        assert any("Top-10 holders" in f for f in result.failures)

    async def test_holder_fetch_error_fail_closed(self) -> None:
        sf = SecurityFilter(SecurityFilterConfig(fail_open_on_holder_fetch_error=False), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=_clean_mint_info())  # type: ignore[method-assign]
        sf._fetch_lp_burned_pct = AsyncMock(return_value=99.0)  # type: ignore[method-assign]
        sf._fetch_top10_holder_pct = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await sf.check(TOKEN, LP_MINT)
        assert result.passed is False
        assert any("holder distribution" in f for f in result.failures)

    async def test_holder_fetch_error_fail_open(self) -> None:
        sf = SecurityFilter(SecurityFilterConfig(fail_open_on_holder_fetch_error=True), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=_clean_mint_info())  # type: ignore[method-assign]
        sf._fetch_lp_burned_pct = AsyncMock(return_value=99.0)  # type: ignore[method-assign]
        sf._fetch_top10_holder_pct = AsyncMock(return_value=None)  # type: ignore[method-assign]
        result = await sf.check(TOKEN, LP_MINT)
        assert result.passed is True
        assert any("Holder distribution unavailable" in w for w in result.warnings)

    async def test_mint_failure_short_circuits_before_lp_check(self) -> None:
        info = _clean_mint_info()
        info["mint_authority"] = "StillHasAuthority1111111111111111111111111111"
        lp_mock = AsyncMock(return_value=99.0)
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        sf._fetch_mint_info = AsyncMock(return_value=info)  # type: ignore[method-assign]
        sf._fetch_lp_burned_pct = lp_mock  # type: ignore[method-assign]
        result = await sf.check(TOKEN, LP_MINT)
        assert result.passed is False
        lp_mock.assert_not_awaited()

    async def test_close_is_safe_without_session(self) -> None:
        sf = SecurityFilter(SecurityFilterConfig(), RPC_URL)
        await sf.close()  # no session created — must not raise


def test_security_result_str() -> None:
    ok = SecurityCheckResult(passed=True, token_address=TOKEN)
    assert ok.token_address[:8] in str(ok)
    assert "PASS" in str(ok)

    bad = SecurityCheckResult(passed=False, token_address=TOKEN, failures=["nope"])
    assert "FAIL" in str(bad)
    assert "nope" in str(bad)


@pytest.mark.parametrize(
    "cfg_cls",
    [LiveLaunchFilterConfig, MomentumBreakoutFilterConfig],
)
def test_market_config_defaults_construct(cfg_cls: type) -> None:
    cfg = cfg_cls()
    assert cfg.min_liquidity_usd > 0
