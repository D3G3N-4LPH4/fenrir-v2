#!/usr/bin/env python3
"""
FENRIR - Price feed tests

`PriceQuote.price` is SOL-per-token and the aggregator confidence-weights every
source into one number, which the engine marks positions against. These tests pin
the denomination contract. Network is fully mocked.
"""

from __future__ import annotations

from typing import Any, cast

import aiohttp
import pytest

from fenrir.data.price_feed import WSOL_MINT, PriceFeedManager, PriceSource

MINT = "METvsvVRapdj9cFLzq4Tr43xK4tAjQfwX76z3n6mWQL"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def _pair(quote_mint: str, price_native: str, liq: float, base_mint: str = MINT) -> dict[str, Any]:
    return {
        "baseToken": {"address": base_mint, "symbol": "MET"},
        "quoteToken": {"address": quote_mint, "symbol": "Q"},
        "priceNative": price_native,
        "liquidity": {"usd": liq},
        "volume": {"h24": 1000.0},
        "priceChange": {"h24": 1.0},
    }


class _Resp:
    def __init__(self, body: dict[str, Any], status: int = 200):
        self.status = status
        self._body = body

    async def json(self, **_kw: Any) -> dict[str, Any]:
        return self._body

    async def __aenter__(self) -> _Resp:
        return self

    async def __aexit__(self, *_a: Any) -> bool:
        return False


class _Session:
    def __init__(self, body: dict[str, Any]):
        self._body = body

    def get(self, *_a: Any, **_kw: Any) -> _Resp:
        return _Resp(self._body)

    async def close(self) -> None:
        pass


def _feed(body: dict[str, Any]) -> PriceFeedManager:
    pf = PriceFeedManager()
    pf.session = cast(aiohttp.ClientSession, _Session(body))
    return pf


class TestDexScreenerDenomination:
    """priceNative is quoted in the pair's QUOTE token — only SOL pairs give SOL."""

    @pytest.mark.asyncio
    async def test_prefers_sol_pair_over_deeper_usdc_pair(self):
        """Regression: MET's $413M USDC pool reported priceNative=778.36 (dollars)
        while the token actually traded at ~0.00206 SOL. Selecting purely by
        liquidity marked positions ~380,000x off and produced phantom PnL."""
        body = {
            "pairs": [
                _pair(USDC, "778.3560", 413_343_481.0),  # deepest, but USD-denominated
                _pair(WSOL_MINT, "0.002057", 753_123.0),  # the real SOL price
            ]
        }
        quote = await _feed(body)._fetch_dexscreener_price(MINT)

        assert quote is not None
        assert quote.price == pytest.approx(0.002057)
        assert quote.source is PriceSource.DEXSCREENER

    @pytest.mark.asyncio
    async def test_picks_deepest_among_sol_pairs(self):
        body = {
            "pairs": [
                _pair(WSOL_MINT, "0.002069", 251_613.0),
                _pair(WSOL_MINT, "0.002057", 753_123.0),  # deeper SOL pool wins
                _pair(USDC, "0.4998", 7_837_487.0),
            ]
        }
        quote = await _feed(body)._fetch_dexscreener_price(MINT)

        assert quote is not None
        assert quote.price == pytest.approx(0.002057)

    @pytest.mark.asyncio
    async def test_no_sol_pair_returns_none_rather_than_usd(self):
        """Without a SOL market, emit nothing — never a USD price into a SOL field."""
        body = {"pairs": [_pair(USDC, "0.4998", 7_837_487.0)]}
        assert await _feed(body)._fetch_dexscreener_price(MINT) is None

    @pytest.mark.asyncio
    async def test_ignores_pairs_where_mint_is_the_quote_token(self):
        """A SOL/MET pair prices SOL in MET — the inverse of what we want."""
        body = {
            "pairs": [
                _pair(MINT, "0.5", 9_000_000.0, base_mint=WSOL_MINT),  # SOL/MET, wrong way
            ]
        }
        assert await _feed(body)._fetch_dexscreener_price(MINT) is None


class TestSourceSelection:
    def test_birdeye_excluded_from_aggregation(self):
        """Birdeye returns USD; including it would corrupt the SOL average."""
        import inspect

        src = inspect.getsource(PriceFeedManager._fetch_all_sources)
        assert "_fetch_birdeye_price" not in src
        assert "_fetch_dexscreener_price" in src
