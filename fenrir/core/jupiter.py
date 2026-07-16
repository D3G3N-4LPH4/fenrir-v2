#!/usr/bin/env python3
"""
FENRIR - Jupiter Swap Engine

Jupiter aggregator integration.
Finding the best price is an art form.
"""

from typing import Any, cast

import aiohttp

from fenrir.config import BotConfig
from fenrir.core.circuit_breaker import CircuitBreaker, CircuitOpen
from fenrir.logger import FenrirLogger


class JupiterSwapEngine:
    """
    Jupiter aggregator integration.
    Finding the best price is an art form.
    """

    # Jupiter's legacy quote-api.jup.ag/v6 host was retired; the current free
    # (keyless) endpoint is lite-api.jup.ag/swap/v1 (paid tier: api.jup.ag).
    JUPITER_API = "https://lite-api.jup.ag/swap/v1"
    # Tokens v2 (keyless) — trending/top lists for the market scanner.
    TOKENS_API = "https://lite-api.jup.ag/tokens/v2"

    def __init__(
        self, config: BotConfig, logger: FenrirLogger, breaker: CircuitBreaker | None = None
    ):
        self.config = config
        self.logger = logger
        self._breaker = breaker
        self.session: aiohttp.ClientSession | None = None

    async def initialize(self):
        """Start the HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def get_quote(
        self, input_mint: str, output_mint: str, amount: int, slippage_bps: int
    ) -> dict | None:
        """
        Request a quote from Jupiter.
        The first step in any great trade.
        """
        if not self.session:
            await self.initialize()
        assert self.session is not None

        if self._breaker:
            try:
                self._breaker.check()
            except CircuitOpen:
                self.logger.warning("Jupiter circuit OPEN: get_quote")
                return None

        try:
            url = f"{self.JUPITER_API}/quote"
            params: dict[str, str] = {
                "inputMint": str(input_mint),
                "outputMint": str(output_mint),
                "amount": str(amount),
                "slippageBps": str(slippage_bps),
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    if self._breaker:
                        self._breaker.record_success()
                    return cast("dict[Any, Any]", await response.json())
                else:
                    if self._breaker:
                        self._breaker.record_failure(f"HTTP {response.status}")
                    self.logger.warning(f"Jupiter quote failed: {response.status}")
                    return None
        except Exception as e:
            if self._breaker:
                self._breaker.record_failure(type(e).__name__)
            self.logger.error("Failed to get Jupiter quote", e)
            return None

    async def get_swap_transaction(
        self, quote: dict, user_public_key: str, priority_fee_lamports: int | None = None
    ) -> str | None:
        """
        Build the swap transaction from a quote.
        Turning intention into executable bytes.

        ``priority_fee_lamports`` overrides the flat configured fee — callers that
        know the trade size pass a size-capped value, since the raw config fee is a
        lamport constant (0.002 SOL on degen) that eats small positions alive.
        """
        if not self.session:
            await self.initialize()
        assert self.session is not None

        if self._breaker:
            try:
                self._breaker.check()
            except CircuitOpen:
                self.logger.warning("Jupiter circuit OPEN: get_swap_transaction")
                return None

        try:
            url = f"{self.JUPITER_API}/swap"
            payload = {
                "quoteResponse": quote,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": (
                    priority_fee_lamports
                    if priority_fee_lamports is not None
                    else self.config.priority_fee_lamports
                ),
            }

            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    if self._breaker:
                        self._breaker.record_success()
                    data = await response.json()
                    return cast("str | None", data.get("swapTransaction"))
                else:
                    if self._breaker:
                        self._breaker.record_failure(f"HTTP {response.status}")
                    self.logger.warning(f"Jupiter swap tx failed: {response.status}")
                    return None
        except Exception as e:
            if self._breaker:
                self._breaker.record_failure(type(e).__name__)
            self.logger.error("Failed to build swap transaction", e)
            return None

    async def get_trending_tokens(
        self, category: str = "toptraded", interval: str = "24h"
    ) -> list[dict]:
        """Fetch a Jupiter trending/top token list (keyless Tokens-v2).

        category: toporganicscore | toptraded | toptrending | recent; interval:
        5m|1h|6h|24h. ``recent`` is a newly-created-pairs feed and takes no
        interval (``/tokens/v2/recent``); the others are interval-scoped.
        Each token carries mcap, liquidity, usdPrice, holderCount, isVerified,
        organicScore, socials, decimals, graduatedAt. Returns [] on any failure.
        """
        if not self.session:
            await self.initialize()
        assert self.session is not None
        try:
            url = (
                f"{self.TOKENS_API}/{category}"
                if category == "recent"
                else f"{self.TOKENS_API}/{category}/{interval}"
            )
            async with self.session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(
                        f"Jupiter trending {category}/{interval}: {response.status}"
                    )
                    return []
                data = await response.json()
            tokens = data if isinstance(data, list) else data.get("tokens", [])
            return cast("list[dict]", tokens)
        except Exception as e:
            self.logger.error("Failed to fetch Jupiter trending tokens", e)
            return []

    async def search_token(self, mint: str) -> dict | None:
        """Look up ONE token by mint address (keyless Tokens-v2 search).

        Returns the same token shape as :meth:`get_trending_tokens` entries (mcap,
        liquidity, holderCount, graduatedAt, audit, socials) or None if unknown.
        Search is fuzzy, so only an exact mint match is accepted.
        """
        if not self.session:
            await self.initialize()
        assert self.session is not None
        try:
            url = f"{self.TOKENS_API}/search?query={mint}"
            async with self.session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"Jupiter search {mint[:8]}...: {response.status}")
                    return None
                data = await response.json()
            tokens = data if isinstance(data, list) else data.get("tokens", [])
            for tok in tokens:
                if tok.get("id") == mint:
                    return cast("dict", tok)
            return None
        except Exception as e:
            self.logger.error("Failed to search Jupiter token", e)
            return None

    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
