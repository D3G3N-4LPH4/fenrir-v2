#!/usr/bin/env python3
"""
FENRIR - Jupiter Swap Engine

Jupiter aggregator integration.
Finding the best price is an art form.
"""

import aiohttp

from fenrir.config import BotConfig
from fenrir.logger import FenrirLogger


class JupiterSwapEngine:
    """
    Jupiter aggregator integration.
    Finding the best price is an art form.
    """

    JUPITER_API = "https://quote-api.jup.ag/v6"

    def __init__(self, config: BotConfig, logger: FenrirLogger):
        self.config = config
        self.logger = logger
        self.session: aiohttp.ClientSession | None = None

    async def initialize(self):
        """Start the HTTP session."""
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

        try:
            url = f"{self.JUPITER_API}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": slippage_bps,
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.warning(f"Jupiter quote failed: {response.status}")
                    return None
        except Exception as e:
            self.logger.error("Failed to get Jupiter quote", e)
            return None

    async def get_swap_transaction(self, quote: dict, user_public_key: str) -> str | None:
        """
        Build the swap transaction from a quote.
        Turning intention into executable bytes.
        """
        if not self.session:
            await self.initialize()

        try:
            url = f"{self.JUPITER_API}/swap"
            payload = {
                "quoteResponse": quote,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": self.config.priority_fee_lamports,
            }

            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("swapTransaction")
                else:
                    self.logger.warning(f"Jupiter swap tx failed: {response.status}")
                    return None
        except Exception as e:
            self.logger.error("Failed to build swap transaction", e)
            return None

    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
