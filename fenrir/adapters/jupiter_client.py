"""
fenrir/adapters/jupiter_client.py

Typed Jupiter API client for FENRIR v2.
Covers the APIs most relevant to FENRIR's strategy stack.
Based on: https://github.com/jup-ag/agent-skills (integrating-jupiter skill)
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiohttp

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def assert_jupiter_auth() -> str:
    """Fail fast if JUPITER_API_KEY is not set."""
    key = os.getenv("JUPITER_API_KEY")
    if not key:
        raise RuntimeError(
            "JUPITER_API_KEY not set. Obtain from portal.jup.ag and add to .env"
        )
    return key


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

# Swap execute error codes from the integrating-jupiter skill
RETRYABLE_SWAP_CODES = frozenset([-1, -1000, -1001, -1004, -2000, -2001, -2003, -2004])

class JupiterErrorCode(Enum):
    RATE_LIMITED      = "RATE_LIMITED"
    EXPIRED_ORDER     = -1
    INVALID_TX        = -2
    INVALID_BYTES     = -3
    FAILED_LANDING    = -1000
    UNKNOWN_AGG       = -1001
    INVALID_AGG_TX    = -1002
    UNSIGNED_TX       = -1003
    STALE_BLOCKHASH   = -1004
    RFQ_FAILED        = -2000
    RFQ_UNKNOWN       = -2001
    RFQ_INVALID       = -2002
    RFQ_EXPIRED       = -2003
    RFQ_REJECTED      = -2004


@dataclass
class JupiterError(Exception):
    code: int | str
    message: str
    retryable: bool

    def __str__(self) -> str:
        return f"JupiterError(code={self.code}, retryable={self.retryable}): {self.message}"


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class TokenInfo:
    mint: str
    symbol: str
    name: str
    is_sus: bool
    organic_score: float
    verified: bool
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict) -> "TokenInfo":
        audit = data.get("audit", {})
        return cls(
            mint=data.get("address", ""),
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            is_sus=audit.get("isSus", True),   # default True = conservative
            organic_score=data.get("organicScore", 0.0),
            verified=data.get("tags", {}).get("verified", False),
            raw=data,
        )


@dataclass
class TokenPrice:
    mint: str
    price_usd: float | None
    confidence: str  # "high", "medium", "low", or ""
    raw: dict = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        return self.price_usd is not None and self.confidence in ("high", "medium")

    @classmethod
    def from_api(cls, mint: str, data: dict) -> "TokenPrice":
        return cls(
            mint=mint,
            price_usd=data.get("price"),
            confidence=data.get("confidenceLevel", ""),
            raw=data,
        )


@dataclass
class WalletPosition:
    platform: str
    element_type: str  # "multiple", "liquidity", "trade", "leverage", "borrowlend"
    label: str
    value_usd: float
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict) -> "WalletPosition":
        return cls(
            platform=data.get("platformId", ""),
            element_type=data.get("type", ""),
            label=data.get("label", ""),
            value_usd=float(data.get("totalUsd", 0)),
            raw=data,
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class JupiterClient:
    """
    Async Jupiter API client for FENRIR v2.

    Usage:
        async with JupiterClient() as jup:
            token = await jup.get_token_info("So11111111111111111111111111111111111111112")
            price = await jup.get_price(["So11...", "EPjFW..."])
    """

    BASE_URL = "https://api.jup.ag"
    QUOTE_TIMEOUT = 5.0    # seconds
    EXECUTE_TIMEOUT = 30.0 # seconds
    MAX_RETRIES = 3

    def __init__(self) -> None:
        self._api_key = assert_jupiter_auth()
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "JupiterClient":
        self._session = aiohttp.ClientSession(
            headers={"x-api-key": self._api_key, "Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Internal request helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict | None = None, timeout: float = 10.0) -> Any:
        assert self._session, "Use async context manager"
        async with self._session.get(
            f"{self.BASE_URL}{path}",
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            return await self._handle_response(resp)

    async def _post(self, path: str, json: dict | None = None, timeout: float = 30.0) -> Any:
        assert self._session, "Use async context manager"
        async with self._session.post(
            f"{self.BASE_URL}{path}",
            json=json,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            return await self._handle_response(resp)

    async def _handle_response(self, resp: aiohttp.ClientResponse) -> Any:
        if resp.status == 429:
            retry_after = int(resp.headers.get("Retry-After", "10"))
            raise JupiterError(
                code="RATE_LIMITED",
                message=f"Rate limited, retry after {retry_after}s",
                retryable=True,
            )
        if not resp.ok:
            body = {}
            try:
                body = await resp.json()
            except Exception:
                body = {"message": await resp.text()}
            code = body.get("code", resp.status)
            raise JupiterError(
                code=code,
                message=body.get("message", f"HTTP_{resp.status}"),
                retryable=code in RETRYABLE_SWAP_CODES,
            )
        return await resp.json()

    async def _with_retry(self, coro_fn, *args, **kwargs) -> Any:
        """Retry with exponential backoff for retryable Jupiter errors."""
        last_err: JupiterError | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except JupiterError as e:
                last_err = e
                if not e.retryable or attempt == self.MAX_RETRIES:
                    raise
                delay = min(1.0 * (2 ** attempt) + random.random() * 0.5, 10.0)
                await asyncio.sleep(delay)
        raise last_err  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Tokens v2 — used by TokenScorer
    # ------------------------------------------------------------------

    async def get_token_info(self, mint: str) -> TokenInfo | None:
        """
        Fetch token metadata including isSus and organicScore.
        Feed into AI scoring prompt alongside SMC signals.
        """
        try:
            data = await self._get("/tokens/v2/search", params={"query": mint})
            tokens = data if isinstance(data, list) else data.get("tokens", [])
            if not tokens:
                return None
            # Match by mint address
            for t in tokens:
                if t.get("address") == mint:
                    return TokenInfo.from_api(t)
            return TokenInfo.from_api(tokens[0])
        except JupiterError:
            return None

    async def get_trending_tokens(
        self,
        category: str = "toporganicscore",
        interval: str = "1h",
    ) -> list[TokenInfo]:
        """
        Fetch trending/top tokens by category and interval.
        categories: toporganicscore | toptraded | toptrending
        intervals:  5m | 1h | 6h | 24h
        """
        data = await self._get(f"/tokens/v2/{category}/{interval}")
        tokens = data if isinstance(data, list) else data.get("tokens", [])
        return [TokenInfo.from_api(t) for t in tokens]

    # ------------------------------------------------------------------
    # Price v3 — used by PortfolioHeatManager
    # ------------------------------------------------------------------

    async def get_prices(self, mints: list[str]) -> dict[str, TokenPrice]:
        """
        Fetch USD prices for up to 50 mints.
        Always filter on is_reliable before using in exposure calculations.
        """
        if not mints:
            return {}
        # API max 50 per request — chunk if needed
        results: dict[str, TokenPrice] = {}
        for i in range(0, len(mints), 50):
            chunk = mints[i:i + 50]
            data = await self._get("/price/v3", params={"ids": ",".join(chunk)})
            price_data = data.get("data", data)
            for mint in chunk:
                entry = price_data.get(mint)
                if entry:
                    results[mint] = TokenPrice.from_api(mint, entry)
                else:
                    # Missing = unreliable, fail closed
                    results[mint] = TokenPrice(
                        mint=mint, price_usd=None, confidence="", raw={}
                    )
        return results

    async def get_price(self, mint: str) -> TokenPrice:
        prices = await self.get_prices([mint])
        return prices.get(mint, TokenPrice(mint=mint, price_usd=None, confidence="", raw={}))

    # ------------------------------------------------------------------
    # Portfolio v1 — used by WhaleCopyStrategy
    # ------------------------------------------------------------------

    async def get_wallet_positions(self, wallet_address: str) -> list[WalletPosition]:
        """
        Fetch all DeFi positions for a wallet across Jupiter platforms.
        Treat empty list as valid state — not an error.
        Used by WhaleCopyStrategy to track whale wallet activity.
        """
        try:
            data = await self._get(f"/portfolio/v1/positions/{wallet_address}")
            positions = data.get("positions", data if isinstance(data, list) else [])
            return [WalletPosition.from_api(p) for p in positions]
        except JupiterError as e:
            if not e.retryable:
                return []  # Treat as empty — beta API
            raise

    # ------------------------------------------------------------------
    # Swap v2 — used by ExecutionEngine
    # ------------------------------------------------------------------

    async def get_swap_order(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        **kwargs: Any,
    ) -> dict:
        """
        Get a swap order. Returns the order payload to sign and execute.
        NOTE: Signed payloads expire in ~2 min. Never cache across ticks.
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
            **{k: str(v) for k, v in kwargs.items()},
        }
        return await self._with_retry(
            self._get, "/swap/v2/order", params=params, timeout=self.QUOTE_TIMEOUT
        )

    async def execute_swap(
        self,
        signed_transaction: str,
        request_id: str,
    ) -> dict:
        """
        Execute a signed swap transaction.
        Idempotent for up to 2 min with same signedTransaction + requestId.
        Always log requestId for audit chain correlation.
        """
        return await self._with_retry(
            self._post,
            "/swap/v2/execute",
            json={"signedTransaction": signed_transaction, "requestId": request_id},
            timeout=self.EXECUTE_TIMEOUT,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_jupiter_client() -> JupiterClient:
    """Factory — call assert_jupiter_auth() eagerly at FENRIR startup."""
    return JupiterClient()
