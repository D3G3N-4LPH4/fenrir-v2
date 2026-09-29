#!/usr/bin/env python3
"""
FENRIR - GeckoTerminal discovery provider (trending + new pools).

Public, keyless endpoints:
  - trending pools: https://api.geckoterminal.com/api/v2/networks/{network}/trending_pools
  - new pools:      https://api.geckoterminal.com/api/v2/networks/{network}/new_pools

Each pool carries ``relationships.base_token.data.id`` shaped like
``"{network}_{token_address}"`` — the base token is the listed asset, the quote
side is the chain's gas/stable. We return base-token addresses only, skipping
well-known quote assets, deduplicated in feed order.

GeckoTerminal indexes both Solana and Robinhood chain (network id "robinhood").
Fail-open everywhere: any fetch/parse problem yields [].
"""

from __future__ import annotations

import logging
from typing import Any

from fenrir.discovery.models import Chain

logger = logging.getLogger("FENRIR.GeckoTerminal")

GECKO_BASE = "https://api.geckoterminal.com/api/v2"

# Chain -> GeckoTerminal network id. Chains absent here are unsupported.
_GECKO_NETWORKS: dict[Chain, str] = {
    Chain.SOLANA: "solana",
    Chain.ROBINHOOD: "robinhood",
}

# Well-known quote-side assets to never treat as discovery candidates.
# (Robinhood chain's wrapped gas token is intentionally left to the
# snapshot/filter stage — unknown address, and filters reject gas tokens anyway.)
_QUOTE_DENYLIST: dict[Chain, set[str]] = {
    Chain.SOLANA: {
        "So11111111111111111111111111111111111111112",  # wSOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNY",  # USDT
        "mSoLzYCxHdYgdzU16g5Qny1pM6rtShwDKTxsM3n8UBKk",  # mSOL
    },
    Chain.ROBINHOOD: set(),
}


def extract_pool_token_addresses(payload: Any, network: str, chain: Chain) -> list[str]:
    """Pure parser: pool-list payload -> base-token addresses (feed order, deduped).

    Split out for unit testing; never raises on malformed input.
    """
    try:
        pools = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(pools, list):
            return []
        deny = _QUOTE_DENYLIST.get(chain, set())
        prefix = f"{network}_"
        seen: set[str] = set()
        out: list[str] = []
        for pool in pools:
            if not isinstance(pool, dict):
                continue
            try:
                token_id = (
                    (pool.get("relationships") or {})
                    .get("base_token", {})
                    .get("data", {})
                    .get("id")
                )
            except AttributeError:
                continue
            if not isinstance(token_id, str) or not token_id.startswith(prefix):
                continue
            addr = token_id[len(prefix):]
            if not addr or addr in deny or addr in seen:
                continue
            seen.add(addr)
            out.append(addr)
        return out
    except Exception:  # noqa: BLE001 - parser must never raise
        return []


class GeckoTerminalProvider:
    """Fetches GeckoTerminal trending/new pools and extracts token addresses.

    Stateless apart from a lazily-created aiohttp session; safe to share.
    """

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self.timeout = timeout_seconds
        self._session: Any = None

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(trust_env=True, headers={"User-Agent": "FENRIR/2.0 discovery"})
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_trending_addresses(self, chain: Chain, limit: int = 20) -> list[str]:
        """Tokens behind the chain's currently trending pools (momentum)."""
        return await self._fetch_feed(chain, "trending_pools", limit)

    async def fetch_new_pool_addresses(self, chain: Chain, limit: int = 20) -> list[str]:
        """Tokens behind the chain's newest pools (earliest post-launch listings)."""
        return await self._fetch_feed(chain, "new_pools", limit)

    async def _fetch_feed(self, chain: Chain, feed: str, limit: int) -> list[str]:
        network = _GECKO_NETWORKS.get(chain)
        if network is None:
            return []
        try:
            session = await self._get_session()
            url = f"{GECKO_BASE}/networks/{network}/{feed}"
            async with session.get(url, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.warning("GeckoTerminal HTTP %d for %s/%s", resp.status, network, feed)
                    return []
                payload = await resp.json()
        except Exception as e:  # noqa: BLE001 - provider must fail-open, never raise
            logger.warning("GeckoTerminal %s/%s failed: %s", network, feed, e)
            return []
        return extract_pool_token_addresses(payload, network, chain)[:limit]
