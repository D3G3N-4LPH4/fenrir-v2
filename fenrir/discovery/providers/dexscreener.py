#!/usr/bin/env python3
"""
FENRIR - DexScreener provider (chain-agnostic market data)

Single source of truth for DexScreener fetch + parse. Produces a normalized
:class:`TokenSnapshot` for ANY supported chain (DexScreener tags each pair with a
``chainId``). This replaces the duplicated fetch/parse that previously lived in
``fenrir/filters/market.py`` — that filter now delegates here and maps the snapshot
onto its legacy ``MarketData`` view.

Endpoints (all public, no key):
  - token pairs:  https://api.dexscreener.com/latest/dex/tokens/{address}
  - token boosts: https://api.dexscreener.com/token-boosts/top/v1  (cross-chain)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fenrir.discovery.models import Chain, TokenSnapshot

logger = logging.getLogger("FENRIR.DexScreener")

DEXSCREENER_TOKENS = "https://api.dexscreener.com/latest/dex/tokens"
DEXSCREENER_BOOSTS = "https://api.dexscreener.com/token-boosts/top/v1"


def _f(value: Any) -> float:
    """Best-effort float (DexScreener returns strings/None inconsistently)."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _i(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


class DexScreenerProvider:
    """Fetches and normalizes DexScreener market data into :class:`TokenSnapshot`.

    Stateless apart from a lazily-created aiohttp session; safe to share.
    """

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self.timeout = timeout_seconds
        self._session: Any = None

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(headers={"User-Agent": "FENRIR/2.0 discovery"})
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Public API ────────────────────────────────────────────────────

    async def fetch_snapshot(
        self, token_address: str, chain: Chain | None = None
    ) -> TokenSnapshot | None:
        """Fetch a token's most-liquid pair and normalize it.

        When ``chain`` is given, only pairs on that chain are considered; otherwise
        the most-liquid pair across all chains wins (and its chain is inferred).
        Returns None when the token is unknown or the fetch fails.
        """
        pairs = await self._fetch_pairs(token_address)
        if not pairs:
            return None

        if chain is not None:
            pairs = [p for p in pairs if Chain.from_dexscreener(p.get("chainId")) is chain]
            if not pairs:
                return None

        best = max(pairs, key=lambda p: _f((p.get("liquidity") or {}).get("usd")))
        snap_chain = chain or Chain.from_dexscreener(best.get("chainId"))
        if snap_chain is None:
            return None
        return self.parse_pair(token_address, snap_chain, best)

    async def _fetch_pairs(self, token_address: str) -> list[dict[str, Any]]:
        try:
            session = await self._get_session()
            url = f"{DEXSCREENER_TOKENS}/{token_address}"
            async with session.get(url, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.warning("DexScreener HTTP %d for %s…", resp.status, token_address[:8])
                    return []
                data = await resp.json()
            return list(data.get("pairs") or [])
        except TimeoutError:
            logger.warning("DexScreener timeout for %s…", token_address[:8])
            return []
        except Exception as e:  # noqa: BLE001 - provider must fail-open, never raise
            logger.warning("DexScreener error for %s…: %s", token_address[:8], e)
            return []

    # ── Parsing (the single shared implementation) ────────────────────

    def parse_pair(self, token_address: str, chain: Chain, pair: dict[str, Any]) -> TokenSnapshot:
        """Normalize one DexScreener pair dict into a :class:`TokenSnapshot`."""
        now = datetime.now(UTC)
        created_ms = pair.get("pairCreatedAt")
        created_at: datetime | None = None
        age_minutes = 0.0
        if created_ms:
            created_at = datetime.fromtimestamp(created_ms / 1000, tz=UTC)
            age_minutes = (now - created_at).total_seconds() / 60.0

        volume = pair.get("volume") or {}
        txns = pair.get("txns") or {}
        txns_5m = txns.get("m5") or {}
        txns_1h = txns.get("h1") or {}
        txns_24h = txns.get("h24") or {}
        price_change = pair.get("priceChange") or {}
        liquidity = pair.get("liquidity") or {}
        info = pair.get("info") or {}
        socials = {
            s.get("type"): s.get("url") for s in (info.get("socials") or []) if isinstance(s, dict)
        }
        websites = [w for w in (info.get("websites") or []) if isinstance(w, dict)]

        return TokenSnapshot(
            chain=chain,
            token_address=token_address,
            symbol=(pair.get("baseToken") or {}).get("symbol", "???"),
            name=(pair.get("baseToken") or {}).get("name", "Unknown"),
            pair_address=pair.get("pairAddress"),
            dex_id=pair.get("dexId"),
            price_usd=_f(pair.get("priceUsd")),
            market_cap_usd=_f(pair.get("marketCap")),
            fdv_usd=_f(pair.get("fdv")),
            liquidity_usd=_f(liquidity.get("usd")),
            volume_5m_usd=_f(volume.get("m5")),
            volume_1h_usd=_f(volume.get("h1")),
            volume_6h_usd=_f(volume.get("h6")),
            volume_24h_usd=_f(volume.get("h24")),
            txns_5m_buys=_i(txns_5m.get("buys")),
            txns_5m_sells=_i(txns_5m.get("sells")),
            txns_1h_buys=_i(txns_1h.get("buys")),
            txns_1h_sells=_i(txns_1h.get("sells")),
            txns_24h_buys=_i(txns_24h.get("buys")),
            txns_24h_sells=_i(txns_24h.get("sells")),
            price_change_5m_pct=_f(price_change.get("m5")),
            price_change_1h_pct=_f(price_change.get("h1")),
            price_change_24h_pct=_f(price_change.get("h24")),
            created_at=created_at,
            age_minutes=age_minutes,
            twitter=socials.get("twitter"),
            telegram=socials.get("telegram"),
            website=(websites[0].get("url") if websites else None),
            raw=pair,
        )
