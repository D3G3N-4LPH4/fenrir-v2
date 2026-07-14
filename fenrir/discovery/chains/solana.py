#!/usr/bin/env python3
"""
FENRIR - Solana discovery adapter

Reuses existing Solana infrastructure:
  - `JupiterSwapEngine.get_trending_tokens` (`fenrir/core/jupiter.py`) — keyless
    trending feed carrying mcap/liquidity/holders/volume/socials/verified/graduated.
  - DexScreener (`DexScreenerProvider`) — pair age + 5m/1h intervals + dex/pair id.
  - RugCheck summary (same keyless endpoint as `fenrir/filters/security.py`) —
    mint/freeze/LP-lock + risk score → :class:`SafetySignals`.

Pure mappers (`snapshot_from_jupiter`, `map_rugcheck_summary`) are unit-testable
without any network.
"""

from __future__ import annotations

import logging
from typing import Any

from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.discovery.providers.dexscreener import DexScreenerProvider

logger = logging.getLogger("FENRIR.SolanaAdapter")

# Same keyless RugCheck host used by fenrir/filters/security.py.
RUGCHECK_SUMMARY = "https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary"
# LP considered "locked/burned" at/above this % (RugCheck lpLockedPct).
LP_LOCKED_MIN_PCT = 90.0


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def snapshot_from_jupiter(tok: dict[str, Any]) -> TokenSnapshot:
    """Map a Jupiter Tokens-v2 entry to a :class:`TokenSnapshot` (pure)."""
    s24 = tok.get("stats24h") or {}
    s1 = tok.get("stats1h") or {}
    audit = tok.get("audit") or {}
    vol_24h = (s24.get("buyVolume") or 0) + (s24.get("sellVolume") or 0)
    vol_1h = (s1.get("buyVolume") or 0) + (s1.get("sellVolume") or 0)
    verified = tok.get("isVerified")

    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address=tok.get("id", ""),
        symbol=tok.get("symbol", "???"),
        name=tok.get("name", "Unknown"),
        price_usd=float(tok.get("usdPrice") or 0),
        market_cap_usd=float(tok.get("mcap") or 0),
        fdv_usd=float(tok.get("fdv") or 0),
        liquidity_usd=float(tok.get("liquidity") or 0),
        volume_1h_usd=float(vol_1h),
        volume_24h_usd=float(vol_24h),
        txns_1h_buys=int(s1.get("numBuys") or 0),
        txns_1h_sells=int(s1.get("numSells") or 0),
        txns_24h_buys=int(s24.get("numBuys") or 0),
        txns_24h_sells=int(s24.get("numSells") or 0),
        price_change_1h_pct=float(s1.get("priceChange") or 0),
        price_change_24h_pct=float(s24.get("priceChange") or 0),
        holder_count=(int(tok["holderCount"]) if tok.get("holderCount") is not None else None),
        top_holder_pct=_f(audit.get("topHoldersPercentage")),
        migrated=bool(tok.get("graduatedAt")),
        organic_score=_f(tok.get("organicScore")),
        twitter=tok.get("twitter"),
        telegram=tok.get("telegram"),
        website=tok.get("website"),
        safety=SafetySignals(contract_verified=bool(verified) if verified is not None else None),
        raw={"jupiter": tok},
    )


def map_rugcheck_summary(summary: dict[str, Any]) -> SafetySignals:
    """Map a RugCheck report-summary dict to :class:`SafetySignals` (pure).

    RugCheck's ``risks`` list flags PROBLEMS; absence of a mint/freeze-authority
    risk implies that authority is revoked. ``score_normalised`` is lower=safer.
    """
    risks = summary.get("risks") or []
    names = " ".join(str(r.get("name", "")).lower() for r in risks)
    lp_locked_pct = _f(summary.get("lpLockedPct"))
    return SafetySignals(
        mint_disabled=("mint authority" not in names),
        freeze_disabled=("freeze authority" not in names),
        lp_locked_or_burned=(
            lp_locked_pct >= LP_LOCKED_MIN_PCT if lp_locked_pct is not None else None
        ),
        lp_locked_pct=lp_locked_pct,
        risk_score=_f(summary.get("score_normalised")),
        risk_flags=[str(r.get("name", "?")) for r in risks],
    )


class SolanaAdapter:
    """Discovery adapter for Solana."""

    chain: Chain = Chain.SOLANA

    def __init__(
        self,
        dexscreener: DexScreenerProvider,
        jupiter: Any = None,  # JupiterSwapEngine (duck-typed: get_trending_tokens)
        categories: list[str] | None = None,
        interval: str = "24h",
        rugcheck_enabled: bool = True,
        rugcheck_timeout_seconds: float = 4.0,
    ) -> None:
        self.dexscreener = dexscreener
        self.jupiter = jupiter
        self.categories = categories or ["toptrending", "toporganicscore"]
        self.interval = interval
        self.rugcheck_enabled = rugcheck_enabled
        self.rugcheck_timeout = rugcheck_timeout_seconds
        self._session: Any = None

    async def discover(self) -> list[TokenSnapshot]:
        if self.jupiter is None:
            return []
        seen: set[str] = set()
        out: list[TokenSnapshot] = []
        for category in self.categories:
            try:
                toks = await self.jupiter.get_trending_tokens(category, self.interval)
            except Exception as e:  # noqa: BLE001 - one feed failing shouldn't sink discovery
                logger.warning("Jupiter trending %s failed: %s", category, e)
                continue
            for tok in toks:
                addr = tok.get("id")
                if not addr or addr in seen:
                    continue
                seen.add(addr)
                out.append(snapshot_from_jupiter(tok))
        return out

    async def enrich(self, snap: TokenSnapshot) -> None:
        # DexScreener: age + 5m/1h intervals + dex/pair id (fill what Jupiter lacks).
        market = await self.dexscreener.fetch_snapshot(snap.token_address, Chain.SOLANA)
        if market is not None:
            snap.pair_address = snap.pair_address or market.pair_address
            snap.dex_id = snap.dex_id or market.dex_id
            snap.created_at = market.created_at
            snap.age_minutes = market.age_minutes
            snap.volume_5m_usd = snap.volume_5m_usd or market.volume_5m_usd
            snap.txns_5m_buys = snap.txns_5m_buys or market.txns_5m_buys
            snap.txns_5m_sells = snap.txns_5m_sells or market.txns_5m_sells
            snap.price_change_5m_pct = snap.price_change_5m_pct or market.price_change_5m_pct
            if not snap.liquidity_usd:
                snap.liquidity_usd = market.liquidity_usd
            if not snap.market_cap_usd:
                snap.market_cap_usd = market.market_cap_usd

        # RugCheck: mint/freeze/LP + risk score → safety (keep any verified flag).
        if self.rugcheck_enabled:
            summary = await self._fetch_rugcheck(snap.token_address)
            if summary is not None:
                mapped = map_rugcheck_summary(summary)
                mapped.contract_verified = snap.safety.contract_verified
                snap.safety = mapped
                snap.raw["rugcheck"] = summary

    async def _fetch_rugcheck(self, mint: str) -> dict[str, Any] | None:
        try:
            if self._session is None or self._session.closed:
                import aiohttp

                self._session = aiohttp.ClientSession()
            url = RUGCHECK_SUMMARY.format(mint=mint)
            async with self._session.get(url, timeout=self.rugcheck_timeout) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data if isinstance(data, dict) else None
        except Exception as e:  # noqa: BLE001 - fail-open like the security filter
            logger.debug("RugCheck fetch failed for %s…: %s", mint[:8], e)
            return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
