#!/usr/bin/env python3
"""
FENRIR - Market Scanner

Multi-tier candidate discovery beyond fresh pump.fun launches. Periodically pulls
Jupiter trending/top token lists, buckets them into mid/large-cap tiers by USD
market cap, applies liquidity/quality filters + a per-mint cooldown, and emits
each survivor to a callback (the bot's AI-evaluation handler).

Low-cap fresh launches still come from PumpFunMonitor; this fills in the mid
(migrated) and large (established) tiers so the AI brain sees the whole market.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.logger import FenrirLogger

__all__ = ["MarketScanner"]


class MarketScanner:
    """Periodic mid/large-cap candidate discovery via Jupiter trending lists."""

    def __init__(self, config: BotConfig, jupiter: JupiterSwapEngine, logger: FenrirLogger):
        self.config = config
        self.jupiter = jupiter
        self.logger = logger
        self.running = False
        self._cooldown: dict[str, datetime] = {}

    async def start_scanning(self, on_candidate: Callable[[dict], Awaitable[None]]) -> None:
        """Loop: scan on a cadence and emit tiered candidates to on_candidate."""
        self.running = True
        self.logger.info(
            f"Market scanner active (every {self.config.scanner_interval_seconds:.0f}s, "
            f"mid>=${self.config.mid_cap_min_usd:,.0f} large>=${self.config.large_cap_min_usd:,.0f})"
        )
        while self.running:
            try:
                await self._scan_once(on_candidate)
            except Exception as e:
                self.logger.error("Market scan error", e)
            await asyncio.sleep(self.config.scanner_interval_seconds)

    async def stop(self) -> None:
        self.running = False

    async def _scan_once(self, on_candidate: Callable[[dict], Awaitable[None]]) -> None:
        now = datetime.now()
        seen: set[str] = set()
        emitted = 0
        for category in self.config.scanner_categories:
            if emitted >= self.config.scanner_max_candidates_per_cycle:
                break
            tokens = await self.jupiter.get_trending_tokens(
                category, self.config.scanner_interval_window
            )
            for tok in tokens:
                if emitted >= self.config.scanner_max_candidates_per_cycle:
                    break
                mint = tok.get("id")
                if not mint or mint in seen:
                    continue
                seen.add(mint)
                candidate = self._evaluate(tok, now)
                if candidate is None:
                    continue
                self._cooldown[mint] = now
                await on_candidate(candidate)
                emitted += 1
        if emitted:
            self.logger.info(f"Scanner emitted {emitted} candidate(s) this cycle")

    def _tier(self, mcap_usd: float) -> str | None:
        if mcap_usd >= self.config.large_cap_min_usd:
            return "large"
        if mcap_usd >= self.config.mid_cap_min_usd:
            return "mid"
        return None

    def _evaluate(self, tok: dict[str, Any], now: datetime) -> dict | None:
        """Apply tier/liquidity/quality/cooldown filters; return a candidate or None."""
        mint = tok.get("id")
        if not mint:
            return None
        last = self._cooldown.get(mint)
        if last and (now - last).total_seconds() < self.config.scanner_cooldown_minutes * 60:
            return None
        mcap = tok.get("mcap") or 0.0
        liquidity = tok.get("liquidity") or 0.0
        tier = self._tier(mcap)
        if tier is None:
            return None
        if liquidity < self.config.scanner_min_liquidity_usd:
            return None
        if self.config.scanner_require_verified and not tok.get("isVerified"):
            return None
        if (
            self.config.scanner_min_organic_score > 0
            and (tok.get("organicScore") or 0) < self.config.scanner_min_organic_score
        ):
            return None
        return self._build_candidate(tok, tier)

    @staticmethod
    def _build_candidate(tok: dict[str, Any], tier: str) -> dict:
        """Shape a Jupiter token entry into the token_data dict the AI/engine use."""
        return {
            "token_address": tok.get("id"),
            "symbol": tok.get("symbol", "???"),
            "name": tok.get("name", "Unknown"),
            "tier": tier,
            "market_cap_usd": tok.get("mcap"),
            "liquidity_usd": tok.get("liquidity"),
            "usd_price": tok.get("usdPrice"),
            "holder_count": tok.get("holderCount"),
            "organic_score": tok.get("organicScore"),
            "decimals": tok.get("decimals", 6),
            "twitter": tok.get("twitter"),
            "telegram": tok.get("telegram"),
            "website": tok.get("website"),
            "migrated": bool(tok.get("graduatedAt")),
            # No pump bonding curve for these — priced via Jupiter/DexScreener.
            "bonding_curve_state": None,
        }
