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
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.logger import FenrirLogger
from fenrir.trading.token_filters import is_tradeable_mint

__all__ = ["MarketScanner"]


def _age_minutes(created_at: str | None) -> float | None:
    """Minutes since an ISO-8601 timestamp; None when unknown (never a fake 0)."""
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - created).total_seconds() / 60.0)


# DexScreener boosts = tokens actively paying for visibility (a meme marketing
# signal). The list is chain-agnostic; we keep Solana and enrich each via the
# tokens endpoint for mcap/liquidity/momentum.
DEXSCREENER_BOOSTS_TOP = "https://api.dexscreener.com/token-boosts/top/v1"
DEXSCREENER_TOKENS = "https://api.dexscreener.com/latest/dex/tokens/"


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class MarketScanner:
    """Periodic mid/large-cap candidate discovery via Jupiter trending lists."""

    def __init__(self, config: BotConfig, jupiter: JupiterSwapEngine, logger: FenrirLogger):
        self.config = config
        self.jupiter = jupiter
        self.logger = logger
        self.running = False
        self._cooldown: dict[str, datetime] = {}
        self._dex_session: Any = None  # aiohttp session for DexScreener, lazy

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
        if self._dex_session is not None and not self._dex_session.closed:
            await self._dex_session.close()

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
        # DexScreener boosts — actively-marketed memes (opt-in extra source).
        if self.config.scanner_dex_boosts_enabled:
            emitted += await self._scan_dex_boosts(on_candidate, now, seen, emitted)
        if emitted:
            self.logger.info(f"Scanner emitted {emitted} candidate(s) this cycle")

    # ── DexScreener boosts discovery ──────────────────────────────────

    async def _get_dex_session(self) -> Any:
        if self._dex_session is None or self._dex_session.closed:
            import aiohttp

            self._dex_session = aiohttp.ClientSession()
        return self._dex_session

    async def _scan_dex_boosts(
        self,
        on_candidate: Callable[[dict], Awaitable[None]],
        now: datetime,
        seen: set[str],
        already_emitted: int,
    ) -> int:
        """Fetch boosted Solana tokens, enrich + tier them, emit survivors."""
        cap = self.config.scanner_max_candidates_per_cycle
        emitted = 0
        try:
            boosts = await self._fetch_dex_boosts()
        except Exception as e:
            self.logger.error("DexScreener boosts fetch failed", e)
            return 0
        for boost in boosts:
            if already_emitted + emitted >= cap:
                break
            mint = boost.get("tokenAddress")
            if not mint or mint in seen:
                continue
            seen.add(mint)
            pair = await self._fetch_dex_pair(mint)
            if pair is None:
                continue
            candidate = self._evaluate_dex(boost, pair, now)
            if candidate is None:
                continue
            self._cooldown[mint] = now
            await on_candidate(candidate)
            emitted += 1
        return emitted

    async def _fetch_dex_boosts(self) -> list[dict]:
        """Return the top boosted tokens on Solana (or [] on error)."""
        session = await self._get_dex_session()
        async with session.get(
            DEXSCREENER_BOOSTS_TOP, timeout=self.config.scanner_dex_timeout_seconds
        ) as resp:
            if resp.status != 200:
                self.logger.warning(f"DexScreener boosts HTTP {resp.status}")
                return []
            data = await resp.json()
        if not isinstance(data, list):
            return []
        return [b for b in data if isinstance(b, dict) and b.get("chainId") == "solana"]

    async def _fetch_dex_pair(self, address: str) -> dict | None:
        """Fetch a token's deepest-liquidity DexScreener pair (or None)."""
        session = await self._get_dex_session()
        url = f"{DEXSCREENER_TOKENS}{address}"
        async with session.get(url, timeout=self.config.scanner_dex_timeout_seconds) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
        pairs = data.get("pairs") if isinstance(data, dict) else None
        if not pairs:
            return None
        best: dict = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0.0)
        return best

    def _evaluate_dex(self, boost: dict, pair: dict, now: datetime) -> dict | None:
        """Apply cooldown/tier/liquidity filters to a boosted token."""
        mint = boost.get("tokenAddress")
        if not mint:
            return None
        if not is_tradeable_mint(mint):
            return None  # stablecoin / WSOL / LST — not a swing target
        last = self._cooldown.get(mint)
        if last and (now - last).total_seconds() < self.config.scanner_cooldown_minutes * 60:
            return None
        mcap = _to_float(pair.get("marketCap")) or _to_float(pair.get("fdv")) or 0.0
        liquidity = _to_float((pair.get("liquidity") or {}).get("usd")) or 0.0
        tier = self._tier(mcap)
        if tier is None:
            return None
        if liquidity < self.config.scanner_min_liquidity_usd:
            return None
        return self._build_dex_candidate(boost, pair, tier)

    @staticmethod
    def _build_dex_candidate(boost: dict, pair: dict, tier: str) -> dict:
        """Shape a boosted token + its pair into the token_data dict the AI/engine use.

        The dex_* keys mirror what the bot attaches on the launch path so the AI's
        market-signal context works for scanner candidates too. decimals defaults
        to 6: boosted memes are overwhelmingly pump.fun-origin (always 6-decimal),
        and the engine falls back to 6 when unset.
        """
        base = pair.get("baseToken") or {}
        vol = pair.get("volume") or {}
        txns_5m = (pair.get("txns") or {}).get("m5") or {}
        price_change = pair.get("priceChange") or {}
        liquidity = _to_float((pair.get("liquidity") or {}).get("usd"))
        buys = int(txns_5m.get("buys") or 0)
        sells = int(txns_5m.get("sells") or 0)
        total = buys + sells
        return {
            "token_address": boost.get("tokenAddress"),
            "symbol": base.get("symbol", "???"),
            "name": base.get("name", "Unknown"),
            "tier": tier,
            "market_cap_usd": _to_float(pair.get("marketCap")) or _to_float(pair.get("fdv")),
            "liquidity_usd": liquidity,
            "usd_price": _to_float(pair.get("priceUsd")),
            "decimals": 6,
            "source": "dexscreener_boost",
            "boost_amount": boost.get("totalAmount"),
            "migrated": True,  # boosted tokens trade on an AMM, not a pump curve
            "bonding_curve_state": None,
            # DexScreener momentum → AI decision context (read by brain).
            "dex_volume_5m_usd": _to_float(vol.get("m5")) or 0.0,
            "dex_txns_5m_buys": buys,
            "dex_txns_5m_sells": sells,
            "dex_buy_pressure_5m": (buys / total) if total > 0 else 0.5,
            "dex_price_change_1h_pct": _to_float(price_change.get("h1")) or 0.0,
            "dex_liquidity_usd": liquidity or 0.0,
        }

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
        if not is_tradeable_mint(mint):
            return None  # stablecoin / WSOL / LST — not a swing target
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
        s1 = tok.get("stats1h") or {}
        s24 = tok.get("stats24h") or {}
        audit = tok.get("audit") or {}
        vol_24h = (s24.get("buyVolume") or 0) + (s24.get("sellVolume") or 0)
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
            "organic_score_label": tok.get("organicScoreLabel"),
            "decimals": tok.get("decimals", 6),
            "twitter": tok.get("twitter"),
            "telegram": tok.get("telegram"),
            "website": tok.get("website"),
            "migrated": bool(tok.get("graduatedAt")),
            # Momentum signals for the AI/panel momentum lens (Jupiter stats).
            "price_change_1h": s1.get("priceChange"),
            "price_change_24h": s24.get("priceChange"),
            "volume_24h_usd": vol_24h,
            "num_buys_24h": s24.get("numBuys"),
            "num_sells_24h": s24.get("numSells"),
            "top_holders_pct": audit.get("topHoldersPercentage"),
            # Creator track record from Jupiter's audit block. A wallet with thousands
            # of mints is a token factory — a real risk signal the AI should weigh.
            "dev_mints": audit.get("devMints"),
            "dev_migrations": audit.get("devMigrations"),
            # Real age, so the AI doesn't guess at timing (it called a 15-day-old
            # token "a fresh launch" when it had no age field at all).
            "age_minutes": _age_minutes(
                tok.get("createdAt") or (tok.get("firstPool") or {}).get("createdAt")
            ),
            # No pump bonding curve for these — priced via Jupiter/DexScreener.
            "bonding_curve_state": None,
        }
