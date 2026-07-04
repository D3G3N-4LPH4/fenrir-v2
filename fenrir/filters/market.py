#!/usr/bin/env python3
"""
FENRIR - Market Condition Filters

Two-tier market scanning with different criteria per token age:

  Tier 1 — Live Launch (0-60 min):   pump.fun migrations + new Raydium pools
  Tier 2 — Momentum Breakout (1-24h): tokens that survived the launch dump

Data source: DexScreener public API (free, no API key required).
Endpoint: https://api.dexscreener.com/latest/dex/tokens/{address}

All USD thresholds are configurable via FilterConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("FENRIR.MarketFilter")

DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/tokens"


@dataclass
class MarketData:
    """Snapshot of market conditions for a token from DexScreener."""

    token_address: str
    pair_address: str | None = None
    dex_id: str | None = None  # "raydium", "pumpfun", etc.

    # Pricing
    price_usd: float = 0.0
    price_sol: float = 0.0

    # Liquidity
    liquidity_usd: float = 0.0

    # Market cap
    market_cap_usd: float = 0.0
    fdv_usd: float = 0.0

    # Volume
    volume_5m_usd: float = 0.0
    volume_1h_usd: float = 0.0
    volume_6h_usd: float = 0.0
    volume_24h_usd: float = 0.0

    # Transactions
    txns_5m_buys: int = 0
    txns_5m_sells: int = 0
    txns_1h_buys: int = 0
    txns_1h_sells: int = 0

    # Unique buyers (not always available from DexScreener)
    unique_buyers_1h: int = 0

    # Price changes
    price_change_5m_pct: float = 0.0
    price_change_1h_pct: float = 0.0
    price_change_6h_pct: float = 0.0
    price_change_24h_pct: float = 0.0

    # Token age
    pair_created_at: datetime | None = None
    age_minutes: float = 0.0

    # Raw data for AI context
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def txns_5m_total(self) -> int:
        return self.txns_5m_buys + self.txns_5m_sells

    @property
    def txns_1h_total(self) -> int:
        return self.txns_1h_buys + self.txns_1h_sells

    @property
    def buy_pressure_5m(self) -> float:
        """Buy/sell ratio over 5 minutes. >0.5 = more buys than sells."""
        total = self.txns_5m_total
        return self.txns_5m_buys / total if total > 0 else 0.5


@dataclass
class MarketFilterResult:
    """Result of market filter evaluation."""

    passed: bool
    tier: str  # "live_launch", "momentum_breakout", or "none"
    token_address: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    market_data: MarketData | None = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        tier_str = f"[{self.tier}]" if self.tier != "none" else ""
        if self.failures:
            return f"[{status}]{tier_str} {self.token_address[:8]}... — {', '.join(self.failures)}"
        return f"[{status}]{tier_str} {self.token_address[:8]}..."


@dataclass
class LiveLaunchFilterConfig:
    """Filters for tokens 0-60 minutes old (Tier 1)."""

    max_age_minutes: float = 60.0
    min_liquidity_usd: float = 10_000.0  # $10k minimum LP
    min_market_cap_usd: float = 30_000.0  # $30k minimum mcap
    max_market_cap_usd: float = 150_000.0  # $150k maximum mcap
    min_volume_5m_usd: float = 15_000.0  # $15k volume in last 5 min
    min_txns_5m: int = 100  # 100 trades in last 5 min


@dataclass
class MomentumBreakoutFilterConfig:
    """Filters for tokens 1-24 hours old (Tier 2)."""

    min_age_minutes: float = 60.0
    max_age_minutes: float = 1440.0  # 24 hours
    min_liquidity_usd: float = 50_000.0  # $50k minimum LP
    min_market_cap_usd: float = 200_000.0  # $200k minimum mcap
    max_market_cap_usd: float = 1_500_000.0  # $1.5M maximum mcap
    min_volume_1h_usd: float = 100_000.0  # $100k volume in last hour
    min_unique_buyers_1h: int = 300  # 300 unique buyers in last hour


@dataclass
class MarketFilterConfig:
    """Combined market filter configuration."""

    enabled: bool = True
    # Allow pass-through if DexScreener is unreachable
    fail_open_on_fetch_error: bool = True
    # Fetch timeout
    fetch_timeout_seconds: float = 5.0
    # Which tiers to evaluate
    live_launch_enabled: bool = True
    momentum_breakout_enabled: bool = True

    live_launch: LiveLaunchFilterConfig = field(default_factory=LiveLaunchFilterConfig)
    momentum: MomentumBreakoutFilterConfig = field(default_factory=MomentumBreakoutFilterConfig)


class MarketFilter:
    """
    Two-tier market condition filter using DexScreener data.

    Usage:
        mf = MarketFilter(config)
        result = await mf.check(token_address)
        if result.passed:
            # inject result.market_data into AI context
    """

    def __init__(self, config: MarketFilterConfig) -> None:
        self.config = config
        self._session: Any = None

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(headers={"User-Agent": "FENRIR/2.0 trading-bot"})
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Public entry point ────────────────────────────────────────────

    async def check(self, token_address: str) -> MarketFilterResult:
        """
        Fetch market data from DexScreener and apply tier-appropriate filters.
        Returns the result including full MarketData for AI context injection.
        """
        if not self.config.enabled:
            return MarketFilterResult(passed=True, tier="disabled", token_address=token_address)

        market_data = await self._fetch_market_data(token_address)

        if market_data is None:
            if self.config.fail_open_on_fetch_error:
                return MarketFilterResult(
                    passed=True,
                    tier="none",
                    token_address=token_address,
                    warnings=["DexScreener unavailable — market filters skipped"],
                )
            return MarketFilterResult(
                passed=False,
                tier="none",
                token_address=token_address,
                failures=["Could not fetch market data from DexScreener"],
            )

        # Determine which tier applies based on token age
        age = market_data.age_minutes
        result = MarketFilterResult(
            passed=False,
            tier="none",
            token_address=token_address,
            market_data=market_data,
        )

        if self.config.live_launch_enabled and age <= self.config.live_launch.max_age_minutes:
            result.tier = "live_launch"
            self._apply_live_launch_filters(result, market_data)

        elif (
            self.config.momentum_breakout_enabled
            and self.config.momentum.min_age_minutes <= age <= self.config.momentum.max_age_minutes
        ):
            result.tier = "momentum_breakout"
            self._apply_momentum_filters(result, market_data)

        else:
            # Token age doesn't fit either tier
            result.passed = False
            result.failures.append(
                f"Token age {age:.0f}m outside filter windows "
                f"(live: 0-{self.config.live_launch.max_age_minutes:.0f}m, "
                f"momentum: {self.config.momentum.min_age_minutes:.0f}-"
                f"{self.config.momentum.max_age_minutes:.0f}m)"
            )

        return result

    # ── Tier filter application ───────────────────────────────────────

    def _apply_live_launch_filters(self, result: MarketFilterResult, md: MarketData) -> None:
        cfg = self.config.live_launch
        failures: list[str] = []

        if md.liquidity_usd < cfg.min_liquidity_usd:
            failures.append(f"LP ${md.liquidity_usd:,.0f} < min ${cfg.min_liquidity_usd:,.0f}")
        if md.market_cap_usd < cfg.min_market_cap_usd:
            failures.append(f"MCap ${md.market_cap_usd:,.0f} < min ${cfg.min_market_cap_usd:,.0f}")
        if md.market_cap_usd > cfg.max_market_cap_usd:
            failures.append(f"MCap ${md.market_cap_usd:,.0f} > max ${cfg.max_market_cap_usd:,.0f}")
        if md.volume_5m_usd < cfg.min_volume_5m_usd:
            failures.append(f"Vol(5m) ${md.volume_5m_usd:,.0f} < min ${cfg.min_volume_5m_usd:,.0f}")
        if md.txns_5m_total < cfg.min_txns_5m:
            failures.append(f"Txns(5m) {md.txns_5m_total} < min {cfg.min_txns_5m}")

        result.failures = failures
        result.passed = len(failures) == 0

    def _apply_momentum_filters(self, result: MarketFilterResult, md: MarketData) -> None:
        cfg = self.config.momentum
        failures: list[str] = []

        if md.liquidity_usd < cfg.min_liquidity_usd:
            failures.append(f"LP ${md.liquidity_usd:,.0f} < min ${cfg.min_liquidity_usd:,.0f}")
        if md.market_cap_usd < cfg.min_market_cap_usd:
            failures.append(f"MCap ${md.market_cap_usd:,.0f} < min ${cfg.min_market_cap_usd:,.0f}")
        if md.market_cap_usd > cfg.max_market_cap_usd:
            failures.append(f"MCap ${md.market_cap_usd:,.0f} > max ${cfg.max_market_cap_usd:,.0f}")
        if md.volume_1h_usd < cfg.min_volume_1h_usd:
            failures.append(f"Vol(1h) ${md.volume_1h_usd:,.0f} < min ${cfg.min_volume_1h_usd:,.0f}")
        if md.unique_buyers_1h < cfg.min_unique_buyers_1h:
            # unique_buyers_1h isn't always available from DexScreener
            # fall back to txns_1h as a proxy if not available
            if md.txns_1h_buys >= cfg.min_unique_buyers_1h:
                result.warnings.append(
                    f"Unique buyers unavailable — using buy txns ({md.txns_1h_buys}) as proxy"
                )
            else:
                failures.append(
                    f"Buyers(1h) {md.unique_buyers_1h} < min {cfg.min_unique_buyers_1h}"
                )

        result.failures = failures
        result.passed = len(failures) == 0

    # ── DexScreener fetch ─────────────────────────────────────────────

    async def _fetch_market_data(self, token_address: str) -> MarketData | None:
        """
        Fetch token market data from DexScreener public API.
        Returns the best/most liquid pair found for the token.
        """
        try:
            session = await self._get_session()
            url = f"{DEXSCREENER_API}/{token_address}"

            async with session.get(url, timeout=self.config.fetch_timeout_seconds) as resp:
                if resp.status != 200:
                    logger.warning(f"DexScreener returned {resp.status} for {token_address[:8]}...")
                    return None
                data = await resp.json()

            pairs: list[dict[str, Any]] = data.get("pairs") or []
            if not pairs:
                logger.debug(f"No pairs found on DexScreener for {token_address[:8]}...")
                return None

            # Pick the pair with the highest liquidity (most reliable data)
            best_pair = max(
                pairs,
                key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0),
            )

            return self._parse_pair(token_address, best_pair)

        except TimeoutError:
            logger.warning(f"DexScreener timeout for {token_address[:8]}...")
            return None
        except Exception as e:
            logger.warning(f"DexScreener fetch error for {token_address[:8]}...: {e}")
            return None

    def _parse_pair(self, token_address: str, pair: dict[str, Any]) -> MarketData:
        """Parse a DexScreener pair response into a MarketData object."""
        now = datetime.now(UTC)

        # Parse creation time and calculate age
        created_at_ms: int | None = pair.get("pairCreatedAt")
        pair_created_at: datetime | None = None
        age_minutes: float = 0.0

        if created_at_ms:
            pair_created_at = datetime.fromtimestamp(created_at_ms / 1000, tz=UTC)
            age_minutes = (now - pair_created_at).total_seconds() / 60.0

        # Volume
        volume = pair.get("volume", {}) or {}
        # Transactions
        txns = pair.get("txns", {}) or {}
        txns_5m = txns.get("m5", {}) or {}
        txns_1h = txns.get("h1", {}) or {}
        # Price changes
        price_change = pair.get("priceChange", {}) or {}
        # Liquidity
        liquidity = pair.get("liquidity", {}) or {}

        return MarketData(
            token_address=token_address,
            pair_address=pair.get("pairAddress"),
            dex_id=pair.get("dexId"),
            price_usd=float(pair.get("priceUsd") or 0),
            liquidity_usd=float(liquidity.get("usd") or 0),
            market_cap_usd=float(pair.get("marketCap") or 0),
            fdv_usd=float(pair.get("fdv") or 0),
            volume_5m_usd=float(volume.get("m5") or 0),
            volume_1h_usd=float(volume.get("h1") or 0),
            volume_6h_usd=float(volume.get("h6") or 0),
            volume_24h_usd=float(volume.get("h24") or 0),
            txns_5m_buys=int(txns_5m.get("buys") or 0),
            txns_5m_sells=int(txns_5m.get("sells") or 0),
            txns_1h_buys=int(txns_1h.get("buys") or 0),
            txns_1h_sells=int(txns_1h.get("sells") or 0),
            price_change_5m_pct=float(price_change.get("m5") or 0),
            price_change_1h_pct=float(price_change.get("h1") or 0),
            price_change_6h_pct=float(price_change.get("h6") or 0),
            price_change_24h_pct=float(price_change.get("h24") or 0),
            pair_created_at=pair_created_at,
            age_minutes=age_minutes,
            raw=pair,
        )
