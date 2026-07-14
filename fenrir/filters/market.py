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
from datetime import datetime
from typing import Any

from fenrir.discovery.models import Chain
from fenrir.discovery.providers.dexscreener import DexScreenerProvider

logger = logging.getLogger("FENRIR.MarketFilter")


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
        # Shared DexScreener fetch/parse — single source of truth for market data
        # (the discovery layer and this trade-gate filter read the same snapshot).
        self._provider = DexScreenerProvider(timeout_seconds=config.fetch_timeout_seconds)

    async def close(self) -> None:
        await self._provider.close()

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
        Fetch token market data via the shared DexScreener provider and map it onto
        the legacy ``MarketData`` view used by the trade-gate filters. Returns the
        best/most-liquid pair found for the token, or None on failure.
        """
        snap = await self._provider.fetch_snapshot(token_address)
        return self._snapshot_to_market_data(token_address, snap) if snap else None

    def _parse_pair(self, token_address: str, pair: dict[str, Any]) -> MarketData:
        """Parse a DexScreener pair into ``MarketData`` (delegates to the provider).

        Kept for backward compatibility; the trade gate is Solana, so the pair is
        parsed under ``Chain.SOLANA`` (the chain only tags the snapshot, not the
        chain-agnostic ``MarketData`` fields).
        """
        snap = self._provider.parse_pair(token_address, Chain.SOLANA, pair)
        return self._snapshot_to_market_data(token_address, snap)

    @staticmethod
    def _snapshot_to_market_data(token_address: str, snap: Any) -> MarketData:
        """Adapt a :class:`~fenrir.discovery.models.TokenSnapshot` to ``MarketData``."""
        price_change = (snap.raw.get("priceChange") or {}) if isinstance(snap.raw, dict) else {}
        return MarketData(
            token_address=token_address,
            pair_address=snap.pair_address,
            dex_id=snap.dex_id,
            price_usd=snap.price_usd,
            liquidity_usd=snap.liquidity_usd,
            market_cap_usd=snap.market_cap_usd,
            fdv_usd=snap.fdv_usd,
            volume_5m_usd=snap.volume_5m_usd,
            volume_1h_usd=snap.volume_1h_usd,
            volume_6h_usd=snap.volume_6h_usd,
            volume_24h_usd=snap.volume_24h_usd,
            txns_5m_buys=snap.txns_5m_buys,
            txns_5m_sells=snap.txns_5m_sells,
            txns_1h_buys=snap.txns_1h_buys,
            txns_1h_sells=snap.txns_1h_sells,
            price_change_5m_pct=snap.price_change_5m_pct,
            price_change_1h_pct=snap.price_change_1h_pct,
            price_change_6h_pct=float(price_change.get("h6") or 0),
            price_change_24h_pct=snap.price_change_24h_pct,
            pair_created_at=snap.created_at,
            age_minutes=snap.age_minutes,
            raw=snap.raw,
        )
