#!/usr/bin/env python3
"""
FENRIR - Discovery scanner service

Orchestrates the discovery pipeline across enabled chains:

    adapter.discover() → adapter.enrich() → FilterEngine → ScoringEngine → rank

Ranked results are cached for the API and high-scoring hits are emitted on the
EventBus as ``DISCOVERY`` alerts (per-token cooldown). Discovery-only — nothing
here trades; it surfaces + scores + alerts across Solana/ETH/BNB/Base.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.discovery.chains.base import ChainAdapter
from fenrir.discovery.config import DiscoveryConfig
from fenrir.discovery.filters import FilterEngine
from fenrir.discovery.models import Chain, ScoreBreakdown, TokenSnapshot
from fenrir.discovery.scoring import ScoringEngine

logger = logging.getLogger("FENRIR.Discovery")


@dataclass
class DiscoveryResult:
    """One token that matched at least one active filter, with its scores."""

    snapshot: TokenSnapshot
    matched_filters: list[str]
    score: ScoreBreakdown
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        s = self.snapshot
        return {
            "chain": s.chain.value,
            "token_address": s.token_address,
            "symbol": s.symbol,
            "name": s.name,
            "market_cap_usd": s.market_cap_usd,
            "liquidity_usd": s.liquidity_usd,
            "volume_24h_usd": s.volume_24h_usd,
            "holder_count": s.holder_count,
            "age_minutes": round(s.age_minutes, 1),
            "dex_id": s.dex_id,
            "matched_filters": self.matched_filters,
            "scores": self.score.as_dict(),
            "socials": {"twitter": s.twitter, "telegram": s.telegram, "website": s.website},
            "detected_at": self.detected_at.isoformat(),
        }


class DiscoveryScanner:
    """Periodic multi-chain discovery + scoring service."""

    def __init__(
        self,
        config: DiscoveryConfig,
        adapters: dict[Chain, ChainAdapter],
        event_bus: Any = None,
    ) -> None:
        self.config = config
        self.adapters = adapters
        self.event_bus = event_bus
        self.filter_engine = FilterEngine(config.thresholds, config.universal_safety)
        self.scoring_engine = ScoringEngine(config.weights)
        self.latest: list[DiscoveryResult] = []
        self._running = False
        self._alerted: dict[str, datetime] = {}

    # ── Public API ────────────────────────────────────────────────────

    async def scan_once(self) -> list[DiscoveryResult]:
        """Run one discovery cycle over enabled chains; update + return ranked results."""
        candidates: list[TokenSnapshot] = []
        for chain in self.config.chains:
            adapter = self.adapters.get(chain)
            if adapter is None:
                continue  # chain adapter not implemented yet (EVM in later PRs)
            try:
                candidates.extend(await adapter.discover())
            except Exception as e:  # noqa: BLE001 - one chain failing shouldn't sink the cycle
                logger.warning("discover() failed for %s: %s", chain.value, e)

        results: list[DiscoveryResult] = []
        for snap in candidates[: self.config.max_candidates_per_cycle]:
            adapter = self.adapters.get(snap.chain)
            if adapter is not None:
                try:
                    await adapter.enrich(snap)
                except Exception as e:  # noqa: BLE001 - enrichment is best-effort
                    logger.debug("enrich() failed for %s…: %s", snap.token_address[:8], e)
            matched = [
                f.value for f in self.config.filters if self.filter_engine.evaluate(snap, f).passed
            ]
            if not matched:
                continue
            results.append(DiscoveryResult(snap, matched, self.scoring_engine.score(snap)))

        results.sort(key=lambda r: r.score.overall, reverse=True)
        self.latest = results
        await self._emit_alerts(results)
        logger.info(
            "Discovery cycle: %d candidates → %d matched (%d chains)",
            len(candidates),
            len(results),
            sum(1 for c in self.config.chains if self.adapters.get(c) is not None),
        )
        return results

    async def start_scanning(self) -> None:
        """Loop ``scan_once`` on the configured cadence until :meth:`stop`."""
        self._running = True
        logger.info(
            "Discovery scanner active (every %.0fs, chains=%s, filters=%s)",
            self.config.interval_seconds,
            [c.value for c in self.config.chains],
            [f.value for f in self.config.filters],
        )
        while self._running:
            try:
                await self.scan_once()
            except Exception as e:  # noqa: BLE001 - never let the loop die
                logger.error("Discovery cycle error: %s", e)
            await asyncio.sleep(self.config.interval_seconds)

    async def stop(self) -> None:
        self._running = False
        for adapter in self.adapters.values():
            try:
                await adapter.close()
            except Exception as e:  # noqa: BLE001 - shutdown must not raise
                logger.debug("adapter close failed: %s", e)

    def get_results(
        self,
        chain: Chain | None = None,
        filter_name: str | None = None,
        min_score: float = 0.0,
        limit: int = 100,
    ) -> list[DiscoveryResult]:
        """Filtered view of the latest ranked results (for the API)."""
        out = self.latest
        if chain is not None:
            out = [r for r in out if r.snapshot.chain is chain]
        if filter_name is not None:
            out = [r for r in out if filter_name in r.matched_filters]
        out = [r for r in out if r.score.overall >= min_score]
        return out[:limit]

    # ── Alerts ────────────────────────────────────────────────────────

    async def _emit_alerts(self, results: list[DiscoveryResult]) -> None:
        if self.event_bus is None:
            return
        from fenrir.events.types import discovery_event

        now = datetime.now(UTC)
        cooldown = self.config.cooldown_minutes * 60
        for r in results:
            if r.score.overall < self.config.min_alert_score:
                continue
            addr = r.snapshot.token_address
            last = self._alerted.get(addr)
            if last is not None and (now - last).total_seconds() < cooldown:
                continue
            self._alerted[addr] = now
            await self.event_bus.emit(
                discovery_event(
                    token_address=addr,
                    symbol=r.snapshot.symbol,
                    chain=r.snapshot.chain.value,
                    filter_name=r.matched_filters[0],
                    overall_score=r.score.overall,
                    scores=r.score.as_dict(),
                )
            )


EVM_CHAINS = frozenset({Chain.ETHEREUM, Chain.BNB, Chain.BASE})


def build_adapters(
    config: DiscoveryConfig,
    dexscreener: Any,
    jupiter: Any = None,
) -> dict[Chain, ChainAdapter]:
    """Construct chain adapters for the enabled chains.

    Solana uses Jupiter + RugCheck; the EVM chains (Ethereum/BNB/Base) share one
    ``EvmAdapter`` + a single GoPlus provider. All read market data from the shared
    DexScreener provider.
    """
    from fenrir.discovery.chains.evm import EvmAdapter
    from fenrir.discovery.chains.solana import SolanaAdapter
    from fenrir.discovery.providers.goplus import GoPlusProvider

    adapters: dict[Chain, ChainAdapter] = {}
    goplus: GoPlusProvider | None = None
    for chain in config.chains:
        if chain is Chain.SOLANA:
            adapters[chain] = SolanaAdapter(dexscreener, jupiter=jupiter)
        elif chain in EVM_CHAINS:
            if goplus is None:
                goplus = GoPlusProvider()
            adapters[chain] = EvmAdapter(chain, dexscreener, goplus)
        else:
            logger.info("Discovery: %s adapter not available — skipping", chain.value)
    return adapters
