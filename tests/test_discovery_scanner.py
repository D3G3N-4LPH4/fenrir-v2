#!/usr/bin/env python3
"""
FENRIR - Discovery scanner service tests

Exercises DiscoveryScanner.scan_once (rank + filter + alert) with fake adapters and
a fake event bus — no network.
"""

from __future__ import annotations

import pytest

from fenrir.discovery.config import DiscoveryConfig
from fenrir.discovery.filters import FilterName
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.discovery.scanner import DiscoveryScanner, build_adapters


def _safe() -> SafetySignals:
    return SafetySignals(
        mint_disabled=True,
        freeze_disabled=True,
        lp_locked_or_burned=True,
        contract_verified=True,
        blacklist_present=False,
        honeypot=False,
    )


def _strong_high_cap(addr: str = "HIGH") -> TokenSnapshot:
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address=addr,
        symbol="BIG",
        market_cap_usd=5_000_000,
        liquidity_usd=500_000,
        volume_24h_usd=2_000_000,
        age_minutes=60 * 24 * 10,
        holder_count=5_000,
        txns_24h_buys=800,
        txns_24h_sells=400,
        price_change_24h_pct=25.0,
        twitter="t",
        telegram="tg",
        website="w",
        safety=_safe(),
    )


def _junk() -> TokenSnapshot:
    # Fails every filter (tiny mcap, no liquidity/volume).
    return TokenSnapshot(chain=Chain.SOLANA, token_address="JUNK", market_cap_usd=500)


class FakeAdapter:
    chain = Chain.SOLANA

    def __init__(self, snaps: list[TokenSnapshot]) -> None:
        self.snaps = snaps
        self.closed = False

    async def discover(self) -> list[TokenSnapshot]:
        return list(self.snaps)

    async def enrich(self, snap: TokenSnapshot) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class FakeBus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:  # noqa: ANN001
        self.events.append(event)


def _cfg(**kw) -> DiscoveryConfig:  # noqa: ANN003
    base = dict(
        enabled=True,
        chains=[Chain.SOLANA],
        filters=[FilterName.HIGH_CAP],
        min_alert_score=50.0,
        cooldown_minutes=30.0,
    )
    base.update(kw)
    return DiscoveryConfig(**base)  # type: ignore[arg-type]


class TestDiscoveryScanner:
    @pytest.mark.asyncio
    async def test_scan_once_ranks_and_filters(self) -> None:
        adapter = FakeAdapter([_junk(), _strong_high_cap()])
        scanner = DiscoveryScanner(_cfg(), {Chain.SOLANA: adapter})  # type: ignore[dict-item]
        results = await scanner.scan_once()
        assert len(results) == 1  # junk filtered out
        assert results[0].snapshot.token_address == "HIGH"
        assert "high_cap" in results[0].matched_filters
        assert results[0].score.overall > 50

    @pytest.mark.asyncio
    async def test_alert_emitted_once_within_cooldown(self) -> None:
        bus = FakeBus()
        adapter = FakeAdapter([_strong_high_cap()])
        scanner = DiscoveryScanner(_cfg(), {Chain.SOLANA: adapter}, event_bus=bus)  # type: ignore[dict-item]
        await scanner.scan_once()
        assert len(bus.events) == 1
        assert bus.events[0].event_type == "DISCOVERY"
        assert bus.events[0].data["chain"] == "solana"
        await scanner.scan_once()  # within cooldown → no new alert
        assert len(bus.events) == 1

    @pytest.mark.asyncio
    async def test_no_alert_below_min_score(self) -> None:
        bus = FakeBus()
        adapter = FakeAdapter([_strong_high_cap()])
        scanner = DiscoveryScanner(
            _cfg(min_alert_score=99.0),
            {Chain.SOLANA: adapter},
            event_bus=bus,  # type: ignore[dict-item]
        )
        await scanner.scan_once()
        assert bus.events == []

    @pytest.mark.asyncio
    async def test_get_results_filtering(self) -> None:
        adapter = FakeAdapter([_strong_high_cap("A"), _strong_high_cap("B")])
        scanner = DiscoveryScanner(_cfg(), {Chain.SOLANA: adapter})  # type: ignore[dict-item]
        await scanner.scan_once()
        assert len(scanner.get_results(chain=Chain.SOLANA)) == 2
        assert len(scanner.get_results(chain=Chain.ETHEREUM)) == 0
        assert len(scanner.get_results(min_score=99.0)) == 0
        assert len(scanner.get_results(filter_name="high_cap")) == 2
        assert len(scanner.get_results(filter_name="low_cap_alpha")) == 0

    @pytest.mark.asyncio
    async def test_stop_closes_adapters(self) -> None:
        adapter = FakeAdapter([])
        scanner = DiscoveryScanner(_cfg(), {Chain.SOLANA: adapter})  # type: ignore[dict-item]
        await scanner.stop()
        assert adapter.closed is True


class TestBuildAdapters:
    def test_solana_built_evm_skipped(self) -> None:
        cfg = DiscoveryConfig(chains=[Chain.SOLANA, Chain.ETHEREUM, Chain.BNB])
        adapters = build_adapters(cfg, dexscreener=object(), jupiter=object())
        assert Chain.SOLANA in adapters
        assert Chain.ETHEREUM not in adapters  # not implemented yet
        assert Chain.BNB not in adapters
