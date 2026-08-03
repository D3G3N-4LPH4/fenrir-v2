#!/usr/bin/env python3
"""
FENRIR - Arbitrage scanner + DexScreener bridge tests (Phase 4.4c-evidence, read-only)

Covers the multi-pool DexScreener → VenueQuote bridge (SOL-quoted filtering, WSOL-side
liquidity, per-DEX fees), the periodic scanner (cross-pool detection, resilience,
per-cycle cap), and the config wiring. No network, no execution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fenrir.trading.arbitrage_monitor import (
    ArbitrageMonitor,
    dexscreener_venue_quotes,
    venue_quotes_from_pairs,
)
from fenrir.trading.arbitrage_scanner import ArbitrageScanner

WSOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
TOKEN = "TokenMint1111111111111111111111111111111111"


def _pair(dex: str, addr: str, price_native: str, liq_quote: float, quote_mint: str = WSOL) -> dict:
    return {
        "dexId": dex,
        "pairAddress": addr,
        "priceNative": price_native,
        "quoteToken": {"address": quote_mint},
        "liquidity": {"usd": liq_quote * 150, "quote": liq_quote},
    }


class TestPairBridge:
    def test_builds_sol_quoted_quotes(self) -> None:
        pairs = [
            _pair("raydium", "RAYPOOL", "0.0010", 100.0),
            _pair("pumpswap", "PUMPPOOL", "0.0011", 80.0),
        ]
        quotes = venue_quotes_from_pairs(pairs)
        assert len(quotes) == 2
        ray = next(q for q in quotes if q.venue.startswith("raydium"))
        assert ray.price == pytest.approx(0.0010)
        assert ray.liquidity_sol == pytest.approx(100.0)  # WSOL-side reserve
        assert ray.fee_bps == 25

    def test_skips_non_sol_quoted(self) -> None:
        # A USDC-quoted pool's priceNative is in USDC, not SOL — must be excluded.
        pairs = [
            _pair("raydium", "RAYPOOL", "0.0010", 100.0),
            _pair("orca", "ORCAPOOL", "1.50", 100.0, quote_mint=USDC),
        ]
        quotes = venue_quotes_from_pairs(pairs)
        assert [q.venue.split(":")[0] for q in quotes] == ["raydium"]

    def test_skips_zero_price(self) -> None:
        quotes = venue_quotes_from_pairs([_pair("raydium", "P", "0", 100.0)])
        assert quotes == []

    def test_unknown_dex_default_fee(self) -> None:
        quotes = venue_quotes_from_pairs([_pair("somenewdex", "P", "0.001", 50.0)])
        assert quotes[0].fee_bps == 30

    async def test_dexscreener_venue_quotes(self) -> None:
        async def fetch_pairs(token: str) -> list[dict]:
            assert token == TOKEN
            return [_pair("raydium", "A", "0.001", 100.0), _pair("orca", "B", "0.0012", 90.0)]

        quotes = await dexscreener_venue_quotes(fetch_pairs, TOKEN)
        assert len(quotes) == 2


class TestScanner:
    def _monitor(self, bus: Any = None) -> ArbitrageMonitor:
        from fenrir.trading.arbitrage import ArbConfig, ArbitrageDetector

        return ArbitrageMonitor(
            ArbitrageDetector(ArbConfig(min_net_edge_bps=50.0, tx_cost_sol=0.002)),
            size_sol=1.0,
            event_bus=bus,
        )

    async def test_detects_cross_pool_divergence(self) -> None:
        bus = SimpleNamespace(emit=AsyncMock())
        # Two SOL pools, ~5% apart with deep liquidity → actionable.
        pairs = [
            _pair("raydium", "A", "1.00", 100_000.0),
            _pair("pumpswap", "B", "1.05", 100_000.0),
        ]

        async def fetch_pairs(token: str) -> list[dict]:
            return pairs

        async def token_source() -> list[str]:
            return [TOKEN]

        scanner = ArbitrageScanner(
            self._monitor(bus), fetch_pairs, token_source, interval_seconds=0
        )
        found = await scanner.scan_once()
        assert found == 1
        assert scanner.tokens_checked == 1
        bus.emit.assert_awaited_once()

    async def test_single_pool_no_divergence(self) -> None:
        async def fetch_pairs(token: str) -> list[dict]:
            return [_pair("raydium", "A", "1.00", 100_000.0)]

        async def token_source() -> list[str]:
            return [TOKEN]

        scanner = ArbitrageScanner(self._monitor(), fetch_pairs, token_source)
        assert await scanner.scan_once() == 0

    async def test_survives_fetch_error(self) -> None:
        async def fetch_pairs(token: str) -> list[dict]:
            raise RuntimeError("dexscreener down")

        async def token_source() -> list[str]:
            return [TOKEN, "Other"]

        scanner = ArbitrageScanner(self._monitor(), fetch_pairs, token_source)
        assert await scanner.scan_once() == 0  # errored per token, loop survived
        assert scanner.cycles == 1

    async def test_survives_token_source_error(self) -> None:
        async def fetch_pairs(token: str) -> list[dict]:
            return []

        async def token_source() -> list[str]:
            raise RuntimeError("source down")

        scanner = ArbitrageScanner(self._monitor(), fetch_pairs, token_source)
        assert await scanner.scan_once() == 0

    async def test_per_cycle_cap(self) -> None:
        checked: list[str] = []

        async def fetch_pairs(token: str) -> list[dict]:
            checked.append(token)
            return []

        async def token_source() -> list[str]:
            return [f"tok{i}" for i in range(10)]

        scanner = ArbitrageScanner(
            self._monitor(), fetch_pairs, token_source, max_tokens_per_cycle=3
        )
        await scanner.scan_once()
        assert len(checked) == 3


class TestConfigWiring:
    def test_defaults_off(self) -> None:
        from fenrir.config import BotConfig, TradingMode

        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.arbitrage_monitor_enabled is False

    def test_build_monitor_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fenrir.config import BotConfig, TradingMode

        monkeypatch.setenv("ARBITRAGE_MONITOR_ENABLED", "true")
        monkeypatch.setenv("ARBITRAGE_MIN_NET_EDGE_BPS", "75")
        monkeypatch.setenv("ARBITRAGE_SIZE_SOL", "0.25")
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.arbitrage_monitor_enabled is True
        mon = cfg.build_arbitrage_monitor()
        assert mon.size_sol == 0.25
        assert mon.detector.config.min_net_edge_bps == 75.0
