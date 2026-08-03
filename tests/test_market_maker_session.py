#!/usr/bin/env python3
"""
FENRIR - Market-making paper session tests (Phase 4.3b, sim-against-real-data)

Verifies the streaming session reproduces the batch simulator exactly, reads a real
price feed read-only via the bridge, skips missing/bad prices, survives feed errors,
and never exposes a live-trade path. No network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fenrir.trading.market_maker import MarketMaker, MarketMakerConfig
from fenrir.trading.market_maker_session import (
    MarketMakingPaperSession,
    price_source_from_feed,
)

TOKEN = "So11111111111111111111111111111111111111112"

_OSC = [1.0, 0.96, 1.04, 0.96, 1.04, 0.96, 1.04, 0.96, 1.04, 0.96, 1.04, 1.0]


def _scripted_source(values: list[float | None]) -> Any:
    it = iter(values)

    async def _get() -> float | None:
        try:
            return next(it)
        except StopIteration:
            return None

    return _get


class TestStreamingMatchesBatch:
    def test_on_price_matches_simulate(self) -> None:
        # Streaming the same prints through the session must match batch simulate()
        # (same starting cash → same fills, same PnL).
        cfg = MarketMakerConfig()
        batch = MarketMaker(cfg).simulate(_OSC)

        session = MarketMakingPaperSession(TOKEN, config=cfg)
        for p in _OSC:
            session.on_price(p)
        rep = session.report()

        assert rep["buys"] == batch.buys
        assert rep["sells"] == batch.sells
        assert rep["realized_pnl_sol"] == pytest.approx(batch.realized_pnl_sol)
        assert rep["total_pnl_sol"] == pytest.approx(batch.total_pnl_sol)
        assert rep["ending_inventory_tokens"] == pytest.approx(batch.ending_inventory_tokens)

    def test_oscillation_captures_spread(self) -> None:
        session = MarketMakingPaperSession(TOKEN)
        for p in _OSC:
            session.on_price(p)
        rep = session.report()
        assert rep["realized_pnl_sol"] > 0
        assert rep["round_trips"] >= 1
        assert rep["simulation"] is True


class TestPriceHandling:
    def test_skips_missing_and_nonpositive(self) -> None:
        session = MarketMakingPaperSession(TOKEN)
        session.on_price(None)
        session.on_price(0.0)
        session.on_price(-1.0)
        assert session.ticks == 0
        assert session.skipped == 3
        session.on_price(1.0)
        assert session.ticks == 1


class TestRun:
    async def test_run_over_scripted_source(self) -> None:
        session = MarketMakingPaperSession(TOKEN)
        source = _scripted_source(list(_OSC))
        rep = await session.run(source, ticks=len(_OSC), interval_seconds=0)
        assert rep["ticks"] == len(_OSC)
        assert rep["realized_pnl_sol"] > 0

    async def test_run_survives_feed_errors(self) -> None:
        calls = {"n": 0}

        async def flaky() -> float | None:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("feed down")
            return 1.0

        session = MarketMakingPaperSession(TOKEN)
        rep = await session.run(flaky, ticks=6, interval_seconds=0)
        # Three good ticks, three errored — the session survived and counted skips.
        assert rep["ticks"] == 3
        assert rep["skipped"] == 3

    async def test_run_marks_none_prices_as_skipped(self) -> None:
        session = MarketMakingPaperSession(TOKEN)
        source = _scripted_source([1.0, None, 1.02, None, 0.98])
        rep = await session.run(source, ticks=5, interval_seconds=0)
        assert rep["ticks"] == 3
        assert rep["skipped"] == 2


class TestFeedBridge:
    async def test_price_source_from_feed_reads_only(self) -> None:
        # A fake aggregated feed: get_price returns an object with .price.
        feed = SimpleNamespace()

        async def get_price(mint: str, force_refresh: bool = False) -> Any:
            assert mint == TOKEN
            return SimpleNamespace(price=0.00042)

        feed.get_price = get_price  # type: ignore[attr-defined]
        source = price_source_from_feed(feed, TOKEN)
        assert await source() == pytest.approx(0.00042)

    async def test_price_source_handles_none_quote(self) -> None:
        feed = SimpleNamespace()

        async def get_price(mint: str, force_refresh: bool = False) -> Any:
            return None

        feed.get_price = get_price  # type: ignore[attr-defined]
        source = price_source_from_feed(feed, TOKEN)
        assert await source() is None
