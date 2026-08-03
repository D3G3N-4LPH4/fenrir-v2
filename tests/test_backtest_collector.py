#!/usr/bin/env python3
"""
FENRIR - Forward-price collector + JSONL loader tests (empirical phase)

The collector samples an injected price source into a BacktestSample-shaped JSONL record;
the round-trip (collect → load_jsonl → backtest) is the whole point, so it is tested end
to end. Plus bot wiring: gated construction and a spawned, deduped collection task on a
strategy flag. No network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fenrir.backtest import (
    ForwardPriceCollector,
    PortfolioBacktester,
    load_jsonl,
    market_data_to_dict,
)
from fenrir.config import BotConfig, TradingMode
from fenrir.filters import MarketData

TOKEN = "So11111111111111111111111111111111111111112"


def _fire_md() -> MarketData:
    return MarketData(
        token_address=TOKEN,
        pair_address="P",
        dex_id="raydium",
        age_minutes=120.0,
        market_cap_usd=500_000.0,
        price_usd=0.001,
        liquidity_usd=100_000.0,
        volume_5m_usd=30_000.0,
        volume_1h_usd=200_000.0,
        txns_5m_buys=70,
        txns_5m_sells=30,
        price_change_5m_pct=2.0,
        price_change_1h_pct=25.0,
        price_change_24h_pct=150.0,
    )


def _price_getter(values: list[float | None]) -> Any:
    it = iter(values)

    async def _get(token: str) -> float | None:
        try:
            return next(it)
        except StopIteration:
            return None

    return _get


class TestMarketDataSerialization:
    def test_json_safe_fields_only(self) -> None:
        d = market_data_to_dict(_fire_md())
        assert d["price_change_1h_pct"] == 25.0
        assert d["token_address"] == TOKEN
        assert "raw" not in d  # non-serializable field excluded
        # Reconstructable as MarketData kwargs.
        assert MarketData(**d).price_change_1h_pct == 25.0


class TestCollect:
    async def test_writes_record(self, tmp_path: Path) -> None:
        out = tmp_path / "samples.jsonl"
        col = ForwardPriceCollector(
            _price_getter([1.0, 1.2, 1.6, 1.7]), out, frame_seconds=0, max_frames=4
        )
        rec = await col.collect(TOKEN, _fire_md(), symbol="X")
        assert rec is not None
        assert rec["forward_prices"] == [1.0, 1.2, 1.6, 1.7]
        assert rec["market_data"]["price_change_1h_pct"] == 25.0
        assert out.exists()
        assert col.collected == 1

    async def test_skips_bad_prices(self, tmp_path: Path) -> None:
        out = tmp_path / "s.jsonl"
        col = ForwardPriceCollector(
            _price_getter([1.0, None, 0.0, 1.5]), out, frame_seconds=0, max_frames=4
        )
        rec = await col.collect(TOKEN, _fire_md())
        assert rec is not None
        assert rec["forward_prices"] == [1.0, 1.5]  # None and 0 dropped

    async def test_none_when_too_few_prices(self, tmp_path: Path) -> None:
        out = tmp_path / "s.jsonl"
        col = ForwardPriceCollector(_price_getter([1.0]), out, frame_seconds=0, max_frames=1)
        assert await col.collect(TOKEN, _fire_md()) is None
        assert not out.exists()

    async def test_survives_price_errors(self, tmp_path: Path) -> None:
        calls = {"n": 0}

        async def flaky(token: str) -> float | None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("feed down")
            return 1.0 + calls["n"] * 0.1

        col = ForwardPriceCollector(flaky, tmp_path / "s.jsonl", frame_seconds=0, max_frames=4)
        rec = await col.collect(TOKEN, _fire_md())
        assert rec is not None
        assert len(rec["forward_prices"]) == 3  # one frame errored


class TestRoundTrip:
    async def test_collect_then_backtest(self, tmp_path: Path) -> None:
        # Collect a momentum-entering sample, load it back, and backtest it.
        from fenrir.strategies.momentum import MomentumStrategy

        out = tmp_path / "samples.jsonl"
        col = ForwardPriceCollector(
            _price_getter([1.0, 1.6, 1.7]), out, frame_seconds=0, max_frames=3
        )
        await col.collect(TOKEN, _fire_md(), symbol="X")

        samples = load_jsonl(out)
        assert len(samples) == 1
        res = PortfolioBacktester().run([MomentumStrategy(BotConfig())], samples)
        assert res.per_strategy["momentum"].samples_entered == 1
        assert res.combined_metrics.trades == 1


class TestJsonlLoader:
    def test_skips_blank_and_bad_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "s.jsonl"
        good = '{"token_address": "%s", "forward_prices": [1.0, 1.1], "market_data": {}}' % TOKEN
        path.write_text(f"{good}\n\nnot json\n", encoding="utf-8")
        samples = load_jsonl(path)
        assert len(samples) == 1


class TestConfig:
    def test_defaults_off(self) -> None:
        assert BotConfig(mode=TradingMode.SIMULATION).sample_collection_enabled is False

    def test_env_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAMPLE_COLLECTION_ENABLED", "true")
        monkeypatch.setenv("SAMPLE_COLLECTION_FRAMES", "10")
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.sample_collection_enabled is True
        assert cfg.sample_collection_frames == 10


class TestBotWiring:
    @pytest.fixture(autouse=True)
    def _iso(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("SAMPLE_COLLECTION_ENABLED", "MULTI_AGENT_PIPELINE_ENABLED"):
            monkeypatch.delenv(var, raising=False)

    def _bot(self, tmp_path: Path, **over: Any) -> Any:
        from fenrir.bot import FenrirBot

        over.setdefault("multi_agent_pipeline_enabled", False)
        cfg = BotConfig(
            mode=TradingMode.SIMULATION,
            ai_analysis_enabled=False,
            log_file=str(tmp_path / "t.log"),
            **over,
        )
        return FenrirBot(cfg)

    def test_collector_absent_when_disabled(self, tmp_path: Path) -> None:
        assert self._bot(tmp_path).sample_collector is None

    def test_collector_present_when_enabled(self, tmp_path: Path) -> None:
        bot = self._bot(tmp_path, sample_collection_enabled=True)
        assert bot.sample_collector is not None

    async def test_spawns_collection_on_flag(self, tmp_path: Path) -> None:
        bot = self._bot(tmp_path, sample_collection_enabled=True)
        bot.security_filter = None

        async def check(token: str) -> Any:
            return SimpleNamespace(passed=True, market_data=_fire_md())

        bot.market_filter = SimpleNamespace(check=check)  # type: ignore[assignment]
        signal = SimpleNamespace(token_address=TOKEN, symbol="X", metadata={"strategy": "s"})
        bot.strategies = [
            SimpleNamespace(
                strategy_id="s",
                uses_market_data=True,
                state=SimpleNamespace(active=True, paused=False),
                evaluate_token=lambda td, md: signal,
                build_ai_context=lambda s: "ctx",
            )
        ]
        bot.event_bus = SimpleNamespace(emit=AsyncMock())  # type: ignore[assignment]
        collect = AsyncMock(return_value=None)
        bot.sample_collector.collect = collect  # type: ignore[method-assign]

        await bot._scan_and_route({"token_address": TOKEN, "symbol": "X", "name": "X"})
        # Let the spawned task run.
        import asyncio

        for _ in range(20):
            await asyncio.sleep(0.005)
            if collect.await_count:
                break
        collect.assert_awaited_once()
        assert collect.await_args is not None
        assert collect.await_args.args[0] == TOKEN
