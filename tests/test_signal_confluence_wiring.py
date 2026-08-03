#!/usr/bin/env python3
"""
FENRIR - Signal confluence wiring tests (Phase 5.3, read-only surfacing)

The bot ingests strategy signals into the aggregator and emits SIGNAL_CONFLUENCE when
independent strategies agree — without changing sizing or execution. Verifies the flag
gating, the emit-on-agreement / no-emit-below-threshold behavior, and that routing is
unaffected. No network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fenrir.bot import FenrirBot
from fenrir.config import BotConfig, TradingMode

TOKEN = "So11111111111111111111111111111111111111112"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "SIGNAL_CONFLUENCE_ENABLED",
        "SIGNAL_CONFLUENCE_MIN_SOURCES",
        "MULTI_AGENT_PIPELINE_ENABLED",
        "AI_EVALUATE_ALL_LAUNCHES",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bot(tmp_path: Path, **overrides: Any) -> FenrirBot:
    overrides.setdefault("multi_agent_pipeline_enabled", False)
    cfg = BotConfig(
        mode=TradingMode.SIMULATION,
        ai_analysis_enabled=False,
        log_file=str(tmp_path / "t.log"),
        **overrides,
    )
    return FenrirBot(cfg)


def _sig_strat(sid: str, score: float) -> Any:
    """A market-data strategy that always claims TOKEN with a bespoke signal carrying a
    metadata['strategy'] tag and a *_score property (so normalization picks it up)."""
    signal = SimpleNamespace(
        token_address=TOKEN, symbol="X", metadata={"strategy": sid}, conviction_score=score
    )
    return SimpleNamespace(
        strategy_id=sid,
        uses_market_data=True,
        state=SimpleNamespace(active=True, paused=False),
        evaluate_token=lambda td, md: signal,
        build_ai_context=lambda s: "ctx",
    )


class _FakeMarket:
    async def check(self, token_address: str) -> Any:
        from fenrir.filters import MarketData

        return SimpleNamespace(passed=True, market_data=MarketData(token_address=TOKEN))


class TestConfig:
    def test_defaults_off(self) -> None:
        assert BotConfig(mode=TradingMode.SIMULATION).signal_confluence_enabled is False

    def test_env_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIGNAL_CONFLUENCE_ENABLED", "true")
        monkeypatch.setenv("SIGNAL_CONFLUENCE_MIN_SOURCES", "3")
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.signal_confluence_enabled is True
        assert cfg.signal_confluence_min_sources == 3


class TestConstruction:
    def test_aggregator_absent_when_disabled(self, tmp_path: Path) -> None:
        assert _make_bot(tmp_path).signal_aggregator is None

    def test_aggregator_present_when_enabled(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, signal_confluence_enabled=True)
        assert bot.signal_aggregator is not None


class TestSurfacing:
    async def test_emits_on_agreement(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, signal_confluence_enabled=True)
        emit = AsyncMock()
        bot.event_bus = SimpleNamespace(emit=emit)  # type: ignore[assignment]
        s1 = SimpleNamespace(
            token_address=TOKEN, symbol="X", metadata={"strategy": "a"}, s_score=0.5
        )
        s2 = SimpleNamespace(
            token_address=TOKEN, symbol="X", metadata={"strategy": "b"}, s_score=0.6
        )

        await bot._surface_confluence(TOKEN, "X", [s1, s2])

        emit.assert_awaited_once()
        assert emit.await_args is not None
        ev = emit.await_args.args[0]
        assert ev.event_type == "SIGNAL_CONFLUENCE"
        assert sorted(ev.data["sources"]) == ["a", "b"]
        # noisy-OR of 0.5 and 0.6 = 1 - 0.5*0.4 = 0.8
        assert ev.data["combined_strength"] == pytest.approx(0.8, abs=1e-6)

    async def test_no_emit_below_min_sources(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, signal_confluence_enabled=True, signal_confluence_min_sources=2)
        emit = AsyncMock()
        bot.event_bus = SimpleNamespace(emit=emit)  # type: ignore[assignment]
        s1 = SimpleNamespace(
            token_address=TOKEN, symbol="X", metadata={"strategy": "a"}, s_score=0.9
        )

        await bot._surface_confluence(TOKEN, "X", [s1])

        emit.assert_not_awaited()


class TestRoutingIntegration:
    async def test_two_strategies_agree_emits_confluence(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, signal_confluence_enabled=True)
        bot.security_filter = None
        bot.market_filter = _FakeMarket()  # type: ignore[assignment]
        bot.strategies = [_sig_strat("stratA", 0.5), _sig_strat("stratB", 0.5)]
        emit = AsyncMock()
        bot.event_bus = SimpleNamespace(emit=emit)  # type: ignore[assignment]

        routed = await bot._scan_and_route({"token_address": TOKEN, "symbol": "X", "name": "X"})

        # Both claimed → routing unaffected (two strategies), and a confluence event fired.
        assert {s.strategy_id for s, _ in routed} == {"stratA", "stratB"}
        conf = [
            c.args[0]
            for c in emit.await_args_list
            if c.args and getattr(c.args[0], "event_type", "") == "SIGNAL_CONFLUENCE"
        ]
        assert len(conf) == 1
        assert sorted(conf[0].data["sources"]) == ["stratA", "stratB"]

    async def test_disabled_does_not_emit(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, signal_confluence_enabled=False)
        bot.security_filter = None
        bot.market_filter = _FakeMarket()  # type: ignore[assignment]
        bot.strategies = [_sig_strat("stratA", 0.5), _sig_strat("stratB", 0.5)]
        emit = AsyncMock()
        bot.event_bus = SimpleNamespace(emit=emit)  # type: ignore[assignment]

        await bot._scan_and_route({"token_address": TOKEN, "symbol": "X", "name": "X"})

        conf = [
            c.args[0]
            for c in emit.await_args_list
            if c.args and getattr(c.args[0], "event_type", "") == "SIGNAL_CONFLUENCE"
        ]
        assert conf == []  # aggregator not built → no surfacing
