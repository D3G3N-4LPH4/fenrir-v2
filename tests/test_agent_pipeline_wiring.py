#!/usr/bin/env python3
"""
FENRIR - Multi-agent pipeline bot wiring tests (Phase 3.2, strangler)

Proves the flag-gated wiring is additive and, crucially, that routing a launch
through the AgentPipeline produces the SAME decisions as the inline loop — the
two paths share _scan_and_route / _size_for_strategy / _execute_sized, so parity
is structural. Collaborators (filters, AI brain, engine) are stubbed; no network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fenrir.bot import FenrirBot
from fenrir.config import BotConfig, TradingMode

TOKEN = "So11111111111111111111111111111111111111112"
_TD = {"token_address": TOKEN, "symbol": "X", "name": "N"}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AI_EVALUATE_ALL_LAUNCHES",
        "MULTI_AGENT_PIPELINE_ENABLED",
        "SMART_MONEY_ENABLED",
        "MARKET_SCANNER_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bot(tmp_path: Path, **overrides: Any) -> FenrirBot:
    cfg = BotConfig(
        mode=TradingMode.SIMULATION,
        ai_analysis_enabled=False,
        log_file=str(tmp_path / "t.log"),
        **overrides,
    )
    return FenrirBot(cfg)


def _classic(strategy_id: str, should_eval: bool = True) -> Any:
    return SimpleNamespace(
        strategy_id=strategy_id,
        uses_market_data=False,
        state=SimpleNamespace(active=True, paused=False),
        should_evaluate=AsyncMock(return_value=should_eval),
    )


class TestConfigFlag:
    def test_defaults_off(self) -> None:
        assert BotConfig(mode=TradingMode.SIMULATION).multi_agent_pipeline_enabled is False

    def test_env_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MULTI_AGENT_PIPELINE_ENABLED", "true")
        assert BotConfig(mode=TradingMode.SIMULATION).multi_agent_pipeline_enabled is True


class TestConstruction:
    def test_pipeline_absent_when_disabled(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        assert bot.agent_pipeline is None

    def test_pipeline_built_when_enabled(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, multi_agent_pipeline_enabled=True)
        assert bot.agent_pipeline is not None
        # Scanner is fed via submit(), so it is NOT subscribed to the bus TOKEN_DETECTED.
        assert bot.agent_pipeline.scanner not in bot.event_bus._listeners


class TestOnLaunchRouting:
    async def test_disabled_uses_inline_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path)
        bot.event_bus = SimpleNamespace(emit=AsyncMock())  # type: ignore[assignment]
        scan = AsyncMock(return_value=[])
        monkeypatch.setattr(bot, "_scan_and_route", scan)
        await bot._on_token_launch(dict(_TD))
        scan.assert_awaited_once()  # inline path ran the shared router

    async def test_enabled_hands_off_to_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path, multi_agent_pipeline_enabled=True)
        bot.event_bus = SimpleNamespace(emit=AsyncMock())  # type: ignore[assignment]
        submit = AsyncMock()
        assert bot.agent_pipeline is not None
        monkeypatch.setattr(bot.agent_pipeline, "submit", submit)
        # The inline router must NOT run when the pipeline owns the launch.
        scan = AsyncMock(return_value=[])
        monkeypatch.setattr(bot, "_scan_and_route", scan)

        await bot._on_token_launch(dict(_TD))

        submit.assert_awaited_once()
        assert submit.await_args is not None
        assert submit.await_args.args[0]["token_address"] == TOKEN
        scan.assert_not_awaited()  # handed off, inline loop skipped


class TestPipelineCallables:
    async def test_claims_stashes_signal_context(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        strat = _classic("cls")
        bot.strategies = [strat]
        bot.security_filter = None
        bot.market_filter = None
        td = dict(_TD)
        ids = await bot._pipeline_claims(TOKEN, td)
        assert ids == ["cls"]
        assert td["_signal_contexts"] == {"cls": None}

    async def test_size_delegates_with_stashed_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path)
        strat = _classic("cls")
        bot.strategies = [strat]
        size = AsyncMock(return_value=0.05)
        monkeypatch.setattr(bot, "_size_for_strategy", size)
        td = {**_TD, "_signal_contexts": {"cls": "CTX"}}
        amount = await bot._pipeline_size("cls", td)
        assert amount == 0.05
        assert size.await_args is not None
        assert size.await_args.kwargs["signal_context"] == "CTX"

    async def test_size_unknown_strategy_is_none(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        assert await bot._pipeline_size("nope", dict(_TD)) is None

    async def test_execute_delegates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bot = _make_bot(tmp_path)
        strat = _classic("cls")
        bot.strategies = [strat]
        ex = AsyncMock(return_value=True)
        monkeypatch.setattr(bot, "_execute_sized", ex)
        ok = await bot._pipeline_execute(dict(_TD), 0.05, "cls")
        assert ok is True
        assert ex.await_args is not None
        assert ex.await_args.args[0] is strat
        assert ex.await_args.args[2] == 0.05

    async def test_find_any_resolves_scout(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        assert bot._find_any_strategy(bot._ai_scout.strategy_id) is bot._ai_scout


class TestParity:
    """The pipeline and inline paths must execute the same trade for one launch."""

    async def test_same_trade_both_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _run(pipeline: bool) -> list[tuple[str, float, str]]:
            bot = _make_bot(tmp_path, multi_agent_pipeline_enabled=pipeline)
            # NB: keep the real EventBus — in pipeline mode emit() is the transport
            # that carries CANDIDATE_FLAGGED/POSITION_SIZED between the agents.
            bot.security_filter = None
            bot.market_filter = None
            bot.strategies = [_classic("cls", should_eval=True)]

            calls: list[tuple[str, float, str]] = []

            # Stub sizing to approve a fixed amount and execution to record the call,
            # exercising the real _scan_and_route → size → execute wiring for both paths.
            async def fake_size(strategy: Any, td: dict, *, signal_context: Any = None) -> float:
                return 0.05

            async def fake_exec(strategy: Any, td: dict, amount: float) -> bool:
                calls.append((td["token_address"], amount, strategy.strategy_id))
                return True

            monkeypatch.setattr(bot, "_size_for_strategy", fake_size)
            monkeypatch.setattr(bot, "_execute_sized", fake_exec)

            if pipeline:
                assert bot.agent_pipeline is not None
                await bot.agent_pipeline.start()
                await bot._on_token_launch(dict(_TD))
                # Let the three workers drain the launch through to execution.
                for _ in range(50):
                    await asyncio.sleep(0.01)
                    if calls:
                        break
                await bot.agent_pipeline.stop()
            else:
                await bot._on_token_launch(dict(_TD))

            return calls

        inline = await _run(pipeline=False)
        piped = await _run(pipeline=True)

        assert inline == [(TOKEN, 0.05, "cls")]
        assert piped == inline  # identical decision + execution via the pipeline
