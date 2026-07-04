#!/usr/bin/env python3
"""
FENRIR - Pipeline Wiring 5a Test Suite

Covers the first wiring slice: the strategy loader falling back to
config.enabled_strategies, filter instantiation from the config flags, the
TradingStrategy.evaluate_token / uses_market_data dispatch hook, and the
_on_token_launch filter-gate + market-data dispatch branch.

Collaborators (AI brain, execution, event bus, filters) are stubbed — no
network, no live trading. DBs are written under tmp_path.

Run with: pytest tests/test_pipeline_5a.py -v
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from fenrir.bot import FenrirBot
from fenrir.config import BotConfig, TradingMode
from fenrir.filters import MarketData
from fenrir.strategies import (
    MigrationSniperStrategy,
    NarrativeTrackerStrategy,
    ReversalStrategy,
    SniperStrategy,
    VolumeAnomalyStrategy,
)

TOKEN = "So11111111111111111111111111111111111111112"


def _make_bot(tmp_path: Path, **overrides: Any) -> FenrirBot:
    cfg = BotConfig(
        mode=TradingMode.SIMULATION,
        ai_analysis_enabled=False,
        log_file=str(tmp_path / "t.log"),
        **overrides,
    )
    return FenrirBot(cfg)


# ---------------------------------------------------------------------------
# ABC dispatch hook
# ---------------------------------------------------------------------------


class TestAbcHook:
    def test_classic_strategy_defaults(self) -> None:
        s = SniperStrategy(BotConfig())
        assert s.uses_market_data is False
        # Base no-op hook returns None for non-signal strategies.
        assert s.evaluate_token({"token_address": TOKEN}) is None

    @pytest.mark.parametrize(
        "cls",
        [
            MigrationSniperStrategy,
            ReversalStrategy,
            VolumeAnomalyStrategy,
            NarrativeTrackerStrategy,
        ],
    )
    def test_signal_strategies_flagged(self, cls: type) -> None:
        assert cls(BotConfig()).uses_market_data is True


# ---------------------------------------------------------------------------
# Loader + filter instantiation
# ---------------------------------------------------------------------------


class TestLoaderAndFilters:
    def test_default_loads_sniper_only_no_filters(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        assert [s.strategy_id for s in bot.strategies] == ["sniper"]
        assert bot.market_filter is None
        assert bot.security_filter is None
        assert bot._needs_market_data is False

    def test_enabled_strategies_from_config(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, enabled_strategies=["sniper", "volume_anomaly"])
        assert [s.strategy_id for s in bot.strategies] == ["sniper", "volume_anomaly"]

    def test_signal_strategy_forces_market_data_provider(self, tmp_path: Path) -> None:
        # market gate off, but a signal strategy needs the MarketData snapshot.
        bot = _make_bot(tmp_path, enabled_strategies=["reversal"])
        assert bot._needs_market_data is True
        assert bot.market_filter is not None
        assert bot.market_filter.config.enabled is True  # provider mode

    def test_market_gate_flag_builds_filter(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, market_filter_enabled=True)
        assert bot.market_filter is not None

    def test_security_flag_builds_filter(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path, security_filter_enabled=True)
        assert bot.security_filter is not None

    def test_explicit_strategies_arg_overrides_config(self, tmp_path: Path) -> None:
        cfg = BotConfig(
            mode=TradingMode.SIMULATION,
            ai_analysis_enabled=False,
            enabled_strategies=["reversal"],
            log_file=str(tmp_path / "t.log"),
        )
        bot = FenrirBot(cfg, strategies=["graduation"])
        assert [s.strategy_id for s in bot.strategies] == ["graduation"]


# ---------------------------------------------------------------------------
# Dispatch (_on_token_launch)
# ---------------------------------------------------------------------------


class _Res:
    def __init__(self, passed: bool, md: Any = None) -> None:
        self.passed = passed
        self.market_data = md

    def __str__(self) -> str:
        return "res"


class _FakeMarket:
    def __init__(self, res: _Res) -> None:
        self._res = res
        self.check = AsyncMock(return_value=res)


class _FakeSecurity:
    def __init__(self, passed: bool) -> None:
        self.check = AsyncMock(return_value=_Res(passed))


def _classic(strategy_id: str, should_eval: bool) -> Any:
    return SimpleNamespace(
        strategy_id=strategy_id,
        uses_market_data=False,
        state=SimpleNamespace(active=True, paused=False),
        should_evaluate=AsyncMock(return_value=should_eval),
    )


def _signal(strategy_id: str, signal: Any) -> Any:
    return SimpleNamespace(
        strategy_id=strategy_id,
        uses_market_data=True,
        state=SimpleNamespace(active=True, paused=False),
        evaluate_token=Mock(return_value=signal),
        build_ai_context=Mock(return_value="CTX"),
    )


def _neuter(bot: FenrirBot, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    bot.event_bus = SimpleNamespace(emit=AsyncMock())  # type: ignore[assignment]
    eval_mock = AsyncMock()
    monkeypatch.setattr(bot, "_evaluate_and_execute", eval_mock)
    bot.security_filter = None
    bot.market_filter = None
    return eval_mock


_TD = {"token_address": TOKEN, "symbol": "X", "name": "N"}


class TestDispatch:
    async def test_signal_path_passes_market_data_and_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path)
        eval_mock = _neuter(bot, monkeypatch)
        md = MarketData(token_address=TOKEN)
        bot.market_filter = _FakeMarket(_Res(True, md))  # type: ignore[assignment]
        sig = object()
        strat = _signal("sig", sig)
        bot.strategies = [strat]

        await bot._on_token_launch(dict(_TD))

        strat.evaluate_token.assert_called_once()
        assert strat.evaluate_token.call_args.args[1] is md
        eval_mock.assert_awaited_once_with(strat, dict(_TD), signal_context="CTX")

    async def test_signal_none_skips_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path)
        eval_mock = _neuter(bot, monkeypatch)
        bot.market_filter = _FakeMarket(_Res(True, MarketData(token_address=TOKEN)))  # type: ignore[assignment]
        strat = _signal("sig", None)
        bot.strategies = [strat]

        await bot._on_token_launch(dict(_TD))

        strat.evaluate_token.assert_called_once()
        eval_mock.assert_not_awaited()

    async def test_classic_path_uses_should_evaluate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path)
        eval_mock = _neuter(bot, monkeypatch)
        strat = _classic("cls", should_eval=True)
        bot.strategies = [strat]

        await bot._on_token_launch(dict(_TD))

        strat.should_evaluate.assert_awaited_once()
        eval_mock.assert_awaited_once_with(strat, dict(_TD))

    async def test_classic_should_evaluate_false_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path)
        eval_mock = _neuter(bot, monkeypatch)
        bot.strategies = [_classic("cls", should_eval=False)]

        await bot._on_token_launch(dict(_TD))

        eval_mock.assert_not_awaited()

    async def test_security_reject_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path)
        eval_mock = _neuter(bot, monkeypatch)
        bot.security_filter = _FakeSecurity(passed=False)  # type: ignore[assignment]
        bot.strategies = [_classic("cls", should_eval=True)]

        await bot._on_token_launch(dict(_TD))

        eval_mock.assert_not_awaited()

    async def test_market_gate_reject_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Gate ON (config flag True) + not passed → skip.
        bot = _make_bot(tmp_path, market_filter_enabled=True)
        eval_mock = _neuter(bot, monkeypatch)
        bot.market_filter = _FakeMarket(_Res(False, MarketData(token_address=TOKEN)))  # type: ignore[assignment]
        bot.strategies = [_classic("cls", should_eval=True)]

        await bot._on_token_launch(dict(_TD))

        eval_mock.assert_not_awaited()

    async def test_market_provider_mode_does_not_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Gate OFF (data-provider only): a not-passed result must NOT skip; the
        # signal strategy still receives the snapshot.
        bot = _make_bot(tmp_path)  # market_filter_enabled defaults False
        eval_mock = _neuter(bot, monkeypatch)
        md = MarketData(token_address=TOKEN)
        bot.market_filter = _FakeMarket(_Res(False, md))  # type: ignore[assignment]
        strat = _signal("sig", object())
        bot.strategies = [strat]

        await bot._on_token_launch(dict(_TD))

        strat.evaluate_token.assert_called_once()
        assert strat.evaluate_token.call_args.args[1] is md
        eval_mock.assert_awaited_once()

    async def test_paused_strategy_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = _make_bot(tmp_path)
        eval_mock = _neuter(bot, monkeypatch)
        strat = _classic("cls", should_eval=True)
        strat.state.paused = True
        bot.strategies = [strat]

        await bot._on_token_launch(dict(_TD))

        eval_mock.assert_not_awaited()
