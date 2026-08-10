#!/usr/bin/env python3
"""
FENRIR - Observation-window boot test (empirical phase)

The empirical phase runs the bot in SIMULATION with every read-only instrument on
(multi-agent pipeline + momentum/mean_reversion + sample collection + confluence
surfacing + arbitrage monitor) to gather real data without risking capital. This test
pins that the fully-instrumented config constructs cleanly and wires every instrument —
so a wiring regression can't silently waste an observation window. No network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fenrir.bot import FenrirBot
from fenrir.config import BotConfig, TradingMode


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "MULTI_AGENT_PIPELINE_ENABLED",
        "SAMPLE_COLLECTION_ENABLED",
        "SIGNAL_CONFLUENCE_ENABLED",
        "ARBITRAGE_MONITOR_ENABLED",
        "AI_EVALUATE_ALL_LAUNCHES",
        "ENABLED_STRATEGIES",
    ):
        monkeypatch.delenv(var, raising=False)


def _observation_bot(tmp_path: Path) -> FenrirBot:
    cfg = BotConfig(
        mode=TradingMode.SIMULATION,
        ai_analysis_enabled=False,
        enabled_strategies=["momentum", "mean_reversion"],
        multi_agent_pipeline_enabled=True,
        sample_collection_enabled=True,
        sample_collection_path=str(tmp_path / "samples.jsonl"),
        signal_confluence_enabled=True,
        arbitrage_monitor_enabled=True,
        log_file=str(tmp_path / "obs.log"),
    )
    return FenrirBot(cfg)


class TestObservationWindowBoots:
    def test_all_instruments_constructed(self, tmp_path: Path) -> None:
        bot = _observation_bot(tmp_path)
        # Every read-only instrument is wired in __init__.
        assert bot.agent_pipeline is not None
        assert bot.sample_collector is not None
        assert bot.signal_aggregator is not None
        # Both directional strategies are loaded.
        assert {s.strategy_id for s in bot.strategies} == {"momentum", "mean_reversion"}

    def test_simulation_mode_no_real_trades(self, tmp_path: Path) -> None:
        bot = _observation_bot(tmp_path)
        assert bot.config.mode == TradingMode.SIMULATION

    def test_arbitrage_monitor_builds_from_config(self, tmp_path: Path) -> None:
        # The arb monitor is built in start(); confirm its factory works with this config.
        bot = _observation_bot(tmp_path)
        monitor = bot.config.build_arbitrage_monitor(event_bus=bot.event_bus, logger=bot.logger)
        assert monitor is not None
        assert monitor.size_sol == bot.config.arbitrage_size_sol

    def test_collector_writes_to_configured_path(self, tmp_path: Path) -> None:
        bot = _observation_bot(tmp_path)
        assert bot.sample_collector.out_path == tmp_path / "samples.jsonl"

    def test_disabled_config_has_no_instruments(self, tmp_path: Path) -> None:
        # The default (no flags) must remain lean — the instruments are strictly opt-in.
        cfg = BotConfig(
            mode=TradingMode.SIMULATION,
            ai_analysis_enabled=False,
            multi_agent_pipeline_enabled=False,
            log_file=str(tmp_path / "d.log"),
        )
        bot = FenrirBot(cfg)
        assert bot.sample_collector is None
        assert bot.signal_aggregator is None
        assert bot.agent_pipeline is None
