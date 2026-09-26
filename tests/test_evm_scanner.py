#!/usr/bin/env python3
"""
FENRIR - EVM evaluator scanner + wiring tests (Phase 7.1, read-only)

The scanner runs a token source through the EVM evaluator and emits EVM_SIGNAL events
for tokens that fire on an enabled chain — read-only, no execution. Plus config gating
and bot wiring. Evaluator/source are injected; no network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.evm import EvmEvaluatorScanner
from fenrir.evm.evaluator import EvmEvaluation
from fenrir.filters import MarketData
from fenrir.signals import ConfluenceResult, SignalDirection
from fenrir.signals.models import Signal

ETH_TOKEN = "0x1234567890abcdef1234567890abcdef12345678"


def _eval(
    chain: str, sources: list[str], strengths: list[float], confluent: bool = False
) -> EvmEvaluation:
    signals = [
        Signal(s, ETH_TOKEN, SignalDirection.LONG, st)
        for s, st in zip(sources, strengths, strict=True)
    ]
    confluence = None
    if confluent:
        confluence = ConfluenceResult(
            token_address=ETH_TOKEN,
            direction=SignalDirection.LONG,
            sources=sorted(sources),
            combined_strength=0.8,
            max_strength=max(strengths),
            signals=signals,
        )
    return EvmEvaluation(
        chain=chain,
        token_address=ETH_TOKEN,
        symbol="PEPE",
        market_data=MarketData(token_address=ETH_TOKEN),
        signals=signals,
        confluence=confluence,
    )


class _FakeEvaluator:
    def __init__(self, results: dict[str, Any]) -> None:
        self._results = results
        self.calls: list[str] = []

    async def evaluate(self, token: str, chain: Any = None) -> Any:
        self.calls.append(token)
        r = self._results.get(token, None)
        if r == "raise":
            raise RuntimeError("evaluate boom")
        return r


def _source_inner(tokens: list[str]) -> Any:
    async def _get() -> list[str]:
        return tokens

    return _get


class TestScan:
    async def test_emits_on_signal(self) -> None:
        bus = SimpleNamespace(emit=AsyncMock())
        ev = _FakeEvaluator({ETH_TOKEN: _eval("ethereum", ["momentum"], [0.6])})
        sc = EvmEvaluatorScanner(ev, _source_inner([ETH_TOKEN]), event_bus=bus)
        n = await sc.scan_once()
        assert n == 1
        assert sc.signals_surfaced == 1
        bus.emit.assert_awaited_once()
        e = bus.emit.await_args.args[0]
        assert e.event_type == "EVM_SIGNAL"
        assert e.data["chain"] == "ethereum"
        assert e.data["sources"] == ["momentum"]
        assert e.data["combined_strength"] == pytest.approx(0.6)
        assert e.data["confluent"] is False

    async def test_confluent_flag(self) -> None:
        bus = SimpleNamespace(emit=AsyncMock())
        ev = _FakeEvaluator(
            {ETH_TOKEN: _eval("base", ["momentum", "mean_reversion"], [0.6, 0.5], confluent=True)}
        )
        sc = EvmEvaluatorScanner(
            ev, _source_inner([ETH_TOKEN]), enabled_chains={"base"}, event_bus=bus
        )
        await sc.scan_once()
        e = bus.emit.await_args.args[0]
        assert e.data["confluent"] is True
        assert e.data["combined_strength"] == pytest.approx(0.8)

    async def test_no_signal_no_emit(self) -> None:
        bus = SimpleNamespace(emit=AsyncMock())
        ev = _FakeEvaluator({ETH_TOKEN: _eval("ethereum", [], [])})
        sc = EvmEvaluatorScanner(ev, _source_inner([ETH_TOKEN]), event_bus=bus)
        assert await sc.scan_once() == 0
        bus.emit.assert_not_awaited()

    async def test_none_result_skipped(self) -> None:
        bus = SimpleNamespace(emit=AsyncMock())
        ev = _FakeEvaluator({ETH_TOKEN: None})
        sc = EvmEvaluatorScanner(ev, _source_inner([ETH_TOKEN]), event_bus=bus)
        assert await sc.scan_once() == 0

    async def test_disabled_chain_filtered(self) -> None:
        bus = SimpleNamespace(emit=AsyncMock())
        # Resolves to Solana → not in enabled EVM chains → not surfaced.
        ev = _FakeEvaluator({ETH_TOKEN: _eval("solana", ["momentum"], [0.9])})
        sc = EvmEvaluatorScanner(
            ev, _source_inner([ETH_TOKEN]), enabled_chains={"ethereum"}, event_bus=bus
        )
        assert await sc.scan_once() == 0
        bus.emit.assert_not_awaited()

    async def test_evaluate_error_survives(self) -> None:
        ev = _FakeEvaluator({ETH_TOKEN: "raise", "0xother": _eval("ethereum", ["momentum"], [0.6])})
        sc = EvmEvaluatorScanner(ev, _source_inner([ETH_TOKEN, "0xother"]))
        assert await sc.scan_once() == 1  # first errored, second surfaced

    async def test_token_source_error_survives(self) -> None:
        async def bad() -> list[str]:
            raise RuntimeError("source down")

        sc = EvmEvaluatorScanner(_FakeEvaluator({}), bad)
        assert await sc.scan_once() == 0
        assert sc.cycles == 1

    async def test_per_cycle_cap(self) -> None:
        ev = _FakeEvaluator({})
        sc = EvmEvaluatorScanner(
            ev, _source_inner([f"0x{i}" for i in range(10)]), max_tokens_per_cycle=3
        )
        await sc.scan_once()
        assert len(ev.calls) == 3

    async def test_no_bus_ok(self) -> None:
        ev = _FakeEvaluator({ETH_TOKEN: _eval("ethereum", ["momentum"], [0.6])})
        sc = EvmEvaluatorScanner(ev, _source_inner([ETH_TOKEN]))
        assert await sc.scan_once() == 1  # emits nothing, returns count


class TestCollection:
    async def test_spawns_collection_on_flag(self) -> None:
        ev = _FakeEvaluator({ETH_TOKEN: _eval("ethereum", ["momentum"], [0.6])})
        collector = SimpleNamespace(collect=AsyncMock(return_value=None))
        sc = EvmEvaluatorScanner(ev, _source_inner([ETH_TOKEN]), collector=collector)
        await sc.scan_once()
        import asyncio

        for _ in range(20):
            await asyncio.sleep(0.005)
            if collector.collect.await_count:
                break
        collector.collect.assert_awaited_once()
        # collect(token, market_data, symbol)
        args = collector.collect.await_args.args
        assert args[0] == ETH_TOKEN
        assert isinstance(args[1], MarketData)
        assert args[2] == "PEPE"

    async def test_no_collection_without_collector(self) -> None:
        ev = _FakeEvaluator({ETH_TOKEN: _eval("ethereum", ["momentum"], [0.6])})
        sc = EvmEvaluatorScanner(ev, _source_inner([ETH_TOKEN]))  # no collector
        assert await sc.scan_once() == 1  # surfaced, but nothing to collect

    async def test_round_trip_flag_to_backtestable_sample(self, tmp_path: Path) -> None:
        # The money path: an EVM flag → collected forward-price sample → backtest enters.
        import asyncio

        from fenrir.backtest import ForwardPriceCollector, PortfolioBacktester, load_jsonl
        from fenrir.discovery.models import Chain, TokenSnapshot
        from fenrir.evm import EvmTokenEvaluator, snapshot_to_market_data
        from fenrir.strategies.momentum import MomentumStrategy

        snap = TokenSnapshot(
            chain=Chain.ETHEREUM,
            token_address=ETH_TOKEN,
            symbol="PEPE",
            dex_id="uniswap",
            price_usd=1.0,
            market_cap_usd=5_000_000.0,
            liquidity_usd=500_000.0,
            volume_5m_usd=60_000.0,
            volume_1h_usd=400_000.0,
            txns_5m_buys=70,
            txns_5m_sells=30,
            price_change_5m_pct=2.0,
            price_change_1h_pct=25.0,
            price_change_24h_pct=150.0,
            age_minutes=120.0,
        )

        async def fetch(token: str, chain: Any = None) -> TokenSnapshot:
            return snap

        evaluator = EvmTokenEvaluator([MomentumStrategy(BotConfig())], fetch)

        # Collector prices climb to a take-profit so the backtest records a real trade.
        prices = iter([1.0, 1.3, 1.65, 1.7])

        async def get_price(token: str) -> float:
            return next(prices)

        out = tmp_path / "evm_samples.jsonl"
        collector = ForwardPriceCollector(get_price, out, frame_seconds=0, max_frames=4)

        sc = EvmEvaluatorScanner(evaluator, _source_inner([ETH_TOKEN]), collector=collector)
        await sc.scan_once()
        for _ in range(30):
            await asyncio.sleep(0.005)
            if collector.collected:
                break

        samples = load_jsonl(out)
        assert len(samples) == 1
        # The collected snapshot's MarketData round-trips (momentum still fires on it).
        assert snapshot_to_market_data(snap).price_change_1h_pct == 25.0
        res = PortfolioBacktester().run([MomentumStrategy(BotConfig())], samples)
        assert res.per_strategy["momentum"].samples_entered == 1
        assert res.combined_metrics.trades == 1


class TestConfig:
    def test_defaults_off(self) -> None:
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.evm_evaluation_enabled is False
        assert "ethereum" in cfg.evm_chains

    def test_env_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVM_EVALUATION_ENABLED", "true")
        monkeypatch.setenv("EVM_CHAINS", "base,bnb")
        monkeypatch.setenv("EVM_WATCHLIST", "0xaaa,0xbbb")
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.evm_evaluation_enabled is True
        assert cfg.evm_chains == ["base", "bnb"]
        assert cfg.evm_watchlist == ["0xaaa", "0xbbb"]

    def test_build_evaluator_brain_gated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fetch(t: str, c: Any = None) -> Any:
            return None

        brain = object()
        cfg = BotConfig(mode=TradingMode.SIMULATION)  # evm_use_ai_brain default False
        ev = cfg.build_evm_evaluator(strategies=[], fetch_snapshot=fetch, brain=brain)
        assert ev._brain is None  # AI not called unless opted in

        monkeypatch.setenv("EVM_USE_AI_BRAIN", "true")
        cfg2 = BotConfig(mode=TradingMode.SIMULATION)
        ev2 = cfg2.build_evm_evaluator(strategies=[], fetch_snapshot=fetch, brain=brain)
        assert ev2._brain is brain


class TestBotWiring:
    @pytest.fixture(autouse=True)
    def _iso(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("EVM_EVALUATION_ENABLED", "MULTI_AGENT_PIPELINE_ENABLED"):
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

    def test_attrs_none_until_start(self, tmp_path: Path) -> None:
        # Built in start() (needs a provider); None after construction either way.
        bot = self._bot(tmp_path, evm_evaluation_enabled=True)
        assert bot.evm_evaluator is None
        assert bot.evm_scanner is None

    async def test_evm_token_source_returns_watchlist(self, tmp_path: Path) -> None:
        bot = self._bot(tmp_path, evm_evaluation_enabled=True, evm_watchlist=["0xaaa", "0xbbb"])
        assert await bot._evm_token_source() == ["0xaaa", "0xbbb"]
