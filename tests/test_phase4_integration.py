#!/usr/bin/env python3
"""
FENRIR - Phase 4 consolidation: multi-strategy through the live pipeline (integration)

Proves the two ready directional strategies (momentum, mean_reversion) run TOGETHER
through the multi-agent pipeline on a real detection stream: each independently claims
the launches that fit its thesis, they route concurrently, and sizing/execution happen
per-strategy. Exercises the real _scan_and_route + _size_for_strategy (AI disabled →
deterministic) end-to-end; only the terminal executor is captured. No network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fenrir.bot import FenrirBot
from fenrir.config import BotConfig, TradingMode
from fenrir.filters import MarketData

# Momentum fires: uptrend +25% 1h, +2% 5m, volume accelerating, buyers dominant.
_MOMENTUM_MD = dict(
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

# Mean-reversion fires: oversold -25% 1h but stabilizing (+1% 5m), buyers returning.
_REVERSION_MD = dict(
    age_minutes=180.0,
    market_cap_usd=500_000.0,
    price_usd=0.001,
    liquidity_usd=100_000.0,
    volume_5m_usd=30_000.0,
    volume_1h_usd=200_000.0,
    txns_5m_buys=55,
    txns_5m_sells=45,
    price_change_5m_pct=1.0,
    price_change_1h_pct=-25.0,
    price_change_24h_pct=-10.0,
)

# Neutral: flat — neither strategy claims it.
_NEUTRAL_MD = dict(
    age_minutes=120.0,
    market_cap_usd=500_000.0,
    price_usd=0.001,
    liquidity_usd=100_000.0,
    volume_5m_usd=30_000.0,
    volume_1h_usd=200_000.0,
    txns_5m_buys=50,
    txns_5m_sells=50,
    price_change_5m_pct=0.0,
    price_change_1h_pct=0.0,
    price_change_24h_pct=0.0,
)


class _FakeMarketFilter:
    """Returns a per-token MarketData snapshot (data-provider mode, never gates)."""

    def __init__(self, by_token: dict[str, MarketData]) -> None:
        self._by_token = by_token

    async def check(self, token_address: str) -> Any:
        md = self._by_token.get(token_address)
        return SimpleNamespace(passed=True, market_data=md)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AI_EVALUATE_ALL_LAUNCHES",
        "MULTI_AGENT_PIPELINE_ENABLED",
        "ENABLED_STRATEGIES",
        "MARKET_FILTER_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bot(tmp_path: Path) -> FenrirBot:
    cfg = BotConfig(
        mode=TradingMode.SIMULATION,
        ai_analysis_enabled=False,
        enabled_strategies=["momentum", "mean_reversion"],
        multi_agent_pipeline_enabled=True,
        log_file=str(tmp_path / "t.log"),
    )
    return FenrirBot(cfg)


def _md(token: str, fields: dict) -> MarketData:
    return MarketData(token_address=token, pair_address="PAIR", dex_id="raydium", **fields)


async def _drain(bot: FenrirBot) -> None:
    assert bot.agent_pipeline is not None
    for _ in range(200):
        await asyncio.sleep(0.005)
        if not any(a._queue.qsize() for a in bot.agent_pipeline._agents):  # type: ignore[attr-defined]
            break
    await asyncio.sleep(0.02)


class TestMultiStrategyPipeline:
    async def test_both_strategies_claim_and_execute_independently(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        assert {s.strategy_id for s in bot.strategies} == {"momentum", "mean_reversion"}
        assert bot.agent_pipeline is not None  # pipeline is the live path

        bot.security_filter = None
        bot.market_filter = _FakeMarketFilter(  # type: ignore[assignment]
            {
                "MOM": _md("MOM", _MOMENTUM_MD),
                "REV": _md("REV", _REVERSION_MD),
                "NEU": _md("NEU", _NEUTRAL_MD),
            }
        )

        executed: list[tuple[str, str]] = []

        async def fake_exec(strategy: Any, td: dict, amount: float) -> bool:
            executed.append((td["token_address"], strategy.strategy_id))
            return True

        bot._execute_sized = fake_exec  # type: ignore[method-assign,assignment]

        await bot.agent_pipeline.start()
        for token in ("MOM", "REV", "NEU"):
            await bot._on_token_launch({"token_address": token, "symbol": token, "name": token})
        await _drain(bot)
        await bot.agent_pipeline.stop()

        # Each thesis claimed exactly its own launch; the neutral one traded nowhere.
        assert set(executed) == {("MOM", "momentum"), ("REV", "mean_reversion")}

    async def test_scanner_keeps_flagging_while_executor_busy(self, tmp_path: Path) -> None:
        # Concurrency: a slow execution on one strategy must not stall routing of the
        # next launch for the other strategy.
        bot = _make_bot(tmp_path)
        bot.security_filter = None
        bot.market_filter = _FakeMarketFilter(  # type: ignore[assignment]
            {"MOM": _md("MOM", _MOMENTUM_MD), "REV": _md("REV", _REVERSION_MD)}
        )

        release = asyncio.Event()
        executed: list[str] = []

        async def slow_exec(strategy: Any, td: dict, amount: float) -> bool:
            await release.wait()  # both trades block here
            executed.append(td["token_address"])
            return True

        bot._execute_sized = slow_exec  # type: ignore[method-assign,assignment]

        assert bot.agent_pipeline is not None
        await bot.agent_pipeline.start()
        await bot._on_token_launch({"token_address": "MOM", "symbol": "MOM", "name": "M"})
        await bot._on_token_launch({"token_address": "REV", "symbol": "REV", "name": "R"})
        await asyncio.sleep(0.05)

        # Both launches were scanned + sized (candidate/position events flowed) while the
        # executor is blocked — nothing executed yet.
        assert executed == []
        assert bot.agent_pipeline.scanner.processed == 2

        release.set()
        await _drain(bot)
        await bot.agent_pipeline.stop()
        assert set(executed) == {"MOM", "REV"}
