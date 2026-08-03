#!/usr/bin/env python3
"""
FENRIR - Multi-agent pipeline STREAM parity (Phase 3.3, strangler cutover)

3.2 proved a single launch takes the same decision inline vs. through the pipeline.
3.3 is the cutover proof: drive a whole *stream* of diverse launches — some rejected
by the security gate, some claimed by no strategy, some claimed by one or two — and
assert the set of executed (token, strategy, amount) trades is byte-identical whether
routed by the inline loop or the started AgentPipeline.

To keep the diff meaningful and order-independent, each launch's outcome is a pure
function of its own token_data (security gate + per-strategy should_evaluate), and the
recording executor is stubbed so no shared budget/exposure state couples launches.
That isolates exactly what the flip changes — dispatch/concurrency — from what it must
NOT change: the decisions. The real _scan_and_route and _size_for_strategy (AI disabled
→ deterministic) run on both paths. No network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fenrir.bot import FenrirBot
from fenrir.config import BotConfig, TradingMode
from fenrir.strategies.base import TradeParams

Trade = tuple[str, str, float]  # (token_address, strategy_id, amount_sol)


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
        ai_analysis_enabled=False,  # deterministic auto-buy: (True, None, None)
        log_file=str(tmp_path / "t.log"),
        **overrides,
    )
    return FenrirBot(cfg)


class _SecurityGate:
    def __init__(self, blocked: set[str]) -> None:
        self._blocked = blocked

    async def check(self, token_addr: str, lp_mint: str | None = None) -> Any:
        passed = token_addr not in self._blocked
        return SimpleNamespace(passed=passed, details={}, __str__=lambda self: "sec")


def _strategy(strategy_id: str, wants: set[str]) -> Any:
    """Classic strategy that claims only launches whose address is in ``wants``."""

    async def should_evaluate(td: dict) -> bool:
        return td["token_address"] in wants

    return SimpleNamespace(
        strategy_id=strategy_id,
        uses_market_data=False,
        state=SimpleNamespace(active=True, paused=False),
        should_evaluate=should_evaluate,
        # get_ai_context / get_trade_params are consulted by the real _size_for_strategy.
        get_ai_context=lambda: "ctx",
        get_trade_params=lambda: TradeParams(buy_amount_sol=0.02),
        budget_sol=100.0,
        max_concurrent_positions=100,
    )


async def _run_stream(bot: FenrirBot, launches: list[dict], pipeline: bool) -> list[Trade]:
    """Feed every launch through the bot and return the executed trades (as a set-like
    list). Real scan+size run; only the terminal executor is captured."""
    executed: list[Trade] = []

    async def fake_exec(strategy: Any, td: dict, amount: float) -> bool:
        executed.append((td["token_address"], strategy.strategy_id, round(amount, 6)))
        return True

    bot._execute_sized = fake_exec  # type: ignore[method-assign,assignment]

    if pipeline:
        assert bot.agent_pipeline is not None
        await bot.agent_pipeline.start()
        for td in launches:
            await bot._on_token_launch(dict(td))
        # Drain: wait until every agent's queue is empty and idle.
        for _ in range(200):
            await asyncio.sleep(0.005)
            qsizes = [a._queue.qsize() for a in bot.agent_pipeline._agents]  # type: ignore[attr-defined]
            if not any(qsizes):
                break
        await asyncio.sleep(0.02)
        await bot.agent_pipeline.stop()
    else:
        for td in launches:
            await bot._on_token_launch(dict(td))

    return executed


def _launches() -> list[dict]:
    # 8 diverse launches: BLK* blocked by security, NONE* claimed by nobody,
    # A* claimed by alpha, B* by beta, AB* by both.
    addrs = ["BLK1", "NONE1", "A1", "B1", "AB1", "A2", "BLK2", "AB2"]
    return [{"token_address": a, "symbol": a, "name": a} for a in addrs]


class TestStreamParity:
    async def test_inline_and_pipeline_execute_identical_set(self, tmp_path: Path) -> None:
        blocked = {"BLK1", "BLK2"}
        alpha_wants = {"A1", "A2", "AB1", "AB2"}
        beta_wants = {"B1", "AB1", "AB2"}

        def _configure(bot: FenrirBot) -> None:
            bot.security_filter = _SecurityGate(blocked)  # type: ignore[assignment]
            bot.market_filter = None
            bot.strategies = [
                _strategy("alpha", alpha_wants),
                _strategy("beta", beta_wants),
            ]

        launches = _launches()

        inline_bot = _make_bot(tmp_path, multi_agent_pipeline_enabled=False)
        _configure(inline_bot)
        inline = await _run_stream(inline_bot, launches, pipeline=False)

        piped_bot = _make_bot(tmp_path, multi_agent_pipeline_enabled=True)
        _configure(piped_bot)
        piped = await _run_stream(piped_bot, launches, pipeline=True)

        # Same trades, regardless of dispatch order (pipeline runs concurrently).
        assert set(inline) == set(piped)
        # And the expected (token, strategy) pairs specifically: blocked launches never
        # trade, the unclaimed one never trades, both-claimed launches trade twice.
        # (Amount is geometry-derived but identical across paths, so compare pairs.)
        expected_pairs = {
            ("A1", "alpha"),
            ("A2", "alpha"),
            ("B1", "beta"),
            ("AB1", "alpha"),
            ("AB1", "beta"),
            ("AB2", "alpha"),
            ("AB2", "beta"),
        }
        assert {(t, s) for t, s, _ in inline} == expected_pairs
        assert {(t, s) for t, s, _ in piped} == expected_pairs

    async def test_scout_fallback_parity(self, tmp_path: Path) -> None:
        # With the always-on scout enabled and no strategy claiming, both paths must
        # fall the launch back to the scout identically.
        launches = [{"token_address": "S1", "symbol": "S1", "name": "S1"}]

        def _configure(bot: FenrirBot) -> None:
            bot.security_filter = None
            bot.market_filter = None
            bot.strategies = []  # nothing claims → scout fallback

        inline_bot = _make_bot(
            tmp_path, multi_agent_pipeline_enabled=False, ai_evaluate_all_launches=True
        )
        _configure(inline_bot)
        inline = await _run_stream(inline_bot, launches, pipeline=False)

        piped_bot = _make_bot(
            tmp_path, multi_agent_pipeline_enabled=True, ai_evaluate_all_launches=True
        )
        _configure(piped_bot)
        piped = await _run_stream(piped_bot, launches, pipeline=True)

        assert set(inline) == set(piped)
        assert [t[1] for t in inline] == ["ai_scout"]
