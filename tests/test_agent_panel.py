#!/usr/bin/env python3
"""
FENRIR - MultiAgentPanel Test Suite

Covers the veto-aware aggregation, response parsing, and the parallel score()
path. Network is fully mocked.
"""

from __future__ import annotations

from typing import cast

import aiohttp
import pytest

from fenrir.ai.agent_panel import Agent, AgentResult, MultiAgentPanel
from fenrir.ai.ensemble_scorer import ConvictionLevel


def _panel() -> MultiAgentPanel:
    return MultiAgentPanel(api_key="k", model="test/model")  # noqa: S106


def _ar(name: str, score: float, veto: bool = False, failed: bool = False) -> AgentResult:
    return AgentResult(name, score, "BUY" if score >= 60 else "SKIP", "r", veto, failed)


class TestAggregate:
    def test_all_buy_high_conviction(self):
        r = _panel()._aggregate(
            [_ar("risk", 80, veto=True), _ar("momentum", 70), _ar("narrative", 75)]
        )
        assert r.conviction is ConvictionLevel.HIGH_CONVICTION
        assert r.position_multiplier == 1.0
        assert r.should_trade is True

    def test_risk_veto_rejects_despite_others(self):
        r = _panel()._aggregate(
            [_ar("risk", 30, veto=True), _ar("momentum", 95), _ar("narrative", 90)]
        )
        assert r.conviction is ConvictionLevel.REJECT
        assert r.position_multiplier == 0.0
        assert r.should_trade is False
        assert "risk" in (r.veto_reason or "")

    def test_majority_buy_low_conviction(self):
        r = _panel()._aggregate(
            [_ar("risk", 70, veto=True), _ar("momentum", 70), _ar("narrative", 30)]
        )
        assert r.conviction is ConvictionLevel.LOW_CONVICTION
        assert r.position_multiplier == 0.5
        assert r.should_trade is True

    def test_minority_buy_rejects(self):
        r = _panel()._aggregate(
            [_ar("risk", 70, veto=True), _ar("momentum", 30), _ar("narrative", 30)]
        )
        assert r.conviction is ConvictionLevel.REJECT
        assert r.should_trade is False

    def test_all_failed_rejects(self):
        r = _panel()._aggregate(
            [_ar("risk", 0, veto=True, failed=True), _ar("momentum", 0, failed=True)]
        )
        assert r.conviction is ConvictionLevel.REJECT
        assert r.veto_reason == "all agents failed"

    def test_partial_failure_is_degraded(self):
        r = _panel()._aggregate(
            [_ar("risk", 80, veto=True), _ar("momentum", 70), _ar("narrative", 0, failed=True)]
        )
        assert r.conviction is ConvictionLevel.DEGRADED
        assert r.position_multiplier == 1.0
        assert r.should_trade is True


class TestParse:
    def test_valid_json(self):
        a = _panel()._parse(
            Agent("risk", "p", veto=True), '{"score": 82, "decision": "BUY", "reasoning": "safe"}'
        )
        assert a.score == 82.0
        assert a.decision == "BUY"
        assert a.veto is True
        assert a.failed is False

    def test_markdown_wrapped_json(self):
        a = _panel()._parse(
            Agent("m", "p"), 'here: ```json\n{"score": 40, "decision": "SKIP"}\n```'
        )
        assert a.score == 40.0
        assert a.decision == "SKIP"

    def test_bad_response_fails(self):
        a = _panel()._parse(Agent("m", "p"), "no json here")
        assert a.failed is True
        assert a.score == 0.0


# --- integration: parallel score() with a mocked session -------------------


class _Resp:
    def __init__(self, content: str):
        self.status = 200
        self._c = content

    async def text(self):  # pragma: no cover - only on error paths
        return self._c

    async def json(self):
        return {"choices": [{"message": {"content": self._c}}]}


class _CM:
    def __init__(self, content):
        self._c = content

    async def __aenter__(self):
        return _Resp(self._c)

    async def __aexit__(self, *a):
        return False


class _Session:
    def __init__(self, content):
        self._c = content
        self.calls = 0

    def post(self, url, headers=None, json=None):  # noqa: A002
        self.calls += 1
        return _CM(self._c)

    async def close(self):
        pass


class TestScoreIntegration:
    @pytest.mark.asyncio
    async def test_all_agents_buy(self):
        panel = _panel()
        panel._session = cast(
            aiohttp.ClientSession, _Session('{"score": 78, "decision": "BUY", "reasoning": "ok"}')
        )
        result = await panel.score("some token context", sol_amount=0.5)
        assert result.conviction is ConvictionLevel.HIGH_CONVICTION
        assert result.should_trade is True
        assert len(result.agents) == 3  # default panel size
        assert cast(_Session, panel._session).calls == 3  # one call per agent, in parallel

    @pytest.mark.asyncio
    async def test_summary_is_readable(self):
        panel = _panel()
        panel._session = cast(aiohttp.ClientSession, _Session('{"score": 78, "decision": "BUY"}'))
        result = await panel.score("ctx", sol_amount=0.5)
        s = result.summary()
        assert "risk=" in s and "momentum=" in s and "narrative=" in s
