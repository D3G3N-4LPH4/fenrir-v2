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


async def _no_sleep(*_a, **_k):
    """Patch out backoff delays so retry tests run instantly."""
    return None


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

    def test_all_failed_degrades_to_brain_only(self):
        # Panel unavailable (every agent errored) must NOT flip the brain's YES to a
        # NO — it degrades to a brain-only pass, matching the veto path. Previously
        # this hard-rejected, so an API throttle silently killed every buy.
        r = _panel()._aggregate(
            [_ar("risk", 0, veto=True, failed=True), _ar("momentum", 0, failed=True)]
        )
        assert r.conviction is ConvictionLevel.DEGRADED
        assert r.position_multiplier == 1.0
        assert r.should_trade is True
        assert r.veto_reason == "panel unavailable — brain-only"

    def test_relaxed_buy_threshold_passes_moderate_scores(self):
        # Established swing scores (risk 72 / momentum 50 / narrative 40): the
        # default 60 bar rejects (only risk qualifies), but a relaxed 48 bar lets
        # risk+momentum count → majority → trade.
        agents = [_ar("risk", 72, veto=True), _ar("momentum", 50), _ar("narrative", 40)]
        assert _panel()._aggregate(agents).should_trade is False  # default 60
        relaxed = _panel()._aggregate(agents, buy_threshold=48)
        assert relaxed.should_trade is True
        assert relaxed.conviction is ConvictionLevel.LOW_CONVICTION

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


def _ok_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _throttle_body() -> dict:
    # OpenRouter's rate-limit shape: HTTP 200, an error object, and NO choices.
    return {"error": {"message": "Rate limit exceeded", "code": 429}}


class _Resp:
    def __init__(self, status: int, body: dict):
        self.status = status
        self._body = body

    async def text(self):  # pragma: no cover - only on error paths
        return str(self._body)

    async def json(self, content_type=None):  # accept the content_type kwarg we pass
        return self._body


class _CM:
    def __init__(self, resp: _Resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return False


class _Session:
    """Returns a fixed successful body for every call."""

    def __init__(self, content: str):
        self._c = content
        self.calls = 0

    def post(self, url, headers=None, json=None):  # noqa: A002
        self.calls += 1
        return _CM(_Resp(200, _ok_body(self._c)))

    async def close(self):
        pass


class _SeqSession:
    """Returns a scripted sequence of (status, body) per agent, cycling through the
    same script for each of the 3 agents so retry behavior is deterministic."""

    def __init__(self, script: list[tuple[int, dict]]):
        self._script = script
        self.calls = 0
        self._per_agent: dict[int, int] = {}

    def post(self, url, headers=None, json=None):  # noqa: A002
        # Key the script position by agent persona so each agent walks the same
        # script independently regardless of interleave.
        content = (json or {})["messages"][0]["content"]
        idx = self._per_agent.get(hash(content), 0)
        self._per_agent[hash(content)] = idx + 1
        self.calls += 1
        status, body = self._script[min(idx, len(self._script) - 1)]
        return _CM(_Resp(status, body))

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

    @pytest.mark.asyncio
    async def test_throttle_then_success_is_retried(self, monkeypatch):
        # A 200-with-error throttle body (no choices) must be retried, not read as a
        # verdict — the exact failure that crashed every agent on 'choices'.
        monkeypatch.setattr("fenrir.ai.agent_panel.asyncio.sleep", _no_sleep)
        panel = _panel()
        panel._session = cast(
            aiohttp.ClientSession,
            _SeqSession(
                [
                    (200, _throttle_body()),  # attempt 1: throttled
                    (200, _ok_body('{"score": 78, "decision": "BUY", "reasoning": "ok"}')),
                ]
            ),
        )
        result = await panel.score("ctx", sol_amount=0.5)
        # Every agent recovered on retry → real verdict, not a crash-reject.
        assert result.conviction is ConvictionLevel.HIGH_CONVICTION
        assert all(not a.failed for a in result.agents)
        assert cast(_SeqSession, panel._session).calls == 6  # 3 agents x 2 attempts

    @pytest.mark.asyncio
    async def test_persistent_throttle_exhausts_retries_and_fails_gracefully(self, monkeypatch):
        # A throttle that never clears fails the agent WITHOUT raising, and the panel
        # degrades to brain-only rather than a hard reject.
        monkeypatch.setattr("fenrir.ai.agent_panel.asyncio.sleep", _no_sleep)
        panel = _panel()  # max_retries defaults to 2 → 3 attempts
        panel._session = cast(aiohttp.ClientSession, _SeqSession([(200, _throttle_body())]))
        result = await panel.score("ctx", sol_amount=0.5)
        assert all(a.failed for a in result.agents)
        assert result.conviction is ConvictionLevel.DEGRADED  # brain-only, not REJECT
        assert result.should_trade is True
        assert cast(_SeqSession, panel._session).calls == 9  # 3 agents x 3 attempts

    @pytest.mark.asyncio
    async def test_non_retryable_status_is_not_retried(self, monkeypatch):
        # A 400 (bad request) is a permanent error — fail fast, don't burn retries.
        monkeypatch.setattr("fenrir.ai.agent_panel.asyncio.sleep", _no_sleep)
        panel = _panel()
        panel._session = cast(
            aiohttp.ClientSession, _SeqSession([(400, {"error": {"message": "bad"}})])
        )
        result = await panel.score("ctx", sol_amount=0.5)
        assert all(a.failed for a in result.agents)
        assert cast(_SeqSession, panel._session).calls == 3  # 1 attempt per agent, no retry


class TestDropLenses:
    """Established (mid/large) tokens judge on risk + momentum, not meme-narrative."""

    @pytest.mark.asyncio
    async def test_drop_narrative_lets_safe_momentum_token_pass(self):
        # [risk=85, momentum=85, narrative=18]: with narrative it's minority-buy →
        # REJECT; dropping it → risk+momentum both clear → HIGH_CONVICTION.
        panel = _panel()

        def _content_score(json):  # noqa: A002
            c = json["messages"][0]["content"]
            if "RISK" in c or "RUG" in c:
                return 85
            if "MOMENTUM" in c:
                return 85
            return 18  # narrative

        class _ScoreSession:
            def __init__(self):
                self.calls = 0
                self.names = []

            def post(self, url, headers=None, json=None):  # noqa: A002
                self.calls += 1
                self.names.append(json["messages"][0]["content"][:40])
                s = _content_score(json)
                return _CM(_Resp(200, _ok_body(f'{{"score": {s}, "decision": "BUY"}}')))

            async def close(self):
                pass

        sess = _ScoreSession()
        panel._session = cast(aiohttp.ClientSession, sess)
        result = await panel.score("ctx", buy_threshold=48, drop_lenses={"narrative"})

        assert sess.calls == 2  # narrative agent never called
        assert {a.name for a in result.agents} == {"risk", "momentum"}
        assert result.conviction is ConvictionLevel.HIGH_CONVICTION
        assert result.should_trade is True

    @pytest.mark.asyncio
    async def test_dropping_narrative_keeps_the_risk_veto(self):
        # Even without narrative, an unsafe token (risk below floor) is still vetoed.
        panel = _panel()

        class _LowRisk:
            def __init__(self):
                self.calls = 0

            def post(self, url, headers=None, json=None):  # noqa: A002
                self.calls += 1
                c = json["messages"][0]["content"]
                s = 20 if ("RISK" in c or "RUG" in c) else 70
                return _CM(_Resp(200, _ok_body(f'{{"score": {s}, "decision": "SKIP"}}')))

            async def close(self):
                pass

        panel._session = cast(aiohttp.ClientSession, _LowRisk())
        result = await panel.score("ctx", buy_threshold=48, drop_lenses={"narrative"})
        assert result.conviction is ConvictionLevel.REJECT
        assert "risk" in (result.veto_reason or "")

    @pytest.mark.asyncio
    async def test_never_drops_the_last_veto_agent(self):
        # Dropping "risk" (the only veto) must be refused by score() — it would
        # disable the safety gate. The risk agent must still be evaluated.
        panel = _panel()
        panel._session = cast(aiohttp.ClientSession, _Session('{"score": 70, "decision": "BUY"}'))
        result = await panel.score("ctx", drop_lenses={"risk"})
        assert any(a.name == "risk" for a in result.agents)


class TestVetoOnly:
    """Risk-only (fresh-launch) path: run only the veto lens, aggregate as a veto."""

    def test_aggregate_veto_passes_when_safe(self):
        r = _panel()._aggregate_veto([_ar("risk", 55, veto=True)])
        assert r.conviction is ConvictionLevel.HIGH_CONVICTION
        assert r.position_multiplier == 1.0
        assert r.should_trade is True

    def test_aggregate_veto_rejects_below_floor(self):
        r = _panel()._aggregate_veto([_ar("risk", 18, veto=True)])
        assert r.conviction is ConvictionLevel.REJECT
        assert r.should_trade is False
        assert "risk" in (r.veto_reason or "")

    def test_aggregate_veto_failopen_when_risk_unavailable(self):
        r = _panel()._aggregate_veto([_ar("risk", 0, veto=True, failed=True)])
        # Flaky risk call must not block trading — degrade to a pass (brain-only).
        assert r.should_trade is True
        assert r.conviction is ConvictionLevel.DEGRADED

    @pytest.mark.asyncio
    async def test_score_veto_only_runs_only_risk_agent(self):
        panel = _panel()
        panel._session = cast(
            aiohttp.ClientSession, _Session('{"score": 72, "decision": "BUY", "reasoning": "ok"}')
        )
        result = await panel.score("ctx", sol_amount=0.01, veto_only=True)
        assert len(result.agents) == 1  # only the risk (veto) lens
        assert result.agents[0].name == "risk"
        assert cast(_Session, panel._session).calls == 1
        assert result.should_trade is True
