#!/usr/bin/env python3
"""
FENRIR - MultiAgentPanel: role-specialized second-opinion evaluation

Extends the ensemble/second-opinion idea (EnsembleScorer = two general models
racing the same prompt) toward a small **multi-agent panel**: several agents that
each evaluate the SAME token context through a DIFFERENT specialized lens, in
parallel, then aggregate with a veto.

Default panel:
  - risk      (VETO)  — rug/safety analyst; a low safety score vetoes the trade
                        regardless of the others.
  - momentum         — entry timing / flow (buy pressure, volume, liquidity).
  - narrative        — meme/cultural virality (name originality, meta-fit).

Aggregation:
  - Any veto agent below `veto_floor`            → REJECT (unsafe).
  - All succeeded agents ≥ buy_threshold         → HIGH_CONVICTION (full size).
  - Majority (≥50%) ≥ buy_threshold              → LOW_CONVICTION (half size).
  - Fewer                                        → REJECT.
  - All agents failed                            → REJECT.
  - Some (not all) agents failed but a decision
    was still reached                            → flagged DEGRADED.

The result is duck-type compatible with EnsembleResult (`should_trade`,
`position_multiplier`, `conviction`) so it drops into the brain's ensemble gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import aiohttp

from fenrir.ai.ensemble_scorer import ConvictionLevel

logger = logging.getLogger(__name__)

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class Agent:
    """One specialized evaluator in the panel."""

    name: str
    persona: str  # system-prompt framing for this agent's lens
    veto: bool = False  # a low score from a veto agent rejects the trade


DEFAULT_AGENTS: list[Agent] = [
    Agent(
        name="risk",
        persona=(
            "You are a RISK & RUG-PULL analyst for pump.fun memecoins. Score how SAFE this token "
            "is to buy right now (0 = certain rug, 100 = very safe): weigh mint/freeze authority, "
            "LP, holder concentration, creator history, and liquidity depth. Be harsh — most "
            "memecoins are unsafe."
        ),
        veto=True,
    ),
    Agent(
        name="momentum",
        persona=(
            "You are a MOMENTUM & FLOW analyst. Score the entry timing (0-100): early buy pressure, "
            "volume acceleration, buy/sell ratio, and liquidity depth. Higher = a stronger momentum "
            "entry right now."
        ),
    ),
    Agent(
        name="narrative",
        persona=(
            "You are a MEME / NARRATIVE analyst. Score cultural virality potential (0-100): "
            "name/symbol originality, fit with an active meta-narrative (AI agents, dog breeds, "
            "current memes), and community signals. Higher = stronger narrative with room to run."
        ),
    ),
]

_AGENT_PROMPT = """\
{persona}

# TOKEN CONTEXT
{context}

Respond ONLY with valid JSON (no markdown):
{{"score": <integer 0-100>, "decision": "BUY" or "SKIP", "reasoning": "<one sentence>"}}
Score >= {threshold} means BUY from your lens.
"""


@dataclass
class AgentResult:
    """One agent's verdict."""

    name: str
    score: float  # 0-100 from this agent's lens
    decision: str  # BUY | SKIP
    reasoning: str
    veto: bool = False
    failed: bool = False
    error: str | None = None


@dataclass
class PanelResult:
    """Aggregated panel verdict — duck-type compatible with EnsembleResult."""

    conviction: ConvictionLevel
    position_multiplier: float  # 1.0 | 0.5 | 0.0
    final_score: float  # avg of succeeded agents (0-100)
    agents: list[AgentResult] = field(default_factory=list)
    veto_reason: str | None = None

    @property
    def should_trade(self) -> bool:
        return (
            self.conviction
            in (
                ConvictionLevel.HIGH_CONVICTION,
                ConvictionLevel.LOW_CONVICTION,
                ConvictionLevel.DEGRADED,
            )
            and self.position_multiplier > 0.0
        )

    def summary(self) -> str:
        parts = [f"{a.name}={a.score:.0f}{'✗' if a.failed else ''}" for a in self.agents]
        return f"{self.conviction.value} [{', '.join(parts)}]" + (
            f" veto:{self.veto_reason}" if self.veto_reason else ""
        )


class MultiAgentPanel:
    """Role-specialized multi-agent second opinion. Same context, different lenses."""

    def __init__(
        self,
        api_key: str,
        model: str,
        agents: list[Agent] | None = None,
        buy_threshold: float = 60.0,
        veto_floor: float = 40.0,
        timeout_seconds: float = 10.0,
    ):
        self.api_key = api_key
        self.model = model
        self.agents = agents if agents is not None else DEFAULT_AGENTS
        self.buy_threshold = buy_threshold
        self.veto_floor = veto_floor
        self.timeout = timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def score(
        self, context: str, sol_amount: float = 0.0, veto_only: bool = False
    ) -> PanelResult:
        """Evaluate `context` with the panel in parallel and aggregate.

        `sol_amount` is accepted for interface parity with EnsembleScorer.

        `veto_only` runs ONLY the veto (risk/safety) agents and aggregates them as
        a pure safety veto — for data-poor fresh launches, where the momentum and
        narrative lenses have no volume/social data to score and would otherwise
        reject every snipe. The risk lens still evaluates mint/LP/holder/creator
        signals, so unsafe tokens are still vetoed; safe-enough ones pass.
        """
        await self.initialize()
        agents = [a for a in self.agents if a.veto] if veto_only else self.agents
        raw = await asyncio.gather(
            *(self._ask(agent, context) for agent in agents),
            return_exceptions=True,
        )
        results: list[AgentResult] = []
        for agent, r in zip(agents, raw, strict=True):
            if isinstance(r, AgentResult):
                results.append(r)
            else:
                results.append(
                    AgentResult(agent.name, 0.0, "SKIP", "call raised", agent.veto, True, str(r))
                )
        return self._aggregate_veto(results) if veto_only else self._aggregate(results)

    def _aggregate_veto(self, results: list[AgentResult]) -> PanelResult:
        """Pure safety-veto aggregation for the risk-only (fresh-launch) path.

        Reject only if a veto agent scores below the safety floor; otherwise pass
        at full size. If the risk lens is unavailable (all calls failed), degrade
        to a pass (brain-only) rather than block trading on a flaky call.
        """
        ok = [r for r in results if not r.failed]
        if not ok:
            return PanelResult(
                ConvictionLevel.DEGRADED, 1.0, 0.0, results, "risk lens unavailable — brain-only"
            )
        for r in ok:
            if r.veto and r.score < self.veto_floor:
                return PanelResult(
                    ConvictionLevel.REJECT,
                    0.0,
                    r.score,
                    results,
                    f"{r.name} safety {r.score:.0f} < floor {self.veto_floor:.0f}",
                )
        avg = sum(r.score for r in ok) / len(ok)
        return PanelResult(ConvictionLevel.HIGH_CONVICTION, 1.0, avg, results)

    async def _ask(self, agent: Agent, context: str) -> AgentResult:
        assert self._session is not None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        prompt = _AGENT_PROMPT.format(
            persona=agent.persona, context=context, threshold=int(self.buy_threshold)
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 150,
        }
        try:
            async with self._session.post(OPENROUTER_API, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("Agent %s: HTTP %d — %s", agent.name, resp.status, text[:80])
                    return AgentResult(
                        agent.name, 0.0, "SKIP", "api error", agent.veto, True, text[:100]
                    )
                data = await resp.json()
                return self._parse(agent, data["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001 - one agent failing shouldn't sink the panel
            logger.warning("Agent %s: %s", agent.name, exc)
            return AgentResult(agent.name, 0.0, "SKIP", "exception", agent.veto, True, str(exc))

    def _parse(self, agent: Agent, response: str) -> AgentResult:
        try:
            j0, j1 = response.find("{"), response.rfind("}") + 1
            if j0 == -1 or j1 == 0:
                raise ValueError("no JSON object")
            data = json.loads(response[j0:j1])
            return AgentResult(
                name=agent.name,
                score=float(data.get("score", 0)),
                decision=str(data.get("decision", "SKIP")).upper(),
                reasoning=str(data.get("reasoning", ""))[:200],
                veto=agent.veto,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Agent %s parse error: %s | raw=%r", agent.name, exc, response[:120])
            return AgentResult(agent.name, 0.0, "SKIP", f"parse error: {exc}", agent.veto, True)

    def _aggregate(self, results: list[AgentResult]) -> PanelResult:
        ok = [r for r in results if not r.failed]
        if not ok:
            return PanelResult(ConvictionLevel.REJECT, 0.0, 0.0, results, "all agents failed")

        # Veto: any veto agent below the safety floor kills the trade.
        for r in ok:
            if r.veto and r.score < self.veto_floor:
                return PanelResult(
                    ConvictionLevel.REJECT,
                    0.0,
                    r.score,
                    results,
                    f"{r.name} safety {r.score:.0f} < floor {self.veto_floor:.0f}",
                )

        avg = sum(r.score for r in ok) / len(ok)
        buys = sum(1 for r in ok if r.score >= self.buy_threshold)
        ratio = buys / len(ok)
        degraded = len(ok) < len(results)  # some agent failed but we still decided

        if ratio == 1.0:
            conviction = ConvictionLevel.HIGH_CONVICTION
            mult = 1.0
        elif ratio >= 0.5:
            conviction = ConvictionLevel.LOW_CONVICTION
            mult = 0.5
        else:
            return PanelResult(ConvictionLevel.REJECT, 0.0, avg, results)

        if degraded:
            conviction = ConvictionLevel.DEGRADED
        return PanelResult(conviction, mult, avg, results)
