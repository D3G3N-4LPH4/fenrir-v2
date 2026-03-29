#!/usr/bin/env python3
"""
FENRIR - EnsembleScorer: Parallel Model Racing with Confidence Gating

G0DM0D3 lineage: G0DM0D3 identified single-model calls as a single point
of failure for high-value positions. Independent model verification provides:

  1. SIGNAL QUALITY — Agreement between two independent models (different
     training data, RLHF, architecture) is a stronger BUY signal than
     either model alone.

  2. FAULT TOLERANCE — If the primary model times out or errors,
     the secondary model's response still drives a (DEGRADED) decision
     instead of silently falling back to conservative defaults.

  3. LATENCY OPTIMISATION — Haiku is 3-5x faster than Sonnet. For
     positions below the SOL threshold, haiku-only is the right trade-off.
     The ensemble fires only when it matters.

Agreement logic:
  Both ≥ threshold (70/100) → HIGH_CONVICTION: full position size
  Both < threshold          → REJECT: no trade
  Diverge > 20 pts          → LOW_CONVICTION: 50% position, log reason
  One fails / times out     → DEGRADED: surviving model's result, flagged

Integration with BudgetTracker / capability gating:
    result = await scorer.score(context, sol_amount=0.5)
    if not result.should_trade:
        return
    actual_sol = requested_sol * result.position_multiplier
    auth = budget_tracker.authorize_trade(..., amount_sol=actual_sol)

Usage:
    scorer = EnsembleScorer(api_key="sk-...")
    await scorer.initialize()

    result = await scorer.score(token_context_str, sol_amount=0.5)
    print(result.conviction)           # HIGH_CONVICTION
    print(result.position_multiplier)  # 1.0 or 0.5
    print(result.final_score)          # 0–100

    await scorer.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

import aiohttp

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────


class ConvictionLevel(Enum):
    """
    Outcome of the ensemble agreement algorithm.

    HIGH_CONVICTION  Both models agree BUY. Full position.
    LOW_CONVICTION   Models disagree (divergence > threshold). 50% position.
    REJECT           Both models say SKIP, or neither could produce a score.
    DEGRADED         One model failed; surviving model drives the decision.
                     Conviction is HIGH/LOW based on the survivor's score,
                     but the DEGRADED flag ensures it's logged prominently.
    """

    HIGH_CONVICTION = "high_conviction"
    LOW_CONVICTION = "low_conviction"
    REJECT = "reject"
    DEGRADED = "degraded"


@dataclass
class ModelScore:
    """Score returned by a single model in the ensemble."""

    model: str
    score: float          # 0–100 (confidence * 100 normalised)
    decision: str         # "BUY" or "SKIP"
    reasoning: str
    failed: bool = False
    error: str | None = None


@dataclass
class EnsembleResult:
    """
    Combined output of the two-model ensemble.

    Use .should_trade and .position_multiplier for position sizing.
    Pass .conviction to BudgetTracker capability gates.
    """

    conviction: ConvictionLevel
    primary: ModelScore
    secondary: ModelScore | None   # None when sol_amount < sol_threshold
    position_multiplier: float     # 1.0 | 0.5 | 0.0
    final_score: float             # Consensus score (avg if both, primary if solo)
    disagreement_reason: str | None = None

    @property
    def should_trade(self) -> bool:
        """True for HIGH_CONVICTION, LOW_CONVICTION, and DEGRADED (with caveat)."""
        return self.conviction in (
            ConvictionLevel.HIGH_CONVICTION,
            ConvictionLevel.LOW_CONVICTION,
            ConvictionLevel.DEGRADED,
        ) and self.position_multiplier > 0.0


# ─────────────────────────────────────────────────────────────
#  Prompt template
# ─────────────────────────────────────────────────────────────

_SCORE_PROMPT = """\
You are scoring a pump.fun memecoin launch for a trading bot.

{context}

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
  "score": <integer 0-100>,
  "decision": "BUY" or "SKIP",
  "reasoning": "<one sentence>"
}}

Score ≥ 70 = BUY signal. Score < 70 = SKIP.
Be conservative. Most memecoins rug or go to zero.
"""


# ─────────────────────────────────────────────────────────────
#  EnsembleScorer
# ─────────────────────────────────────────────────────────────


class EnsembleScorer:
    """
    Two-model parallel scoring with conviction gating.

    G0DM0D3 lineage: Replaces the single-model Claude call for positions
    above the SOL threshold. Independent model agreement provides a stronger
    and more trustworthy signal than any single model alone, while preserving
    the fast haiku-only path for small positions where latency matters most.
    """

    OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"

    # Primary: fast + cheap; Secondary: independent architecture
    PRIMARY_MODEL = "anthropic/claude-3-5-haiku"
    SECONDARY_MODEL = "openai/gpt-4o-mini"

    def __init__(
        self,
        api_key: str,
        sol_threshold: float = 0.5,         # Below → primary only
        score_threshold: float = 70.0,       # BUY vs SKIP cutoff
        divergence_threshold: float = 20.0,  # Score gap → LOW_CONVICTION
        timeout_seconds: float = 10.0,
    ):
        self.api_key = api_key
        self.sol_threshold = sol_threshold
        self.score_threshold = score_threshold
        self.divergence_threshold = divergence_threshold
        self.timeout = timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        """Open the shared aiohttp session."""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    # ── Public API ────────────────────────────────────────────

    async def score(self, context: str, sol_amount: float = 0.0) -> EnsembleResult:
        """
        Score a token using one or two models based on position size.

        Args:
            context: Formatted token analysis context string (e.g. from
                     ClaudeBrain._build_analysis_prompt or smc_adapter).
            sol_amount: Planned position size in SOL. Determines whether
                        the full ensemble fires or haiku-only.

        Returns:
            EnsembleResult with conviction level and position multiplier.
        """
        await self.initialize()
        prompt = _SCORE_PROMPT.format(context=context)

        if sol_amount < self.sol_threshold:
            # Fast path: primary model only
            primary = await self._call_model(self.PRIMARY_MODEL, prompt)
            return self._solo_result(primary, secondary=None)

        # Full ensemble: both models in parallel
        raw = await asyncio.gather(
            self._call_model(self.PRIMARY_MODEL, prompt),
            self._call_model(self.SECONDARY_MODEL, prompt),
            return_exceptions=True,
        )

        primary = (
            raw[0]
            if isinstance(raw[0], ModelScore)
            else ModelScore(
                model=self.PRIMARY_MODEL,
                score=0.0,
                decision="SKIP",
                reasoning="call raised exception",
                failed=True,
                error=str(raw[0]),
            )
        )
        secondary = (
            raw[1]
            if isinstance(raw[1], ModelScore)
            else ModelScore(
                model=self.SECONDARY_MODEL,
                score=0.0,
                decision="SKIP",
                reasoning="call raised exception",
                failed=True,
                error=str(raw[1]),
            )
        )

        return self._agreement_logic(primary, secondary)

    # ── Internal helpers ──────────────────────────────────────

    async def _call_model(self, model: str, prompt: str) -> ModelScore:
        """POST to OpenRouter and parse the score JSON."""
        if not self._session:
            await self.initialize()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 150,
        }

        try:
            async with self._session.post(
                self.OPENROUTER_API, headers=headers, json=payload
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("EnsembleScorer %s: HTTP %d — %s", model, resp.status, text[:80])
                    return ModelScore(
                        model=model,
                        score=0.0,
                        decision="SKIP",
                        reasoning=f"API error {resp.status}",
                        failed=True,
                        error=text[:100],
                    )
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse(model, content)

        except asyncio.TimeoutError:
            logger.warning("EnsembleScorer %s: timeout after %.1fs", model, self.timeout)
            return ModelScore(
                model=model,
                score=0.0,
                decision="SKIP",
                reasoning="timeout",
                failed=True,
                error="timeout",
            )
        except Exception as exc:
            logger.warning("EnsembleScorer %s: %s", model, exc)
            return ModelScore(
                model=model,
                score=0.0,
                decision="SKIP",
                reasoning="exception",
                failed=True,
                error=str(exc),
            )

    def _parse(self, model: str, response: str) -> ModelScore:
        """Extract score JSON from a raw model response."""
        try:
            j0 = response.find("{")
            j1 = response.rfind("}") + 1
            if j0 == -1 or j1 == 0:
                raise ValueError("No JSON object in response")
            data = json.loads(response[j0:j1])
            return ModelScore(
                model=model,
                score=float(data.get("score", 0)),
                decision=str(data.get("decision", "SKIP")).upper(),
                reasoning=str(data.get("reasoning", ""))[:200],
            )
        except Exception as exc:
            logger.debug("EnsembleScorer parse error (%s): %s | raw=%r", model, exc, response[:120])
            return ModelScore(
                model=model,
                score=0.0,
                decision="SKIP",
                reasoning=f"parse error: {exc}",
                failed=True,
            )

    def _solo_result(self, primary: ModelScore, secondary: ModelScore | None) -> EnsembleResult:
        """Build EnsembleResult from a single model (no ensemble race)."""
        if primary.failed:
            return EnsembleResult(
                conviction=ConvictionLevel.REJECT,
                primary=primary,
                secondary=secondary,
                position_multiplier=0.0,
                final_score=0.0,
            )
        if primary.score >= self.score_threshold:
            return EnsembleResult(
                conviction=ConvictionLevel.HIGH_CONVICTION,
                primary=primary,
                secondary=secondary,
                position_multiplier=1.0,
                final_score=primary.score,
            )
        return EnsembleResult(
            conviction=ConvictionLevel.REJECT,
            primary=primary,
            secondary=secondary,
            position_multiplier=0.0,
            final_score=primary.score,
        )

    def _agreement_logic(self, primary: ModelScore, secondary: ModelScore) -> EnsembleResult:
        """Apply the full ensemble agreement rules."""
        p_ok = not primary.failed
        s_ok = not secondary.failed

        # ── One model failed ──────────────────────────────────
        if p_ok and not s_ok:
            result = self._solo_result(primary, secondary)
            result.conviction = ConvictionLevel.DEGRADED
            logger.warning(
                "EnsembleScorer: DEGRADED — secondary (%s) failed: %s",
                secondary.model,
                secondary.error,
            )
            return result

        if s_ok and not p_ok:
            result = self._solo_result(secondary, None)
            result.primary = primary
            result.secondary = secondary
            result.conviction = ConvictionLevel.DEGRADED
            logger.warning(
                "EnsembleScorer: DEGRADED — primary (%s) failed: %s",
                primary.model,
                primary.error,
            )
            return result

        if not p_ok and not s_ok:
            return EnsembleResult(
                conviction=ConvictionLevel.REJECT,
                primary=primary,
                secondary=secondary,
                position_multiplier=0.0,
                final_score=0.0,
                disagreement_reason="both models failed",
            )

        # ── Both succeeded ────────────────────────────────────
        p_score = primary.score
        s_score = secondary.score
        divergence = abs(p_score - s_score)
        avg_score = (p_score + s_score) / 2.0

        both_buy = (p_score >= self.score_threshold and s_score >= self.score_threshold)
        both_skip = (p_score < self.score_threshold and s_score < self.score_threshold)

        if both_buy:
            return EnsembleResult(
                conviction=ConvictionLevel.HIGH_CONVICTION,
                primary=primary,
                secondary=secondary,
                position_multiplier=1.0,
                final_score=avg_score,
            )

        if both_skip:
            return EnsembleResult(
                conviction=ConvictionLevel.REJECT,
                primary=primary,
                secondary=secondary,
                position_multiplier=0.0,
                final_score=avg_score,
            )

        # Models disagree — check divergence magnitude
        if divergence > self.divergence_threshold:
            reason = (
                f"{self.PRIMARY_MODEL.split('/')[-1]}={p_score:.0f} vs "
                f"{self.SECONDARY_MODEL.split('/')[-1]}={s_score:.0f} "
                f"(divergence={divergence:.0f}pts > threshold={self.divergence_threshold:.0f})"
            )
            logger.warning("EnsembleScorer: LOW_CONVICTION — %s", reason)
            return EnsembleResult(
                conviction=ConvictionLevel.LOW_CONVICTION,
                primary=primary,
                secondary=secondary,
                position_multiplier=0.5,
                final_score=avg_score,
                disagreement_reason=reason,
            )

        # Mild disagreement within tolerance: use average to decide
        conviction = (
            ConvictionLevel.HIGH_CONVICTION
            if avg_score >= self.score_threshold
            else ConvictionLevel.REJECT
        )
        multiplier = 1.0 if conviction == ConvictionLevel.HIGH_CONVICTION else 0.0
        return EnsembleResult(
            conviction=conviction,
            primary=primary,
            secondary=secondary,
            position_multiplier=multiplier,
            final_score=avg_score,
        )
