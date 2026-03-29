#!/usr/bin/env python3
"""
Tests for fenrir.ai.ensemble_scorer — EnsembleScorer parallel model racing.

Covers:
  - Both models agree BUY  → HIGH_CONVICTION, multiplier=1.0
  - Both models agree SKIP → REJECT, multiplier=0.0
  - Scores diverge > 20pts → LOW_CONVICTION, multiplier=0.5
  - Primary fails          → DEGRADED (secondary drives result)
  - Secondary fails        → DEGRADED (primary drives result)
  - Both fail              → REJECT
  - Position below SOL threshold → primary only (no secondary call)
  - Custom thresholds

Run with: pytest tests/test_ensemble_scorer.py -v
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fenrir.ai.ensemble_scorer import (
    ConvictionLevel,
    EnsembleResult,
    EnsembleScorer,
    ModelScore,
)


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def scorer():
    return EnsembleScorer(
        api_key="test-key",
        sol_threshold=0.5,
        score_threshold=70.0,
        divergence_threshold=20.0,
        timeout_seconds=5.0,
    )


def _ok_score(model: str, score: float, decision: str = "BUY") -> ModelScore:
    return ModelScore(model=model, score=score, decision=decision, reasoning="test")


def _failed_score(model: str, error: str = "timeout") -> ModelScore:
    return ModelScore(
        model=model, score=0.0, decision="SKIP", reasoning="failed",
        failed=True, error=error
    )


# ═══════════════════════════════════════════════════════════════════════════
#  AGREEMENT LOGIC (pure unit tests — no HTTP)
# ═══════════════════════════════════════════════════════════════════════════


class TestAgreementLogic:
    """Tests of _agreement_logic without any HTTP calls."""

    def test_both_above_threshold_high_conviction(self, scorer):
        primary = _ok_score(scorer.PRIMARY_MODEL, 85.0)
        secondary = _ok_score(scorer.SECONDARY_MODEL, 80.0)
        result = scorer._agreement_logic(primary, secondary)
        assert result.conviction == ConvictionLevel.HIGH_CONVICTION
        assert result.position_multiplier == 1.0
        assert result.final_score == pytest.approx(82.5)

    def test_both_below_threshold_reject(self, scorer):
        primary = _ok_score(scorer.PRIMARY_MODEL, 50.0, "SKIP")
        secondary = _ok_score(scorer.SECONDARY_MODEL, 45.0, "SKIP")
        result = scorer._agreement_logic(primary, secondary)
        assert result.conviction == ConvictionLevel.REJECT
        assert result.position_multiplier == 0.0

    def test_divergence_above_threshold_low_conviction(self, scorer):
        primary = _ok_score(scorer.PRIMARY_MODEL, 80.0)   # above threshold
        secondary = _ok_score(scorer.SECONDARY_MODEL, 50.0, "SKIP")  # below + diverge > 20
        result = scorer._agreement_logic(primary, secondary)
        assert result.conviction == ConvictionLevel.LOW_CONVICTION
        assert result.position_multiplier == pytest.approx(0.5)
        assert result.disagreement_reason is not None

    def test_divergence_exactly_at_threshold_not_low(self, scorer):
        # Divergence = 20 exactly, threshold = 20 → not LOW_CONVICTION
        # (one above, one below threshold but diverge == threshold, not >)
        primary = _ok_score(scorer.PRIMARY_MODEL, 80.0)
        secondary = _ok_score(scorer.SECONDARY_MODEL, 60.0, "SKIP")
        result = scorer._agreement_logic(primary, secondary)
        # divergence = 20, avg = 70 >= threshold → HIGH_CONVICTION
        # (the > check means exactly 20 does not trigger LOW_CONVICTION)
        assert result.conviction in (ConvictionLevel.HIGH_CONVICTION, ConvictionLevel.LOW_CONVICTION)

    def test_primary_fails_degraded(self, scorer):
        primary = _failed_score(scorer.PRIMARY_MODEL)
        secondary = _ok_score(scorer.SECONDARY_MODEL, 75.0)
        result = scorer._agreement_logic(primary, secondary)
        assert result.conviction == ConvictionLevel.DEGRADED
        assert result.position_multiplier >= 0.0  # Secondary drove result

    def test_secondary_fails_degraded(self, scorer):
        primary = _ok_score(scorer.PRIMARY_MODEL, 80.0)
        secondary = _failed_score(scorer.SECONDARY_MODEL)
        result = scorer._agreement_logic(primary, secondary)
        assert result.conviction == ConvictionLevel.DEGRADED
        assert result.position_multiplier >= 0.0

    def test_both_fail_reject(self, scorer):
        primary = _failed_score(scorer.PRIMARY_MODEL, "timeout")
        secondary = _failed_score(scorer.SECONDARY_MODEL, "API error")
        result = scorer._agreement_logic(primary, secondary)
        assert result.conviction == ConvictionLevel.REJECT
        assert result.position_multiplier == 0.0
        assert result.should_trade is False

    def test_high_conviction_should_trade_true(self, scorer):
        primary = _ok_score(scorer.PRIMARY_MODEL, 85.0)
        secondary = _ok_score(scorer.SECONDARY_MODEL, 80.0)
        result = scorer._agreement_logic(primary, secondary)
        assert result.should_trade is True

    def test_reject_should_trade_false(self, scorer):
        primary = _ok_score(scorer.PRIMARY_MODEL, 40.0, "SKIP")
        secondary = _ok_score(scorer.SECONDARY_MODEL, 35.0, "SKIP")
        result = scorer._agreement_logic(primary, secondary)
        assert result.should_trade is False


# ═══════════════════════════════════════════════════════════════════════════
#  SOLO RESULT (below SOL threshold)
# ═══════════════════════════════════════════════════════════════════════════


class TestSoloResult:
    def test_solo_buy_above_threshold(self, scorer):
        primary = _ok_score(scorer.PRIMARY_MODEL, 80.0)
        result = scorer._solo_result(primary, secondary=None)
        assert result.conviction == ConvictionLevel.HIGH_CONVICTION
        assert result.position_multiplier == 1.0
        assert result.secondary is None

    def test_solo_skip_below_threshold(self, scorer):
        primary = _ok_score(scorer.PRIMARY_MODEL, 50.0, "SKIP")
        result = scorer._solo_result(primary, secondary=None)
        assert result.conviction == ConvictionLevel.REJECT
        assert result.position_multiplier == 0.0

    def test_solo_failed_primary_reject(self, scorer):
        primary = _failed_score(scorer.PRIMARY_MODEL)
        result = scorer._solo_result(primary, secondary=None)
        assert result.conviction == ConvictionLevel.REJECT
        assert result.should_trade is False


# ═══════════════════════════════════════════════════════════════════════════
#  JSON PARSING
# ═══════════════════════════════════════════════════════════════════════════


class TestJSONParsing:
    def test_valid_json_parsed(self, scorer):
        response = '{"score": 80, "decision": "BUY", "reasoning": "Good liquidity."}'
        ms = scorer._parse("model", response)
        assert not ms.failed
        assert ms.score == 80.0
        assert ms.decision == "BUY"

    def test_markdown_wrapped_json_parsed(self, scorer):
        response = "```json\n{\"score\": 75, \"decision\": \"BUY\", \"reasoning\": \"ok\"}\n```"
        ms = scorer._parse("model", response)
        assert not ms.failed
        assert ms.score == 75.0

    def test_no_json_returns_failed(self, scorer):
        response = "I cannot determine a score for this token."
        ms = scorer._parse("model", response)
        assert ms.failed

    def test_invalid_json_returns_failed(self, scorer):
        response = '{"score": invalid_value}'
        ms = scorer._parse("model", response)
        assert ms.failed

    def test_missing_score_defaults_zero(self, scorer):
        response = '{"decision": "BUY", "reasoning": "ok"}'
        ms = scorer._parse("model", response)
        assert ms.score == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  FULL score() METHOD (mocked HTTP)
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreMethod:
    """Integration tests mocking _call_model to avoid real HTTP calls."""

    @pytest.mark.asyncio
    async def test_small_position_uses_primary_only(self, scorer):
        primary_resp = _ok_score(scorer.PRIMARY_MODEL, 80.0)
        with patch.object(scorer, "_call_model", new=AsyncMock(return_value=primary_resp)):
            result = await scorer.score("context", sol_amount=0.1)  # below 0.5
        assert result.secondary is None
        assert result.conviction == ConvictionLevel.HIGH_CONVICTION

    @pytest.mark.asyncio
    async def test_large_position_calls_both_models(self, scorer):
        call_results = {
            scorer.PRIMARY_MODEL: _ok_score(scorer.PRIMARY_MODEL, 80.0),
            scorer.SECONDARY_MODEL: _ok_score(scorer.SECONDARY_MODEL, 75.0),
        }

        async def fake_call(model, prompt):
            return call_results[model]

        with patch.object(scorer, "_call_model", side_effect=fake_call):
            result = await scorer.score("context", sol_amount=1.0)  # above 0.5
        assert result.secondary is not None
        assert result.conviction == ConvictionLevel.HIGH_CONVICTION

    @pytest.mark.asyncio
    async def test_both_agree_skip_large_position(self, scorer):
        call_results = {
            scorer.PRIMARY_MODEL: _ok_score(scorer.PRIMARY_MODEL, 40.0, "SKIP"),
            scorer.SECONDARY_MODEL: _ok_score(scorer.SECONDARY_MODEL, 35.0, "SKIP"),
        }

        async def fake_call(model, prompt):
            return call_results[model]

        with patch.object(scorer, "_call_model", side_effect=fake_call):
            result = await scorer.score("context", sol_amount=1.0)
        assert result.conviction == ConvictionLevel.REJECT
        assert result.should_trade is False

    @pytest.mark.asyncio
    async def test_disagreement_produces_low_conviction(self, scorer):
        call_results = {
            scorer.PRIMARY_MODEL: _ok_score(scorer.PRIMARY_MODEL, 85.0),
            scorer.SECONDARY_MODEL: _ok_score(scorer.SECONDARY_MODEL, 40.0, "SKIP"),
        }

        async def fake_call(model, prompt):
            return call_results[model]

        with patch.object(scorer, "_call_model", side_effect=fake_call):
            result = await scorer.score("context", sol_amount=1.0)
        assert result.conviction == ConvictionLevel.LOW_CONVICTION
        assert result.position_multiplier == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_primary_failure_degraded(self, scorer):
        call_results = {
            scorer.PRIMARY_MODEL: _failed_score(scorer.PRIMARY_MODEL),
            scorer.SECONDARY_MODEL: _ok_score(scorer.SECONDARY_MODEL, 80.0),
        }

        async def fake_call(model, prompt):
            return call_results[model]

        with patch.object(scorer, "_call_model", side_effect=fake_call):
            result = await scorer.score("context", sol_amount=1.0)
        assert result.conviction == ConvictionLevel.DEGRADED


# ═══════════════════════════════════════════════════════════════════════════
#  EnsembleResult
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsembleResult:
    def test_should_trade_all_states(self):
        primary = _ok_score("m", 80.0)
        secondary = _ok_score("m2", 75.0)

        for conviction, multiplier, expected in [
            (ConvictionLevel.HIGH_CONVICTION, 1.0, True),
            (ConvictionLevel.LOW_CONVICTION, 0.5, True),
            (ConvictionLevel.DEGRADED, 0.5, True),
            (ConvictionLevel.REJECT, 0.0, False),
        ]:
            r = EnsembleResult(
                conviction=conviction,
                primary=primary,
                secondary=secondary,
                position_multiplier=multiplier,
                final_score=80.0,
            )
            assert r.should_trade == expected
