#!/usr/bin/env python3
"""
Tests for fenrir.ai.hedge_detector — HedgeDetector STM output normalization.

Covers:
  - Stripping all 8 epistemic hedge patterns
  - Stripping 4 preamble phrase patterns
  - hedge_count accuracy
  - Rolling hedge rate calculation
  - Drift detection threshold
  - reset() behaviour
  - get_stats() shape

Run with: pytest tests/test_hedge_detector.py -v
"""

import pytest

from fenrir.ai.hedge_detector import HedgeDetector


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def det():
    return HedgeDetector()


# ═══════════════════════════════════════════════════════════════════════════
#  BASIC CONTRACT
# ═══════════════════════════════════════════════════════════════════════════


class TestBasicContract:
    def test_returns_tuple(self, det):
        cleaned, count = det.process("Hello world")
        assert isinstance(cleaned, str)
        assert isinstance(count, int)

    def test_clean_response_unchanged(self, det):
        json_resp = '{"score": 75, "decision": "BUY", "reasoning": "Strong liquidity."}'
        cleaned, count = det.process(json_resp)
        assert count == 0
        # Content should be preserved
        assert '"score": 75' in cleaned

    def test_empty_string(self, det):
        cleaned, count = det.process("")
        assert cleaned == ""
        assert count == 0


# ═══════════════════════════════════════════════════════════════════════════
#  EPISTEMIC HEDGE STRIPPING
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicHedges:
    @pytest.mark.parametrize("phrase,expected_count", [
        ("I think this token is a good buy.", 1),
        ("I believe the confidence is high.", 1),
        ("Perhaps the liquidity is sufficient.", 1),
        ("maybe this will pump.", 1),
        ("It seems like a solid project.", 1),
        ("This is potentially profitable.", 1),
        ("The token might perform well.", 1),
        ("This could be a good entry.", 1),
        ("The confidence is approximately 0.75.", 1),
    ])
    def test_single_hedge_removed(self, det, phrase, expected_count):
        cleaned, count = det.process(phrase)
        assert count == expected_count
        # The phrase itself should be removed or altered
        hedge_words = [
            "I think", "I believe", "Perhaps", "maybe", "It seems",
            "potentially", "might", "could be", "approximately"
        ]
        for hw in hedge_words:
            if hw.lower() in phrase.lower():
                assert hw.lower() not in cleaned.lower()

    def test_multiple_hedges_counted(self, det):
        text = "I think this might perhaps be a good buy."
        cleaned, count = det.process(text)
        # "I think", "might", "perhaps" = 3 hedges
        assert count >= 3

    def test_hedge_in_json_still_counted(self, det):
        json_with_hedge = (
            '{"score": 70, "decision": "BUY", '
            '"reasoning": "I think this token has potential, maybe worth buying."}'
        )
        cleaned, count = det.process(json_with_hedge)
        assert count >= 2

    def test_case_insensitive(self, det):
        _, count = det.process("PERHAPS this is good. MAYBE it will work.")
        assert count >= 2


# ═══════════════════════════════════════════════════════════════════════════
#  PREAMBLE STRIPPING
# ═══════════════════════════════════════════════════════════════════════════


class TestPreambleStripping:
    @pytest.mark.parametrize("preamble", [
        "Based on the analysis, the token looks strong.",
        "Looking at the metrics, I recommend a buy.",
        "Given the liquidity, this seems promising.",
        "Considering that the holder count is 500, this is worth buying.",
    ])
    def test_preamble_removed(self, det, preamble):
        cleaned, count = det.process(preamble)
        assert count >= 1

    def test_preamble_at_line_start_removed(self, det):
        text = "First analysis complete.\nBased on the data, BUY.\nDone."
        cleaned, count = det.process(text)
        assert count >= 1
        # Preamble phrase should be stripped
        assert "Based on the data" not in cleaned

    def test_midline_based_on_not_stripped(self, det):
        # "based on" mid-sentence is less clearly a preamble; patterns are
        # line-start anchored (^) with re.M, so mid-sentence is safe
        text = 'The decision is based on multiple factors.'
        cleaned, count = det.process(text)
        # Mid-sentence "based on" should not be stripped by line-start pattern
        # (it's not at the start of a line)
        assert "based on multiple factors" in cleaned.lower()


# ═══════════════════════════════════════════════════════════════════════════
#  ROLLING HEDGE RATE
# ═══════════════════════════════════════════════════════════════════════════


class TestRollingHedgeRate:
    def test_zero_rate_initially(self, det):
        assert det.get_rolling_hedge_rate() == 0.0

    def test_rate_after_clean_responses(self, det):
        for _ in range(5):
            det.process('{"score": 80}')
        assert det.get_rolling_hedge_rate() == 0.0

    def test_rate_after_all_hedged(self, det):
        for _ in range(10):
            det.process("I think perhaps this might be good.")
        assert det.get_rolling_hedge_rate() == 1.0

    def test_rate_half(self, det):
        for _ in range(5):
            det.process("Clean JSON response.")
        for _ in range(5):
            det.process("I think maybe this is a buy.")
        rate = det.get_rolling_hedge_rate()
        assert pytest.approx(rate, abs=0.05) == 0.5

    def test_window_enforced(self, det):
        # Fill 10 clean, then 10 hedged — only hedged should remain
        for _ in range(10):
            det.process("Clean response.")
        for _ in range(10):
            det.process("I think perhaps maybe.")
        rate = det.get_rolling_hedge_rate()
        assert rate == 1.0  # Window only shows last 10 (all hedged)

    def test_avg_hedge_count(self, det):
        det.process("I think maybe perhaps.")  # 3 hedges
        det.process("Clean.")                  # 0 hedges
        avg = det.get_rolling_avg_hedge_count()
        assert avg == pytest.approx(1.5, abs=0.5)


# ═══════════════════════════════════════════════════════════════════════════
#  DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestDriftDetection:
    def test_not_drifting_initially(self, det):
        assert det.is_drifting() is False

    def test_not_drifting_with_partial_window(self, det):
        # Only 5 samples, window=10 → not enough to declare drift
        for _ in range(5):
            det.process("I think perhaps this might be good.")
        assert det.is_drifting() is False

    def test_drifting_when_above_threshold(self, det):
        # 7/10 hedged = 70% > 60% threshold
        for _ in range(7):
            det.process("I think perhaps.")
        for _ in range(3):
            det.process("Clean.")
        assert det.is_drifting() is True

    def test_not_drifting_below_threshold(self, det):
        # 5/10 hedged = 50% < 60% threshold
        for _ in range(5):
            det.process("I think perhaps.")
        for _ in range(5):
            det.process("Clean.")
        assert det.is_drifting() is False

    def test_custom_threshold(self):
        det = HedgeDetector(window_size=5, drift_threshold=0.4)
        for _ in range(3):  # 3/5 = 60% > 40%
            det.process("I think perhaps.")
        for _ in range(2):
            det.process("Clean.")
        assert det.is_drifting() is True


# ═══════════════════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════


class TestStateManagement:
    def test_reset_clears_window(self, det):
        for _ in range(10):
            det.process("I think perhaps maybe.")
        assert det.is_drifting() is True
        det.reset()
        assert det.is_drifting() is False
        assert det.get_rolling_hedge_rate() == 0.0

    def test_get_stats_shape(self, det):
        for i in range(5):
            det.process("I think maybe." if i % 2 == 0 else "Clean.")
        stats = det.get_stats()
        assert "window_size" in stats
        assert "samples" in stats
        assert "rolling_hedge_rate" in stats
        assert "avg_hedge_count" in stats
        assert "is_drifting" in stats
        assert "drift_threshold" in stats
        assert stats["samples"] == 5

    def test_stats_is_drifting_consistent(self, det):
        for _ in range(10):
            det.process("I think perhaps maybe.")
        stats = det.get_stats()
        assert stats["is_drifting"] == det.is_drifting()


# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT QUALITY
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputQuality:
    def test_json_still_parseable_after_cleaning(self, det):
        import json
        response = (
            'I think the analysis shows:\n'
            '```json\n'
            '{"score": 75, "decision": "BUY", "reasoning": "Good fundamentals."}\n'
            '```'
        )
        cleaned, count = det.process(response)
        # JSON block should still be findable
        j0 = cleaned.find("{")
        j1 = cleaned.rfind("}") + 1
        assert j0 != -1
        data = json.loads(cleaned[j0:j1])
        assert data["score"] == 75

    def test_no_double_spaces_after_stripping(self, det):
        text = "I think maybe this is perhaps a good buy."
        cleaned, _ = det.process(text)
        assert "  " not in cleaned  # No double spaces

    def test_stripping_does_not_break_numbers(self, det):
        text = "I think the score is about 75 and maybe the risk is approximately 5."
        cleaned, count = det.process(text)
        assert count > 0
        # Numbers should survive
        assert "75" in cleaned
        assert "5" in cleaned
