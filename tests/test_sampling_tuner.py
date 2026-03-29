#!/usr/bin/env python3
"""
Tests for fenrir.ai.sampling_tuner — SamplingTuner EMA parameter adaptation.

Covers:
  - Default params per regime
  - SamplingParams clamping
  - EMA update on positive outcome (drift toward params_used)
  - EMA update on negative outcome (drift toward defaults)
  - Effective alpha scales with |reward|
  - SQLite persistence (save + reload)
  - ART export record format
  - reset() behaviour

Run with: pytest tests/test_sampling_tuner.py -v
"""

import os
import sqlite3
import tempfile

import pytest

from fenrir.ai.sampling_tuner import (
    EMA_ALPHA,
    MarketRegime,
    SamplingParams,
    SamplingTuner,
    _DEFAULTS,
)


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_tuner.db")


@pytest.fixture
def tuner(db_path):
    return SamplingTuner(db_path=db_path)


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaults:
    def test_all_regimes_have_defaults(self):
        for regime in MarketRegime:
            assert regime in _DEFAULTS
            p = _DEFAULTS[regime]
            assert 0.1 <= p.temperature <= 0.9
            assert 0.1 <= p.top_p <= 1.0
            assert 0.0 <= p.frequency_penalty <= 2.0

    def test_initial_params_match_defaults(self, tuner):
        for regime in MarketRegime:
            p = tuner.get_params(regime)
            d = _DEFAULTS[regime]
            assert p.temperature == pytest.approx(d.temperature, abs=1e-6)
            assert p.top_p == pytest.approx(d.top_p, abs=1e-6)

    def test_snipe_is_fast_and_decisive(self):
        p = _DEFAULTS[MarketRegime.SNIPE]
        assert p.temperature <= 0.4  # Decisive

    def test_high_volatility_is_exploratory(self):
        p = _DEFAULTS[MarketRegime.HIGH_VOLATILITY]
        assert p.temperature > _DEFAULTS[MarketRegime.SNIPE].temperature


# ═══════════════════════════════════════════════════════════════════════════
#  SamplingParams
# ═══════════════════════════════════════════════════════════════════════════


class TestSamplingParams:
    def test_clamp_temperature_min(self):
        p = SamplingParams(temperature=0.0).clamp()
        assert p.temperature == 0.1

    def test_clamp_temperature_max(self):
        p = SamplingParams(temperature=2.0).clamp()
        assert p.temperature == 0.9

    def test_clamp_top_p(self):
        p = SamplingParams(top_p=1.5).clamp()
        assert p.top_p == 1.0

    def test_clamp_frequency_penalty(self):
        p = SamplingParams(frequency_penalty=-1.0).clamp()
        assert p.frequency_penalty == 0.0

    def test_to_dict(self):
        p = SamplingParams(temperature=0.3, top_p=0.9, frequency_penalty=0.0)
        d = p.to_dict()
        assert d["temperature"] == 0.3
        assert d["top_p"] == 0.9
        assert d["frequency_penalty"] == 0.0

    def test_to_api_kwargs_excludes_penalty(self):
        p = SamplingParams(temperature=0.3, top_p=0.9, frequency_penalty=0.5)
        kwargs = p.to_api_kwargs()
        assert "temperature" in kwargs
        assert "top_p" in kwargs
        assert "frequency_penalty" not in kwargs  # Not all APIs accept this


# ═══════════════════════════════════════════════════════════════════════════
#  EMA UPDATE — POSITIVE OUTCOME
# ═══════════════════════════════════════════════════════════════════════════


class TestEMAPositiveOutcome:
    def test_positive_pnl_nudges_toward_params_used(self, tuner):
        regime = MarketRegime.SNIPE
        default_temp = _DEFAULTS[regime].temperature
        # Use params slightly different from default
        params_used = SamplingParams(temperature=default_temp + 0.1, top_p=0.9)
        before = tuner.get_params(regime).temperature

        tuner.record_outcome(regime, params_used, pnl_pct=50.0)

        after = tuner.get_params(regime).temperature
        # Should drift toward params_used.temperature (which is higher)
        assert after > before

    def test_positive_pnl_reward_scales_alpha(self, tuner):
        regime = MarketRegime.SNIPE
        default_temp = _DEFAULTS[regime].temperature
        params_used = SamplingParams(temperature=default_temp + 0.2, top_p=0.9)

        # Large win: reward=1.0, effective_alpha = EMA_ALPHA * 1.0 = 0.1
        tuner.record_outcome(regime, params_used, pnl_pct=100.0)
        after_big = tuner.get_params(regime).temperature

        # Reset
        tuner.reset(regime)

        # Small win: reward=0.1, effective_alpha = EMA_ALPHA * 0.1 = 0.01
        tuner.record_outcome(regime, params_used, pnl_pct=10.0)
        after_small = tuner.get_params(regime).temperature

        # Bigger win should produce larger drift
        delta_big = abs(after_big - default_temp)
        delta_small = abs(after_small - default_temp)
        assert delta_big > delta_small


# ═══════════════════════════════════════════════════════════════════════════
#  EMA UPDATE — NEGATIVE OUTCOME
# ═══════════════════════════════════════════════════════════════════════════


class TestEMANegativeOutcome:
    def test_negative_pnl_nudges_toward_defaults(self, tuner):
        regime = MarketRegime.HIGH_VOLATILITY
        # First move params away from default with a positive outcome
        default_temp = _DEFAULTS[regime].temperature
        away_params = SamplingParams(temperature=default_temp + 0.2, top_p=0.9)
        tuner.record_outcome(regime, away_params, pnl_pct=100.0)
        moved_temp = tuner.get_params(regime).temperature
        assert moved_temp > default_temp

        # Now record a loss — params should drift back toward default
        current_params = tuner.get_params(regime)
        tuner.record_outcome(regime, current_params, pnl_pct=-50.0)
        after_loss_temp = tuner.get_params(regime).temperature
        assert after_loss_temp < moved_temp  # Reverted toward default

    def test_large_loss_clamps_at_bounds(self, tuner):
        regime = MarketRegime.SNIPE
        extreme = SamplingParams(temperature=0.5, top_p=0.9)
        # Many large losses won't push below minimum
        for _ in range(50):
            tuner.record_outcome(regime, extreme, pnl_pct=-100.0)
        p = tuner.get_params(regime)
        assert p.temperature >= 0.1
        assert p.top_p >= 0.1


# ═══════════════════════════════════════════════════════════════════════════
#  TRADE COUNT
# ═══════════════════════════════════════════════════════════════════════════


class TestTradeCount:
    def test_trade_count_increments(self, tuner):
        regime = MarketRegime.SNIPE
        params = tuner.get_params(regime)
        for i in range(5):
            tuner.record_outcome(regime, params, pnl_pct=10.0)
        all_params = tuner.get_all_params()
        assert all_params[regime.value]["trade_count"] == 5

    def test_different_regimes_tracked_independently(self, tuner):
        params = SamplingParams()
        tuner.record_outcome(MarketRegime.SNIPE, params, pnl_pct=10.0)
        tuner.record_outcome(MarketRegime.SNIPE, params, pnl_pct=10.0)
        tuner.record_outcome(MarketRegime.GRADUATION, params, pnl_pct=10.0)
        all_params = tuner.get_all_params()
        assert all_params[MarketRegime.SNIPE.value]["trade_count"] == 2
        assert all_params[MarketRegime.GRADUATION.value]["trade_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════
#  SQLite PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════


class TestPersistence:
    def test_params_survive_restart(self, db_path):
        # Session 1: record several trades
        t1 = SamplingTuner(db_path=db_path)
        params_used = SamplingParams(temperature=0.5, top_p=0.9)
        for _ in range(5):
            t1.record_outcome(MarketRegime.SNIPE, params_used, pnl_pct=80.0)
        snapshot = t1.get_params(MarketRegime.SNIPE)

        # Session 2: fresh instance, same db
        t2 = SamplingTuner(db_path=db_path)
        restored = t2.get_params(MarketRegime.SNIPE)
        assert restored.temperature == pytest.approx(snapshot.temperature, abs=1e-4)
        assert restored.top_p == pytest.approx(snapshot.top_p, abs=1e-4)

    def test_trade_count_survives_restart(self, db_path):
        t1 = SamplingTuner(db_path=db_path)
        params = SamplingParams()
        for _ in range(3):
            t1.record_outcome(MarketRegime.GRADUATION, params, pnl_pct=20.0)

        t2 = SamplingTuner(db_path=db_path)
        assert t2.get_all_params()[MarketRegime.GRADUATION.value]["trade_count"] == 3

    def test_db_table_created(self, db_path):
        SamplingTuner(db_path=db_path)
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "sampling_tuner" in table_names


# ═══════════════════════════════════════════════════════════════════════════
#  ART EXPORT
# ═══════════════════════════════════════════════════════════════════════════


class TestARTExport:
    def test_to_art_record_shape(self, tuner):
        regime = MarketRegime.SNIPE
        params = tuner.get_params(regime)
        record = tuner.to_art_record(regime, params, pnl_pct=+42.0)
        assert record["regime"] == "snipe"
        assert "params_used" in record
        assert "current_ema" in record
        assert "default" in record
        assert record["pnl_pct"] == pytest.approx(42.0)
        assert -1.0 <= record["reward"] <= 1.0

    def test_reward_clamp(self, tuner):
        regime = MarketRegime.SNIPE
        params = tuner.get_params(regime)
        record = tuner.to_art_record(regime, params, pnl_pct=9999.0)
        assert record["reward"] == pytest.approx(1.0)

        record2 = tuner.to_art_record(regime, params, pnl_pct=-9999.0)
        assert record2["reward"] == pytest.approx(-1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  RESET
# ═══════════════════════════════════════════════════════════════════════════


class TestReset:
    def test_reset_single_regime(self, tuner):
        regime = MarketRegime.SNIPE
        params_used = SamplingParams(temperature=0.8, top_p=0.9)
        for _ in range(10):
            tuner.record_outcome(regime, params_used, pnl_pct=100.0)
        assert tuner.get_params(regime).temperature != _DEFAULTS[regime].temperature

        tuner.reset(regime)
        assert tuner.get_params(regime).temperature == pytest.approx(
            _DEFAULTS[regime].temperature, abs=1e-4
        )

    def test_reset_all(self, tuner):
        params_used = SamplingParams(temperature=0.8, top_p=0.9)
        for regime in MarketRegime:
            tuner.record_outcome(regime, params_used, pnl_pct=100.0)
        tuner.reset()
        for regime in MarketRegime:
            assert tuner.get_params(regime).temperature == pytest.approx(
                _DEFAULTS[regime].temperature, abs=1e-4
            )

    def test_reset_all_persisted(self, db_path):
        t = SamplingTuner(db_path=db_path)
        params_used = SamplingParams(temperature=0.8, top_p=0.9)
        t.record_outcome(MarketRegime.SNIPE, params_used, pnl_pct=100.0)
        t.reset(MarketRegime.SNIPE)

        t2 = SamplingTuner(db_path=db_path)
        assert t2.get_params(MarketRegime.SNIPE).temperature == pytest.approx(
            _DEFAULTS[MarketRegime.SNIPE].temperature, abs=1e-4
        )
