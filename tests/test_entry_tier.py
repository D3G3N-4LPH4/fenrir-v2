"""Tests for fenrir.discovery.entry_tier — the ignition/standard/late split."""

from fenrir.discovery.entry_tier import (
    TIER_IGNITION,
    TIER_LATE,
    TIER_STANDARD,
    classify_entry_tier,
    tier_alerts,
)


def _cand(**kw):
    base = {
        "passed_filters": ["volatility_breakout"],
        "price_change_1h_pct": 60.0,
        "bond_progress_pct": 40.0,
        "age_minutes": 25,
    }
    base.update(kw)
    return base


def test_ignition_filter_wins():
    assert classify_entry_tier(_cand(passed_filters=["curve_ignition"])) == TIER_IGNITION
    assert classify_entry_tier(_cand(passed_filters=["graduation_watch"])) == TIER_IGNITION


def test_ignition_beats_late_signals():
    # early curve + vertical move = the good stuff, not a late entry
    c = _cand(
        passed_filters=["curve_ignition", "volatility_breakout"],
        price_change_1h_pct=95.0,
        bond_progress_pct=30.0,
    )
    assert classify_entry_tier(c) == TIER_IGNITION


def test_vb_deep_into_band_is_late():
    c = _cand(passed_filters=["volatility_breakout"], price_change_1h_pct=95.0)
    assert classify_entry_tier(c) == TIER_LATE


def test_vb_graduated_curve_is_late():
    c = _cand(
        passed_filters=["volatility_breakout"],
        price_change_1h_pct=55.0,
        bond_progress_pct=92.0,
    )
    assert classify_entry_tier(c) == TIER_LATE


def test_vb_mid_move_is_standard():
    assert classify_entry_tier(_cand()) == TIER_STANDARD


def test_non_vb_is_standard():
    assert classify_entry_tier(_cand(passed_filters=["mid_cap_momentum"])) == TIER_STANDARD


def test_missing_signals_stay_standard():
    # fail-open: unknown 1h move / no curve data never demotes to late
    c = _cand(
        passed_filters=["volatility_breakout"],
        price_change_1h_pct=None,
        bond_progress_pct=None,
    )
    assert classify_entry_tier(c) == TIER_STANDARD


def test_tier_alerts():
    assert tier_alerts(TIER_IGNITION) is True
    assert tier_alerts(TIER_STANDARD) is True
    assert tier_alerts(TIER_LATE) is False
