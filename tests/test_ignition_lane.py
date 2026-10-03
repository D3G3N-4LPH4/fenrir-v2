"""Tests for the rules-only block-zero ignition gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fenrir.trading.ignition_lane import IgnitionDecision, evaluate_ignition


def _token(**overrides) -> dict:
    base = {
        "symbol": "TEST",
        "creator": "creator111111111111111111111111111111111",
        "initial_liquidity_sol": 2.0,
        "market_cap_sol": 5.0,
        "bonding_curve_state": None,
    }
    base.update(overrides)
    return base


def test_pass_returns_fixed_buy() -> None:
    d = evaluate_ignition(_token())
    assert d == IgnitionDecision(True, 0.05, "TEST rules pass")


def test_blocked_creator_skips() -> None:
    d = evaluate_ignition(_token(), blocked_creators={"creator111111111111111111111111111111111"})
    assert not d.buy
    assert d.amount_sol == 0.0
    assert "creator blocked" in d.reason


def test_low_liquidity_skips() -> None:
    d = evaluate_ignition(_token(initial_liquidity_sol=0.1))
    assert not d.buy
    assert "liquidity" in d.reason


def test_missing_liquidity_treated_as_zero() -> None:
    d = evaluate_ignition(_token(initial_liquidity_sol=None))
    assert not d.buy


def test_high_mcap_skips() -> None:
    d = evaluate_ignition(_token(market_cap_sol=31.0))
    assert not d.buy
    assert "mcap" in d.reason


def test_already_migrated_skips() -> None:
    curve = SimpleNamespace(complete=True)
    d = evaluate_ignition(_token(bonding_curve_state=curve))
    assert not d.buy
    assert "migrated" in d.reason


def test_live_curve_passes() -> None:
    curve = SimpleNamespace(complete=False)
    d = evaluate_ignition(_token(bonding_curve_state=curve))
    assert d.buy


def test_custom_amount_and_thresholds() -> None:
    d = evaluate_ignition(
        _token(initial_liquidity_sol=1.0, market_cap_sol=25.0),
        min_liquidity_sol=0.5,
        max_market_cap_sol=30.0,
        amount_sol=0.1,
    )
    assert d.buy
    assert d.amount_sol == pytest.approx(0.1)


def test_decision_is_immutable() -> None:
    d = evaluate_ignition(_token())
    with pytest.raises(AttributeError):
        d.buy = False  # type: ignore[misc]
