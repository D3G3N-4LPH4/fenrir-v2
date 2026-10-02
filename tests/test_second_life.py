"""Tests for the second-life (SAPLING) model: floor baseline + re-ignition filter.

The model: d3g3n's 20x SAPLING looked identical to every dead launch at his
$183k entry. The discriminator was time (floor held for days, holders stayed)
plus re-ignition (volume/edge/price lifting off the base). These tests pin:
- dead-week tape does NOT fire second_life
- re-ignition tape DOES fire
- missing baseline fails closed
- a broken floor (went to zero vs its own range) fails
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.discovery.filters import FilterEngine, FilterName  # noqa: E402
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot  # noqa: E402
from fenrir.discovery.second_life import (  # noqa: E402
    build_baseline,
    parse_ohlcv,
    survived,
)


def _safe() -> SafetySignals:
    return SafetySignals(
        mint_disabled=True,
        freeze_disabled=True,
        lp_locked_or_burned=True,
        contract_verified=True,
        blacklist_present=False,
        honeypot=False,
    )


def _day(day: int, low: float, high: float, vol: float) -> list[tuple]:
    """24 hourly candles for one UTC day (vol = daily volume)."""
    return [(day * 86400 + h * 3600, low, high, low, low, vol / 24.0) for h in range(24)]


def _daily(ts: float, low: float, high: float, vol: float) -> tuple:
    return (ts, low, high, low, low, vol)  # (ts, o, h, l, c, v)


def _reigniting_snap() -> TokenSnapshot:
    """SAPLING-like at re-ignition: 6d old, floor held, tape exploding."""
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address="REIGNITE",
        price_usd=0.004,
        market_cap_usd=3_200_000,
        liquidity_usd=200_000,
        volume_24h_usd=2_000_000,
        volume_1h_usd=300_000,  # 5x the 60k baseline
        age_minutes=6 * 24 * 60,
        holder_count=2_900,
        txns_24h_buys=5_000,
        txns_24h_sells=3_000,
        txns_1h_buys=4_500,
        txns_1h_sells=2_800,  # 1.6x edge
        price_change_1h_pct=120.0,
        price_change_24h_pct=800.0,
        top_holder_pct=10.0,
        top10_holder_pct=40.0,
        safety=_safe(),
        # trailing baseline: floor $0.002 (2x below current), max $0.006
        base_floor_price_usd=0.002,
        base_max_price_usd=0.006,
        base_median_1h_volume_usd=60_000,
        base_lookback_days=7.0,
    )


def test_reignition_passes():
    engine = FilterEngine()
    r = engine.evaluate(_reigniting_snap(), FilterName.SECOND_LIFE)
    assert r.passed, f"expected pass, failures={r.failures}"


def test_dead_week_does_not_fire():
    """Same coin during the grind: no volume, no edge, price on the floor."""
    snap = _reigniting_snap()
    snap.volume_1h_usd = 20_000  # 0.33x baseline
    snap.txns_1h_buys = 100
    snap.txns_1h_sells = 120
    snap.price_usd = 0.0021  # 1.05x floor
    snap.price_change_1h_pct = 2.0
    engine = FilterEngine()
    r = engine.evaluate(snap, FilterName.SECOND_LIFE)
    assert not r.passed
    assert any("baseline" in f or "floor" in f for f in r.failures)


def test_missing_baseline_fails_closed():
    snap = _reigniting_snap()
    snap.base_floor_price_usd = None
    snap.base_max_price_usd = None
    snap.base_median_1h_volume_usd = None
    engine = FilterEngine()
    r = engine.evaluate(snap, FilterName.SECOND_LIFE)
    assert not r.passed
    assert any("baseline unknown" in f for f in r.failures)


def test_broken_floor_fails():
    """Coin that went to zero vs its own range: floor 2% of trailing max."""
    snap = _reigniting_snap()
    snap.base_floor_price_usd = 0.00012  # 2% of $0.006 max
    engine = FilterEngine()
    r = engine.evaluate(snap, FilterName.SECOND_LIFE)
    assert not r.passed
    assert any("floor broken" in f for f in r.failures)


def test_too_young_fails():
    snap = _reigniting_snap()
    snap.age_minutes = 60.0  # 1h old — no history to judge
    engine = FilterEngine()
    r = engine.evaluate(snap, FilterName.SECOND_LIFE)
    assert not r.passed


def test_build_baseline_needs_three_candles():
    assert build_baseline([_daily(1, 1.0, 2.0, 100), _daily(2, 1.0, 2.0, 100)]) is None


def test_build_baseline_floor_is_median_of_lows():
    candles = [c for i in range(1, 8) for c in _day(i, float(i), 10.0, 240.0)]
    b = build_baseline(candles)
    assert b is not None
    assert b.floor_price_usd == 3.5  # median of lows 1..6 (day 7 excluded)
    assert b.max_price_usd == 10.0
    assert b.low_price_usd == 1.0
    assert b.median_1h_volume_usd == 10.0  # 240/24
    assert b.lookback_days == 6  # latest day excluded


def test_build_baseline_excludes_ignition_day():
    # 6 quiet days + a spike today: the spike must not contaminate floor/max.
    candles = [c for i in range(6) for c in _day(i, 1.0, 2.0, 240.0)]
    candles += _day(6, 50.0, 100.0, 100000.0)
    b = build_baseline(candles)
    assert b is not None
    assert b.floor_price_usd == 1.0
    assert b.max_price_usd == 2.0
    assert b.lookback_days == 6


def test_survived_rejects_death():
    candles = [c for i in range(7) for c in _day(i, 0.01, 10.0, 100.0)]
    b = build_baseline(candles)
    assert b is not None
    assert not survived(b)


def test_survived_accepts_healthy():
    candles = [c for i in range(7) for c in _day(i, 4.0, 10.0, 100.0)]
    b = build_baseline(candles)
    assert b is not None
    assert survived(b)


def test_parse_ohlcv_malformed():
    assert parse_ohlcv(None) == []
    assert parse_ohlcv({}) == []
    assert parse_ohlcv({"data": {"attributes": {"ohlcv_list": "nope"}}}) == []
    assert parse_ohlcv({"data": {"attributes": {"ohlcv_list": [[1, 2]]}}}) == []


def test_parse_ohlcv_valid():
    payload = {
        "data": {
            "attributes": {
                "ohlcv_list": [
                    [1700000000, 1.0, 2.0, 0.5, 1.5, 1000.0],
                    [1700086400, 1.5, 3.0, 1.0, 2.5, 2000.0],
                ]
            }
        }
    }
    rows = parse_ohlcv(payload)
    assert len(rows) == 2
    assert rows[0] == (1700000000.0, 1.0, 2.0, 0.5, 1.5, 1000.0)
