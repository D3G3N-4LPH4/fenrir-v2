"""Tests for poll-over-poll acceleration tracking and the momentum_transition filter.

Covers fenrir/discovery/acceleration.py (AccelTracker: history, growth math,
persistence, pruning) and the momentum_transition entry filter
(fenrir/discovery/filters.py) — the pre-run catcher for the $75k-$2M mcap gap.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.discovery.acceleration import AccelTracker  # noqa: E402
from fenrir.discovery.filters import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    FilterEngine,
    FilterName,
)
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot  # noqa: E402


def _safe() -> SafetySignals:
    return SafetySignals(
        mint_disabled=True,
        freeze_disabled=True,
        lp_locked_or_burned=True,
        contract_verified=True,
        blacklist_present=False,
        honeypot=False,
    )


def _snap(**over) -> TokenSnapshot:
    """A snapshot that PASSES momentum_transition once acceleration is attached.

    $500k mcap, 2h old, $100k LP, $600k 24h vol with 5% in the last hour,
    1.3x 1h buy/sell, +25% 1h (moving, not vertical), distributed holders.
    """
    base: dict[str, Any] = dict(
        chain=Chain.SOLANA,
        token_address="ACCEL000000000000000000000000000000000000001",
        symbol="ACCEL",
        market_cap_usd=500_000.0,
        liquidity_usd=100_000.0,
        volume_24h_usd=600_000.0,
        volume_1h_usd=30_000.0,
        age_minutes=120.0,
        holder_count=160,
        txns_1h_buys=130,
        txns_1h_sells=100,
        txns_24h_buys=900,
        txns_24h_sells=700,
        price_change_1h_pct=25.0,
        price_change_24h_pct=60.0,
        top_holder_pct=10.0,
        top10_holder_pct=40.0,
        safety=_safe(),
        # Second-sighting acceleration: activity +80%, holders +60%, edge up.
        accel_txn_growth=1.8,
        accel_holder_growth=1.6,
        accel_edge_delta=0.05,
        accel_polls_seen=2,
    )
    base.update(over)
    return TokenSnapshot(**base)


def _eval(snap):
    return FilterEngine().evaluate(snap, FilterName.MOMENTUM_TRANSITION)


# ── Filter registration ──────────────────────────────────────────────


def test_registered_in_thresholds():
    assert FilterName.MOMENTUM_TRANSITION in DEFAULT_THRESHOLDS
    assert FilterName.MOMENTUM_TRANSITION.value == "momentum_transition"


# ── Filter: pass ─────────────────────────────────────────────────────


def test_passes_accelerating_coin():
    res = _eval(_snap())
    assert res.passed, f"expected pass, failures={res.failures}"


# ── Filter: acceleration gates ───────────────────────────────────────


def test_fails_closed_without_history():
    # First sighting seeds the baseline — the filter must not fire.
    res = _eval(
        _snap(
            accel_txn_growth=None,
            accel_holder_growth=None,
            accel_edge_delta=None,
            accel_polls_seen=0,
        )
    )
    assert not res.passed
    assert any("no acceleration history" in f for f in res.failures)


def test_fails_when_txn_growth_flat():
    res = _eval(_snap(accel_txn_growth=1.2))
    assert not res.passed
    assert any("txn growth" in f for f in res.failures)


def test_fails_when_holder_growth_flat():
    res = _eval(_snap(accel_holder_growth=1.1))
    assert not res.passed
    assert any("holder growth" in f for f in res.failures)


def test_fails_when_edge_not_improving():
    res = _eval(_snap(accel_edge_delta=-0.05))
    assert not res.passed
    assert any("buy-edge delta" in f for f in res.failures)


def test_holder_growth_missing_warns_not_fails():
    # Holder coverage varies by chain — missing data warns, txn accel decides.
    res = _eval(_snap(accel_holder_growth=None))
    assert res.passed, f"expected pass, failures={res.failures}"
    assert any("holder growth unavailable" in w for w in res.warnings)


# ── Filter: static gates ─────────────────────────────────────────────


def test_fails_below_mcap_band():
    res = _eval(_snap(market_cap_usd=50_000.0))
    assert not res.passed
    assert any("MCap" in f for f in res.failures)


def test_fails_above_mcap_band():
    res = _eval(_snap(market_cap_usd=3_000_000.0))
    assert not res.passed


def test_fails_on_vertical_1h():
    # The move already happened — this filter is for before, not after.
    res = _eval(_snap(price_change_1h_pct=150.0))
    assert not res.passed
    assert any("vertical" in f for f in res.failures)


def test_fails_without_buy_edge():
    res = _eval(_snap(txns_1h_buys=100, txns_1h_sells=100))
    assert not res.passed
    assert any("buy/sell" in f for f in res.failures)


def test_fails_on_concentrated_supply():
    res = _eval(_snap(top_holder_pct=25.0))
    assert not res.passed
    assert any("top holder" in f.lower() for f in res.failures)


def test_fails_when_too_young():
    res = _eval(_snap(age_minutes=5.0))
    assert not res.passed


# ── AccelTracker ─────────────────────────────────────────────────────


def _obs_snap(txns_buys, txns_sells, holders, addr="T1"):
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address=addr,
        txns_1h_buys=txns_buys,
        txns_1h_sells=txns_sells,
        holder_count=holders,
    )


def test_tracker_first_sighting_returns_none(tmp_path):
    t = AccelTracker(tmp_path / "accel.json")
    snap = _obs_snap(60, 40, 20)
    assert t.record(snap, now=1000.0) is None
    assert snap.accel_polls_seen == 1
    assert snap.accel_txn_growth is None


def _growth_val(growth: dict[str, float | None] | None, key: str) -> float:
    """Unwrap one acceleration metric; the tests only read keys that must be present."""
    assert growth is not None
    v = growth[key]
    assert v is not None
    return v


def test_tracker_computes_growth(tmp_path):
    t = AccelTracker(tmp_path / "accel.json")
    t.record(_obs_snap(60, 40, 20), now=1000.0)  # 100 txns, edge 0.6, 20 holders
    snap = _obs_snap(108, 72, 32)  # 180 txns, edge 0.6, 32 holders
    growth = t.record(snap, now=1600.0)
    assert growth is not None
    assert abs(_growth_val(growth, "accel_txn_growth") - 1.8) < 1e-9
    assert abs(_growth_val(growth, "accel_holder_growth") - 1.6) < 1e-9
    assert abs(_growth_val(growth, "accel_edge_delta") - 0.0) < 1e-9
    assert snap.accel_polls_seen == 2
    assert snap.accel_txn_growth is not None
    assert abs(snap.accel_txn_growth - 1.8) < 1e-9


def test_tracker_edge_delta(tmp_path):
    t = AccelTracker(tmp_path / "accel.json")
    t.record(_obs_snap(50, 50, 20), now=1000.0)  # edge 0.5
    snap = _obs_snap(70, 30, 25)  # edge 0.7
    growth = t.record(snap, now=1600.0)
    assert abs(_growth_val(growth, "accel_edge_delta") - 0.2) < 1e-9


def test_tracker_dead_tape_waking_up_caps_growth(tmp_path):
    t = AccelTracker(tmp_path / "accel.json")
    t.record(_obs_snap(0, 0, 5), now=1000.0)
    snap = _obs_snap(50, 30, 12)
    growth = t.record(snap, now=1600.0)
    assert _growth_val(growth, "accel_txn_growth") == 999.0  # capped "infinite"


def test_tracker_missing_holders_gives_none_growth(tmp_path):
    t = AccelTracker(tmp_path / "accel.json")
    s1 = _obs_snap(60, 40, 20)
    s1.holder_count = None
    t.record(s1, now=1000.0)
    snap = _obs_snap(108, 72, 32)
    growth = t.record(snap, now=1600.0)
    assert growth is not None
    assert growth["accel_holder_growth"] is None
    assert growth["accel_txn_growth"] is not None


def test_tracker_persists_and_prunes(tmp_path):
    import time as _time

    now = _time.time()
    path = tmp_path / "accel.json"
    t = AccelTracker(path)
    t.record(_obs_snap(60, 40, 20, addr="FRESH"), now=now)
    t.record(_obs_snap(60, 40, 20, addr="STALE"), now=now)
    t.save()

    t2 = AccelTracker(path)  # reload
    assert t2.polls_seen("FRESH") == 1

    # STALE's only observation is older than the 24h window → pruned.
    t3 = AccelTracker(path, max_age_hours=24.0)
    t3._hist["STALE"] = [{"ts": now - 100 * 3600, "txns_1h": 10, "holders": 5, "edge": 0.5}]
    assert t3.prune(now=now) == 1
    assert t3.polls_seen("STALE") == 0
    assert t3.polls_seen("FRESH") == 1


def test_tracker_ignores_empty_address(tmp_path):
    t = AccelTracker(tmp_path / "accel.json")
    snap = _obs_snap(60, 40, 20, addr="")
    assert t.record(snap) is None


# ── Curve ignition filter ────────────────────────────────────────────


def _ignition_snap(**over) -> TokenSnapshot:
    """A snapshot that PASSES curve_ignition: $40k mcap, 30m old, curve at 30%
    with 2.5 SOL of fresh inflow velocity, buy-leaning tape, clean-ish early
    distribution."""
    base: dict[str, Any] = dict(
        chain=Chain.SOLANA,
        token_address="IGNITE00000000000000000000000000000000000001",
        symbol="IGNITE",
        market_cap_usd=40_000.0,
        volume_24h_usd=50_000.0,
        volume_1h_usd=4_000.0,
        age_minutes=30.0,
        holder_count=25,
        txns_1h_buys=140,
        txns_1h_sells=100,
        txns_24h_buys=400,
        txns_24h_sells=300,
        price_change_1h_pct=35.0,
        top_holder_pct=12.0,
        sniper_pct=10.0,
        bundle_pct=8.0,
        bond_progress_pct=30.0,
        bond_inflow_sol=2.5,
        safety=_safe(),
    )
    base.update(over)
    return TokenSnapshot(**base)


def _eval_ignition(snap):
    return FilterEngine().evaluate(snap, FilterName.CURVE_IGNITION)


def test_ignition_registered():
    assert FilterName.CURVE_IGNITION in DEFAULT_THRESHOLDS
    assert FilterName.CURVE_IGNITION.value == "curve_ignition"


def test_ignition_passes():
    res = _eval_ignition(_ignition_snap())
    assert res.passed, f"expected pass, failures={res.failures}"


def test_ignition_fails_past_50pct():
    # 60% belongs to graduation_watch, not ignition.
    res = _eval_ignition(_ignition_snap(bond_progress_pct=60.0))
    assert not res.passed
    assert any("bond" in f.lower() for f in res.failures)


def test_ignition_fails_weak_inflow():
    res = _eval_ignition(_ignition_snap(bond_inflow_sol=0.2))
    assert not res.passed
    assert any("inflow" in f.lower() for f in res.failures)


def test_ignition_fails_without_bond_data():
    res = _eval_ignition(_ignition_snap(bond_progress_pct=None, bond_inflow_sol=None))
    assert not res.passed


def test_ignition_fails_stalled_curve():
    res = _eval_ignition(_ignition_snap(age_minutes=200.0))
    assert not res.passed
    assert any("Age" in f for f in res.failures)


def test_ignition_fails_sniper_farm():
    res = _eval_ignition(_ignition_snap(sniper_pct=40.0))
    assert not res.passed
    assert any("niper" in f for f in res.failures)


def test_ignition_fails_too_big():
    res = _eval_ignition(_ignition_snap(market_cap_usd=500_000.0))
    assert not res.passed


# ── hot_candidates ───────────────────────────────────────────────────


def test_hot_candidates_empty_without_acceleration(tmp_path):
    t = AccelTracker(tmp_path / "a.json")
    t.record(_obs_snap(60, 40, 20, addr="FLAT"), now=1000.0)
    t.record(_obs_snap(62, 41, 20, addr="FLAT"), now=1600.0)
    assert t.hot_candidates(now=1600.0) == []


def test_hot_candidates_picks_accelerating(tmp_path):
    t = AccelTracker(tmp_path / "a.json")
    t.record(_obs_snap(60, 40, 20, addr="HOT"), now=1000.0)
    t.record(_obs_snap(108, 72, 32, addr="HOT"), now=1600.0)  # 1.8x txns
    t.record(_obs_snap(60, 40, 20, addr="COLD"), now=1000.0)
    t.record(_obs_snap(61, 40, 20, addr="COLD"), now=1600.0)
    hot = t.hot_candidates(now=1600.0)
    assert hot == ["HOT"]


def test_hot_candidates_ignores_stale(tmp_path):
    t = AccelTracker(tmp_path / "a.json")
    t.record(_obs_snap(60, 40, 20, addr="OLD"), now=1000.0)
    t.record(_obs_snap(200, 50, 60, addr="OLD"), now=1600.0)
    # Observation is 2h old — outside the 30m window.
    assert t.hot_candidates(now=1600.0 + 7200) == []


def test_hot_candidates_respects_limit(tmp_path):
    t = AccelTracker(tmp_path / "a.json")
    for i in range(5):
        a = f"T{i}"
        t.record(_obs_snap(60, 40, 20, addr=a), now=1000.0)
        t.record(_obs_snap(120, 60, 30, addr=a), now=1600.0)
    assert len(t.hot_candidates(limit=3, now=1600.0)) == 3
