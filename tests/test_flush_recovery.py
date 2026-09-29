"""Tests for the flush_recovery strategy: the post-flush second leg.

Covers the FLUSH_RECOVERY entry filter (fenrir/discovery/filters.py), the
FlushRecoveryStrategy playbook (fenrir/strategies/flush_recovery.py), and the
sparsity_aligned scoring preset (fenrir/discovery/scoring.py).

The pattern (kioto's $100M-runner structure): a coin flushes -60%..-95% in
24h, the knife stops (1h/5m stabilized), buyers step back in, and the holder
base survives the flush (supply migrated to strong hands) — the setup that
precedes the parabolic second leg.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from fenrir.config import BotConfig  # noqa: E402
from fenrir.discovery.filters import FilterEngine, FilterName  # noqa: E402
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot  # noqa: E402
from fenrir.discovery.playbooks import PLAYBOOK_STRATEGY_IDS, PlaybookTagger  # noqa: E402
from fenrir.discovery.scoring import ScoringEngine, ScoringWeights  # noqa: E402
from fenrir.strategies import (  # noqa: E402
    get_strategy_class,
    is_enabled_by_default,
)
from fenrir.strategies.flush_recovery import FlushRecoveryStrategy  # noqa: E402

TOKEN = {"token_address": "FlushRec111111111111111111111111111111111111"}


def _safe() -> SafetySignals:
    return SafetySignals(
        mint_disabled=True,
        freeze_disabled=True,
        lp_locked_or_burned=True,
        contract_verified=True,
        blacklist_present=False,
        honeypot=False,
    )


def _flushed() -> TokenSnapshot:
    """Post-flush snapshot that should PASS every flush_recovery gate.

    $5M mcap, -72% 24h flush, 1h -2% / 5m +1% (stabilized), 1h b/s 1.3 with
    0.6 5m pressure (buyers returning), $200k LP (4% of mcap), $1.5M 24h vol
    (0.3x turnover, 8% in the last hour), 2 days old, distributed supply,
    holder base intact through the flush (0.95x resilience).
    """
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address=TOKEN["token_address"],
        symbol="FLUSHED",
        market_cap_usd=5_000_000,
        liquidity_usd=200_000,
        volume_24h_usd=1_500_000,
        volume_1h_usd=120_000,
        age_minutes=2 * 24 * 60,
        holder_count=12_000,
        txns_1h_buys=1_300,
        txns_1h_sells=1_000,  # 1h b/s = 1.3
        txns_5m_buys=60,
        txns_5m_sells=40,  # 5m pressure 0.6
        price_change_5m_pct=1.0,
        price_change_1h_pct=-2.0,
        price_change_24h_pct=-72.0,  # the flush
        top_holder_pct=12.0,
        top10_holder_pct=45.0,
        accel_holder_growth=0.95,  # holders survived the flush
        safety=_safe(),
    )


def _eval(snap):
    return FlushRecoveryStrategy(BotConfig()).evaluate_token(TOKEN, snap)


# ── Registry ───────────────────────────────────────────────────────────


def test_registered():
    assert get_strategy_class("flush_recovery") is FlushRecoveryStrategy
    assert "flush_recovery" in PLAYBOOK_STRATEGY_IDS
    assert not is_enabled_by_default("flush_recovery")  # opt-in, tagging only


def test_filter_registered():
    assert FilterName.FLUSH_RECOVERY.value == "flush_recovery"
    engine = FilterEngine()
    res = engine.evaluate(_flushed(), FilterName.FLUSH_RECOVERY)
    assert res.passed, f"expected pass, failures={res.failures}"


# ── Strategy entry gates ───────────────────────────────────────────────


def test_passes_post_flush_setup():
    sig = _eval(_flushed())
    assert sig is not None
    assert sig.token_address == TOKEN["token_address"]
    assert 0.0 < sig.flush_score <= 1.0
    assert sig.metadata["strategy"] == "flush_recovery"
    assert sig.holder_resilience == 0.95


def test_rejects_fresh_launch():
    assert _eval(_flushed_with(age_minutes=300.0)) is None  # under the 6h floor


def test_rejects_shallow_dip():
    # -30% is a range_rotation dip, not a flush.
    assert _eval(_flushed_with(price_change_24h_pct=-30.0)) is None


def test_rejects_dead_coin():
    # -97% went to zero and stayed there — not a setup.
    assert _eval(_flushed_with(price_change_24h_pct=-97.0)) is None


def test_rejects_still_dumping_1h():
    assert _eval(_flushed_with(price_change_1h_pct=-25.0)) is None


def test_rejects_no_stabilization_5m():
    assert _eval(_flushed_with(price_change_5m_pct=-8.0)) is None


def test_rejects_no_buy_edge():
    snap = _flushed_with(
        txns_1h_buys=900,
        txns_1h_sells=1_000,  # b/s 0.9
        txns_5m_buys=40,
        txns_5m_sells=60,
    )  # pressure 0.4
    assert _eval(snap) is None


def test_rejects_holder_flight():
    # Holders fled the flush — no supply migration, no second leg.
    assert _eval(_flushed_with(accel_holder_growth=0.60)) is None


def test_passes_without_holder_data():
    # Fail-open when acceleration data is missing (coverage varies).
    assert _eval(_flushed_with(accel_holder_growth=None)) is not None


def test_rejects_concentrated_supply():
    assert _eval(_flushed_with(top_holder_pct=35.0)) is None


def test_rejects_thin_liquidity():
    assert _eval(_flushed_with(liquidity_usd=20_000.0)) is None


def test_rejects_dead_tape():
    assert _eval(_flushed_with(volume_24h_usd=50_000.0)) is None


def _flushed_with(**over) -> TokenSnapshot:
    snap = _flushed()
    for k, v in over.items():
        setattr(snap, k, v)
    return snap


# ── Filter gates ───────────────────────────────────────────────────────


def test_filter_rejects_shallow_dip_with_clear_message():
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(price_change_24h_pct=-30.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed
    assert any("not a flush" in f for f in res.failures)


def test_filter_rejects_dead_coin():
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(price_change_24h_pct=-97.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed
    assert any("dead, not a flush" in f for f in res.failures)


def test_filter_rejects_still_dumping():
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(price_change_1h_pct=-25.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed


def test_filter_rejects_vertical_second_leg():
    # +80% 1h means the second leg already went vertical — don't chase it.
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(price_change_1h_pct=80.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed


def test_filter_rejects_no_buy_edge():
    engine = FilterEngine()
    res = engine.evaluate(
        _flushed_with(txns_1h_buys=900, txns_1h_sells=1_000), FilterName.FLUSH_RECOVERY
    )
    assert not res.passed
    assert any("buy/sell" in f for f in res.failures)


def test_filter_rejects_below_mcap_floor():
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(market_cap_usd=200_000.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed
    assert any("MCap" in f for f in res.failures)


def test_filter_rejects_above_mcap_ceiling():
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(market_cap_usd=250_000_000.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed


def test_filter_rejects_thin_lp():
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(liquidity_usd=10_000.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed


def test_filter_rejects_young_coin():
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(age_minutes=120.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed


def test_filter_rejects_concentrated_top_holder():
    engine = FilterEngine()
    res = engine.evaluate(_flushed_with(top_holder_pct=35.0), FilterName.FLUSH_RECOVERY)
    assert not res.passed


# ── Playbook tagger ────────────────────────────────────────────────────


def test_tagger_tags_flush_recovery():
    tags = PlaybookTagger().tag(_flushed())
    assert "flush_recovery" in tags.playbook_ids


# ── Sparsity-aligned scoring preset ────────────────────────────────────


def test_sparsity_aligned_preset_maps_to_index_areas():
    w = ScoringWeights.sparsity_aligned()
    # Conviction 50% = holder + safety; demand 25% = momentum;
    # health 12.5% = liquidity; attention 12.5% = community.
    # Asserted as relative proportions (the 0.10 risk stabilizer sits
    # outside the four Sparsity areas): 4 : 2 : 1 : 1.
    conviction = w.holder + w.safety
    assert conviction / w.momentum == pytest.approx(2.0)
    assert w.momentum / w.liquidity == pytest.approx(2.0)
    assert w.liquidity / w.community == pytest.approx(1.0)


def test_sparsity_aligned_preset_scores():
    engine = ScoringEngine(ScoringWeights.sparsity_aligned())
    bd = engine.score(_flushed())
    assert 0.0 <= bd.overall <= 100.0
    # Defaults are untouched — the preset is opt-in, not a retune.
    d = ScoringWeights()
    assert (d.momentum, d.safety, d.liquidity, d.holder, d.community) == (
        0.25,
        0.25,
        0.20,
        0.15,
        0.05,
    )
