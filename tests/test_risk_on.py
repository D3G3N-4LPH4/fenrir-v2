"""Tests for the risk-on discovery tier: degen_launch / volatility_breakout
filters plus the degen_ignition / volatility_breakout playbook strategies."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.config import BotConfig  # noqa: E402
from fenrir.discovery.filters import FilterEngine, FilterName  # noqa: E402
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot  # noqa: E402
from fenrir.discovery.playbooks import PlaybookTagger  # noqa: E402
from fenrir.strategies.degen_ignition import DegenIgnitionStrategy  # noqa: E402
from fenrir.strategies.volatility_breakout import VolatilityBreakoutStrategy  # noqa: E402


def _safe() -> SafetySignals:
    return SafetySignals(
        mint_disabled=True,
        freeze_disabled=True,
        lp_locked_or_burned=True,
        contract_verified=True,
        blacklist_present=False,
        honeypot=False,
    )


def _degen() -> TokenSnapshot:
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address="DEGEN",
        market_cap_usd=8_000,
        liquidity_usd=2_000,
        volume_24h_usd=5_000,
        volume_1h_usd=1_000,
        age_minutes=20,
        holder_count=60,
        txns_24h_buys=40,
        txns_24h_sells=8,
        txns_1h_buys=25,
        txns_1h_sells=8,
        price_change_5m_pct=5.0,
        price_change_1h_pct=30.0,
        price_change_24h_pct=60.0,
        top_holder_pct=12.0,
        top10_holder_pct=60.0,
        dev_wallet_pct=8.0,
        bond_progress_pct=30.0,
        sniper_pct=15.0,
        bundle_pct=10.0,
        safety=_safe(),
    )


def _breakout() -> TokenSnapshot:
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address="BRK",
        market_cap_usd=300_000,
        liquidity_usd=40_000,
        volume_24h_usd=300_000,
        volume_1h_usd=60_000,
        age_minutes=300,
        holder_count=800,
        txns_24h_buys=500,
        txns_24h_sells=200,
        txns_1h_buys=120,
        txns_1h_sells=60,
        price_change_5m_pct=3.0,
        price_change_1h_pct=80.0,
        price_change_24h_pct=120.0,
        top_holder_pct=8.0,
        top10_holder_pct=45.0,
        dev_wallet_pct=5.0,
        safety=_safe(),
    )


# ── Filters ───────────────────────────────────────────────────────────


def test_degen_launch_passes_ideal() -> None:
    r = FilterEngine().evaluate(_degen(), FilterName.DEGEN_LAUNCH)
    assert r.passed, r.failures


def test_degen_launch_rejects_old_token() -> None:
    s = _degen()
    s.age_minutes = 120
    r = FilterEngine().evaluate(s, FilterName.DEGEN_LAUNCH)
    assert not r.passed


def test_degen_launch_rejects_weak_buy_edge() -> None:
    s = _degen()
    s.txns_1h_buys, s.txns_1h_sells = 10, 10  # ratio 1.0 < 1.5
    r = FilterEngine().evaluate(s, FilterName.DEGEN_LAUNCH)
    assert not r.passed


def test_volatility_breakout_passes_ideal() -> None:
    r = FilterEngine().evaluate(_breakout(), FilterName.VOLATILITY_BREAKOUT)
    assert r.passed, r.failures


def test_volatility_breakout_rejects_flat_token() -> None:
    s = _breakout()
    s.price_change_1h_pct = 10.0  # no breakout
    r = FilterEngine().evaluate(s, FilterName.VOLATILITY_BREAKOUT)
    assert not r.passed


def test_volatility_breakout_rejects_sell_driven_wick() -> None:
    s = _breakout()
    s.txns_1h_buys, s.txns_1h_sells = 60, 120  # ratio 0.5
    r = FilterEngine().evaluate(s, FilterName.VOLATILITY_BREAKOUT)
    assert not r.passed


# ── volatility_breakout hardening (2026-10-01: SIC/HIHI/JANE/s/acc died,
# Janes +122% lived, SuperCali fading) ──────────────────────────────────


def _janes_like() -> TokenSnapshot:
    """Janes clearance shape: +60.1% 1h, 5.75x edge → passed, +122%."""
    s = _breakout()
    s.price_change_1h_pct = 60.1
    s.txns_1h_buys, s.txns_1h_sells = 575, 100  # ratio 5.75
    return s


def test_volatility_breakout_keeps_janes_shape() -> None:
    r = FilterEngine().evaluate(_janes_like(), FilterName.VOLATILITY_BREAKOUT)
    assert r.passed, r.failures


def test_volatility_breakout_rejects_blowoff_top() -> None:
    """JANE +144%/1h and SuperCali +243%/1h: past ~+120% the move is the exit."""
    for chg in (144.0, 243.0):
        s = _janes_like()
        s.price_change_1h_pct = chg
        r = FilterEngine().evaluate(s, FilterName.VOLATILITY_BREAKOUT)
        assert not r.passed, chg
        assert any("vertical" in f for f in r.failures)


def test_volatility_breakout_rejects_noise_edge() -> None:
    """HIHI 1.54x: 1.3–1.5x is tape noise, not a buy-driven move."""
    s = _janes_like()
    s.txns_1h_buys, s.txns_1h_sells = 154, 100  # ratio 1.54
    r = FilterEngine().evaluate(s, FilterName.VOLATILITY_BREAKOUT)
    assert not r.passed


def test_volatility_breakout_rejects_one_sided_edge() -> None:
    """SIC 11.67x / JANE 9.35x: extreme edge on a vertical move is painted."""
    for buys, sells in ((1167, 100), (935, 100), (200, 0)):  # 11.67, 9.35, inf
        s = _janes_like()
        s.txns_1h_buys, s.txns_1h_sells = buys, sells
        r = FilterEngine().evaluate(s, FilterName.VOLATILITY_BREAKOUT)
        assert not r.passed, (buys, sells)
        assert any("one-sided" in f for f in r.failures)


def test_buy_edge_ceiling_off_by_default() -> None:
    """Only the vertical-move filters set the ceiling — volatility_breakout and
    second_life (both read extreme buy edges as one-sided painter flow)."""
    eng = FilterEngine()
    ceiling_filters = {FilterName.VOLATILITY_BREAKOUT, FilterName.SECOND_LIFE}
    for name, thr in eng.thresholds.items():
        if name in ceiling_filters:
            assert thr.max_buy_sell_ratio_1h == 8.0
        else:
            assert thr.max_buy_sell_ratio_1h is None


def test_new_filters_registered_in_defaults() -> None:
    eng = FilterEngine()
    assert FilterName.DEGEN_LAUNCH in eng.thresholds
    assert FilterName.VOLATILITY_BREAKOUT in eng.thresholds


# ── Strategies ────────────────────────────────────────────────────────


def _active(cls):
    strat = cls(BotConfig())
    strat.state.active = True
    return strat


def test_degen_ignition_signals_on_detonation() -> None:
    sig = _active(DegenIgnitionStrategy).evaluate_token({"token_address": "DEGEN"}, _degen())
    assert sig is not None
    assert 0.0 < sig.ignition_score <= 1.0


def test_degen_ignition_silent_on_old_token() -> None:
    s = _degen()
    s.age_minutes = 300
    assert _active(DegenIgnitionStrategy).evaluate_token({"token_address": "X"}, s) is None


def test_volatility_breakout_signals_on_vertical_move() -> None:
    sig = _active(VolatilityBreakoutStrategy).evaluate_token({"token_address": "BRK"}, _breakout())
    assert sig is not None
    assert 0.0 < sig.breakout_score <= 1.0


def test_volatility_breakout_silent_when_flat() -> None:
    s = _breakout()
    s.price_change_1h_pct = 10.0
    assert _active(VolatilityBreakoutStrategy).evaluate_token({"token_address": "X"}, s) is None


def test_playbook_tagger_includes_risk_on_strategies() -> None:
    tagger = PlaybookTagger()
    assert "degen_ignition" in tagger.strategy_ids
    assert "volatility_breakout" in tagger.strategy_ids
    tags = tagger.tag(_degen())
    assert "degen_ignition" in tags.playbook_ids
    tags2 = tagger.tag(_breakout())
    assert "volatility_breakout" in tags2.playbook_ids
