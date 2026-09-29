"""Tests for the risk-on discovery tier: degen_launch / volatility_breakout
filters plus the degen_ignition / volatility_breakout playbook strategies."""
from __future__ import annotations

import sys
import os

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
