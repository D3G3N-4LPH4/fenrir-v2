"""Tests for the volume_surge strategy: the higher-cap volume trade.

Covers the volume_surge entry filter (fenrir/discovery/filters.py) and the
VolumeSurgeStrategy playbook (fenrir/strategies/volume_surge.py), calibrated
against a PARASITE-like snapshot ($3.7M mcap, $3M 24h volume, +49% 1h).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.config import BotConfig  # noqa: E402
from fenrir.discovery.filters import FilterEngine, FilterName  # noqa: E402
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot  # noqa: E402
from fenrir.discovery.playbooks import PLAYBOOK_STRATEGY_IDS, PlaybookTagger  # noqa: E402
from fenrir.strategies import (  # noqa: E402
    get_strategy_class,
    is_enabled_by_default,
)
from fenrir.strategies.volume_surge import VolumeSurgeStrategy  # noqa: E402


def _safe() -> SafetySignals:
    return SafetySignals(
        mint_disabled=True,
        freeze_disabled=True,
        lp_locked_or_burned=True,
        contract_verified=True,
        blacklist_present=False,
        honeypot=False,
    )


def _parasite() -> TokenSnapshot:
    """PARASITE-like: $3.7M mcap, $3.03M 24h vol (0.82x turnover), +49% 1h."""
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address="3kmygWKZBkCYrgZHKfiuB9UFKTcDLTFFsKo3BWpmpump",
        symbol="PARASITE",
        market_cap_usd=3_700_000,
        liquidity_usd=292_000,
        volume_24h_usd=3_030_000,
        volume_1h_usd=298_000,
        age_minutes=3 * 24 * 60,
        holder_count=8_500,
        txns_24h_buys=11_422,
        txns_24h_sells=10_245,
        txns_1h_buys=898,
        txns_1h_sells=853,
        price_change_1h_pct=48.7,
        price_change_24h_pct=328.0,
        top_holder_pct=6.0,
        top10_holder_pct=35.0,
        dev_wallet_pct=3.0,
        safety=_safe(),
    )


# ── Filter ────────────────────────────────────────────────────────────


def test_volume_surge_filter_passes_parasite_like():
    engine = FilterEngine()
    res = engine.evaluate(_parasite(), FilterName.VOLUME_SURGE)
    assert res.passed, f"expected pass, failures={res.failures}"


def test_volume_surge_filter_rejects_below_cap_band():
    engine = FilterEngine()
    snap = _parasite()
    snap.market_cap_usd = 1_200_000  # under the $2M floor
    res = engine.evaluate(snap, FilterName.VOLUME_SURGE)
    assert not res.passed
    assert any("MCap" in f for f in res.failures)


def test_volume_surge_filter_rejects_quiet_tape():
    engine = FilterEngine()
    snap = _parasite()
    snap.market_cap_usd = 5_000_000
    snap.volume_24h_usd = 2_200_000  # 0.44x turnover < 0.5x
    res = engine.evaluate(snap, FilterName.VOLUME_SURGE)
    assert not res.passed
    assert any("urnover" in f for f in res.failures)


def test_volume_surge_filter_rejects_wash_turnover():
    engine = FilterEngine()
    snap = _parasite()
    snap.market_cap_usd = 2_000_000
    snap.volume_24h_usd = 50_000_000  # 25x turnover > 20x cap
    res = engine.evaluate(snap, FilterName.VOLUME_SURGE)
    assert not res.passed
    assert any("urnover" in f for f in res.failures)


def test_volume_surge_filter_rejects_sell_dominated_flow():
    engine = FilterEngine()
    snap = _parasite()
    snap.txns_1h_buys = 400
    snap.txns_1h_sells = 900  # ratio 0.44 < 1.0
    res = engine.evaluate(snap, FilterName.VOLUME_SURGE)
    assert not res.passed
    assert any("buy/sell" in f for f in res.failures)


def test_volume_surge_filter_rejects_terminal_wick():
    engine = FilterEngine()
    snap = _parasite()
    snap.price_change_1h_pct = 120.0  # > +80% guard
    res = engine.evaluate(snap, FilterName.VOLUME_SURGE)
    assert not res.passed
    assert any("1h" in f for f in res.failures)


def test_volume_surge_filter_allows_exchange_sized_top_holder():
    # 14% top holder passes: at $2M+ the #1 wallet is often a CEX omnibus.
    engine = FilterEngine()
    snap = _parasite()
    snap.top_holder_pct = 14.0
    res = engine.evaluate(snap, FilterName.VOLUME_SURGE)
    assert res.passed, f"expected pass, failures={res.failures}"


def test_volume_surge_filter_rejects_concentrated_top_holder():
    engine = FilterEngine()
    snap = _parasite()
    snap.top_holder_pct = 25.0  # > 15% cap
    res = engine.evaluate(snap, FilterName.VOLUME_SURGE)
    assert not res.passed
    assert any("Top holder" in f for f in res.failures)


def test_volume_surge_filter_rejects_fresh_launch():
    engine = FilterEngine()
    snap = _parasite()
    snap.age_minutes = 30  # < 12h minimum
    res = engine.evaluate(snap, FilterName.VOLUME_SURGE)
    assert not res.passed
    assert any("Age" in f for f in res.failures)


# ── Strategy ──────────────────────────────────────────────────────────


def _strategy() -> VolumeSurgeStrategy:
    strat = VolumeSurgeStrategy(BotConfig())
    strat.state.active = True  # read-only evaluation mode, like the tagger
    return strat


def test_volume_surge_strategy_signals_parasite_like():
    sig = _strategy().evaluate_token({"token_address": _parasite().token_address}, _parasite())
    assert sig is not None
    assert sig.turnover_24h == 3_030_000 / 3_700_000
    assert 0.0 < sig.surge_score <= 1.0


def test_volume_surge_strategy_rejects_sell_dominated():
    snap = _parasite()
    snap.txns_1h_buys = 400
    snap.txns_1h_sells = 900
    sig = _strategy().evaluate_token({"token_address": snap.token_address}, snap)
    assert sig is None


def test_volume_surge_strategy_rejects_quiet_tape():
    snap = _parasite()
    snap.market_cap_usd = 5_000_000
    snap.volume_24h_usd = 2_200_000
    sig = _strategy().evaluate_token({"token_address": snap.token_address}, snap)
    assert sig is None


# ── Registration ──────────────────────────────────────────────────────


def test_volume_surge_registered_and_default_off():
    assert get_strategy_class("volume_surge") is VolumeSurgeStrategy
    assert "volume_surge" in PLAYBOOK_STRATEGY_IDS
    assert not is_enabled_by_default("volume_surge")  # opt-in, like its siblings


def test_volume_surge_playbook_tags_parasite_like():
    tagger = PlaybookTagger()
    assert "volume_surge" in tagger.strategy_ids
    tags = tagger.tag(_parasite())
    assert "volume_surge" in tags.playbook_ids
