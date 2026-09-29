#!/usr/bin/env python3
"""Tests for fenrir.discovery.playbooks — read-only strategy playbook tagging."""

from typing import Any

import pytest

from fenrir.discovery.models import Chain, TokenSnapshot
from fenrir.discovery.playbooks import PLAYBOOK_STRATEGY_IDS, PlaybookTagger


def _snap(**kw: Any) -> TokenSnapshot:
    base: dict[str, Any] = dict(
        chain=Chain.SOLANA,
        token_address="TEST",
        symbol="TST",
        age_minutes=120,
        market_cap_usd=800_000,
        price_usd=0.001,
        liquidity_usd=120_000,
        volume_5m_usd=30_000,
        volume_1h_usd=200_000,
        volume_24h_usd=1_500_000,
        price_change_5m_pct=3.0,
        price_change_1h_pct=18.0,
        price_change_24h_pct=60.0,
        txns_5m_buys=400,
        txns_5m_sells=150,
        txns_1h_buys=3000,
        txns_1h_sells=1200,
    )
    base.update(kw)
    return TokenSnapshot(**base)


def _dead():
    return _snap(
        token_address="DEAD",
        symbol="DEAD",
        age_minutes=60,
        market_cap_usd=50_000,
        price_usd=0.00001,
        liquidity_usd=2_000,
        volume_5m_usd=10,
        volume_1h_usd=100,
        volume_24h_usd=500,
        price_change_5m_pct=-2.0,
        price_change_1h_pct=-40.0,
        price_change_24h_pct=-70.0,
        txns_5m_buys=1,
        txns_5m_sells=5,
        txns_1h_buys=10,
        txns_1h_sells=40,
    )


def test_tagger_loads_all_six_strategies():
    tagger = PlaybookTagger()
    assert tagger.strategy_ids == list(PLAYBOOK_STRATEGY_IDS)
    # Read-only mode: active for evaluation, but never trades.
    for _, strat in tagger._strategies:
        assert strat.state.active is True


def test_momentum_shaped_token_gets_momentum_tag():
    tags = PlaybookTagger().tag(_snap())
    assert "momentum" in tags.playbook_ids
    m = next(m for m in tags.matches if m.strategy_id == "momentum")
    assert 0.0 < m.strength <= 1.0
    assert m.display_name  # human-readable label present


def test_volume_anomaly_shaped_token_gets_tag():
    snap = _snap(
        token_address="VOL",
        age_minutes=400,
        market_cap_usd=1_000_000,
        volume_24h_usd=2_000_000,
        price_change_5m_pct=-1.5,
        price_change_1h_pct=5.0,
        price_change_24h_pct=25.0,
    )
    tags = PlaybookTagger().tag(snap)
    assert "volume_anomaly" in tags.playbook_ids


def test_dead_token_gets_no_tags():
    tags = PlaybookTagger().tag(_dead())
    assert tags.matches == []
    assert tags.confluent is False
    assert tags.combined_strength == 0.0


def test_two_agreeing_strategies_are_confluent():
    # Crashed -20% in 1h but stabilizing, huge volume vs mcap:
    # fits both mean_reversion (oversold bounce) and volume_anomaly (dip scalp).
    snap = _snap(
        token_address="BOTH",
        age_minutes=400,
        market_cap_usd=1_000_000,
        volume_5m_usd=40_000,
        volume_1h_usd=300_000,
        volume_24h_usd=2_000_000,
        price_change_5m_pct=-1.5,
        price_change_1h_pct=-20.0,
        price_change_24h_pct=-30.0,
        txns_5m_buys=300,
        txns_5m_sells=250,
        txns_1h_buys=2500,
        txns_1h_sells=2200,
    )
    tags = PlaybookTagger().tag(snap)
    assert tags.confluent is True
    assert len(tags.sources) >= 2
    assert 0.0 < tags.combined_strength <= 1.0


def test_one_failing_strategy_does_not_kill_tagging():
    tagger = PlaybookTagger()
    sid, strat = tagger._strategies[0]

    def boom(token_data, market_data):
        raise RuntimeError("strategy exploded")

    strat.evaluate_token = boom  # type: ignore[method-assign,assignment]
    tags = tagger.tag(_snap())  # must not raise
    assert sid not in tags.playbook_ids
    # Other strategies still evaluated.
    assert isinstance(tags.matches, list)


def test_as_dict_shape():
    d = PlaybookTagger().tag(_snap()).as_dict()
    assert set(d) == {"playbooks", "confluent", "combined_strength"}
    assert isinstance(d["playbooks"], list)
    for pb in d["playbooks"]:
        assert set(pb) == {"strategy_id", "display_name", "strength", "rationale"}
    # Strongest conviction first.
    strengths = [pb["strength"] for pb in d["playbooks"]]
    assert strengths == sorted(strengths, reverse=True)


def test_buy_pressure_5m_property():
    s = TokenSnapshot(chain=Chain.SOLANA, token_address="X")
    assert s.buy_pressure_5m == 0.5  # neutral when no data
    s2 = TokenSnapshot(chain=Chain.SOLANA, token_address="Y", txns_5m_buys=300, txns_5m_sells=100)
    assert s2.buy_pressure_5m == pytest.approx(0.75)
