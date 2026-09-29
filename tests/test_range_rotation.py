"""Tests for the range_rotation strategy: the community-coin rebalancing play.

A settled, liquid, distributed coin (12h+, $100k+ LP, top holder <= 15%)
sitting in a deep 24h dip (-20%..-60%) that has stabilized on the short
windows with buyers stepping back in — the "rotate into the laggard at
range bottom" setup. Off by default, tagging only.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.config import BotConfig  # noqa: E402
from fenrir.discovery.models import Chain, TokenSnapshot  # noqa: E402
from fenrir.discovery.playbooks import PLAYBOOK_STRATEGY_IDS, PlaybookTagger  # noqa: E402
from fenrir.strategies import (  # noqa: E402
    get_strategy_class,
    is_enabled_by_default,
)
from fenrir.strategies.range_rotation import RangeRotationStrategy  # noqa: E402

TOKEN = {"token_address": "RANGE000000000000000000000000000000000000000"}


def _snap(**over) -> TokenSnapshot:
    """A TokenSnapshot that PASSES every range-rotation gate by default."""
    base: dict[str, Any] = dict(
        chain=Chain.SOLANA,
        token_address=TOKEN["token_address"],
        symbol="SETTLED",
        market_cap_usd=12_000_000.0,
        liquidity_usd=250_000.0,
        volume_1h_usd=80_000.0,
        age_minutes=3 * 24 * 60.0,  # 3 days — settled
        holder_count=9_000,
        txns_1h_buys=140,
        txns_1h_sells=100,  # 1h b/s = 1.4
        txns_5m_buys=60,
        txns_5m_sells=40,  # 5m pressure 0.6
        price_change_5m_pct=0.5,
        price_change_1h_pct=-3.0,
        price_change_24h_pct=-35.0,  # deep dip, not dead
        top_holder_pct=8.0,
    )
    base.update(over)
    return TokenSnapshot(**base)


def _eval(snap):
    return RangeRotationStrategy(BotConfig()).evaluate_token(TOKEN, snap)


# ── Registry ───────────────────────────────────────────────────────────


def test_registered():
    assert get_strategy_class("range_rotation") is RangeRotationStrategy
    assert "range_rotation" in PLAYBOOK_STRATEGY_IDS
    assert not is_enabled_by_default("range_rotation")  # opt-in, tagging only


# ── Entry gates ────────────────────────────────────────────────────────


def test_passes_settled_range_bottom():
    sig = _eval(_snap())
    assert sig is not None
    assert sig.token_address == TOKEN["token_address"]
    assert 0.0 < sig.rotation_score <= 1.0
    assert sig.metadata["strategy"] == "range_rotation"


def test_rejects_fresh_launch():
    assert _eval(_snap(age_minutes=600.0)) is None  # under the 12h floor


def test_rejects_shallow_dip():
    assert _eval(_snap(price_change_24h_pct=-15.0)) is None


def test_rejects_dead_coin():
    assert _eval(_snap(price_change_24h_pct=-70.0)) is None


def test_rejects_still_dumping_1h():
    assert _eval(_snap(price_change_1h_pct=-20.0)) is None


def test_rejects_no_5m_stabilization():
    assert _eval(_snap(price_change_5m_pct=-3.0)) is None


def test_rejects_no_buy_edge():
    # b/s below 1.1 AND 5m pressure below 0.55 → nobody stepping in.
    assert (
        _eval(_snap(txns_1h_buys=90, txns_1h_sells=100, txns_5m_buys=50, txns_5m_sells=50)) is None
    )


def test_passes_on_5m_pressure_alone():
    # 1h edge absent but 5m buyers returning counts as stepping back in.
    assert (
        _eval(_snap(txns_1h_buys=100, txns_1h_sells=100, txns_5m_buys=60, txns_5m_sells=40))
        is not None
    )


def test_rejects_thin_liquidity():
    assert _eval(_snap(liquidity_usd=50_000.0)) is None


def test_rejects_dead_tape():
    assert _eval(_snap(volume_1h_usd=10_000.0)) is None


def test_rejects_concentrated_supply():
    assert _eval(_snap(top_holder_pct=25.0)) is None


def test_passes_when_holder_data_absent():
    # DexScreener-only snapshots may lack holder info — the distribution gate
    # only bites when the data exists.
    assert _eval(_snap(top_holder_pct=None)) is not None


def test_requires_market_data():
    assert RangeRotationStrategy(BotConfig()).evaluate_token(TOKEN, None) is None


# ── Scoring ────────────────────────────────────────────────────────────


def test_deeper_dip_scores_higher():
    shallow = _eval(_snap(price_change_24h_pct=-25.0))
    deep = _eval(_snap(price_change_24h_pct=-55.0))
    assert shallow is not None and deep is not None
    assert deep.rotation_score > shallow.rotation_score


def test_ai_context_mentions_exit_plan():
    sig = _eval(_snap())
    assert sig is not None
    ctx = RangeRotationStrategy(BotConfig()).build_ai_context(sig)
    assert "+150%" in ctx and "-30%" in ctx


# ── Playbook tagging ───────────────────────────────────────────────────


def test_playbook_tagger_fires():
    tagger = PlaybookTagger()
    assert "range_rotation" in tagger.strategy_ids
    tags = tagger.tag(_snap())
    assert "range_rotation" in tags.playbook_ids


def test_playbook_tagger_silent_on_fresh_launch():
    tags = PlaybookTagger().tag(_snap(age_minutes=30.0))
    assert "range_rotation" not in tags.playbook_ids
