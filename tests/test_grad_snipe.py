"""Tests for the graduation snipe watcher (tools/grad_snipe.py).

Covers graduation detection, venue pair picking, the snipe gates, and the
alert format. Network I/O is not exercised here.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.discovery.models import SafetySignals  # noqa: E402
from tools.grad_snipe import (  # noqa: E402
    _load_json,
    _save_json,
    format_snipe_alert,
    graduation_inflow_sol,
    is_graduated,
    pick_pair,
    snipe_check,
)


def _safety(**kw) -> SafetySignals:
    base: dict[str, Any] = dict(
        honeypot=False, mint_disabled=True, sell_tax_pct=0.0, buy_tax_pct=0.0
    )
    base.update(kw)
    return SafetySignals(**base)


def _feat(**kw) -> dict:
    base: dict[str, Any] = dict(
        m5_buys=40,
        m5_sells=10,
        h1_buys=200,
        h1_sells=90,
        price_change_m5_pct=8.0,
        liquidity_usd=50_000,
        market_cap_usd=800_000,
    )
    base.update(kw)
    return base


# ── Graduation detection ─────────────────────────────────────────────


def test_graduated_when_curve_account_closed():
    prev = {"last_progress": 96.2, "complete": False}
    assert is_graduated(prev, None) is True


def test_graduated_when_state_flips_complete():
    prev = {"last_progress": 99.1, "complete": False}
    assert is_graduated(prev, types.SimpleNamespace(complete=True)) is True


def test_not_graduated_when_already_complete():
    prev = {"last_progress": 100.0, "complete": True}
    assert is_graduated(prev, None) is False


def test_not_graduated_when_still_bonding():
    prev = {"last_progress": 82.0, "complete": False}
    assert is_graduated(prev, types.SimpleNamespace(complete=False)) is False


def test_graduation_inflow():
    entry = {"last_sol": 84.2, "prev_sol": 71.8}
    assert graduation_inflow_sol(entry) == 12.4


def test_graduation_inflow_unknown_without_prev():
    assert graduation_inflow_sol({"last_sol": 84.2}) is None


# ── Pair picking ─────────────────────────────────────────────────────


def _pair(dex, liq):
    return {"dexId": dex, "liquidity": {"usd": liq}}


def test_pick_pair_prefers_graduation_venue():
    pairs = [_pair("unknown_amm", 500_000), _pair("raydium", 60_000)]
    result = pick_pair(pairs)
    assert result is not None
    assert result["dexId"] == "raydium"


def test_pick_pair_falls_back_to_deepest():
    pairs = [_pair("foo", 10_000), _pair("bar", 90_000)]
    result = pick_pair(pairs)
    assert result is not None
    assert result["dexId"] == "bar"


def test_pick_pair_empty():
    assert pick_pair([]) is None


# ── Snipe gates ──────────────────────────────────────────────────────


def test_snipe_passes_on_hot_early_tape():
    ok, fails = snipe_check(_feat(), _safety())
    assert ok, fails


def test_snipe_passes_on_hourly_edge_when_5m_sample_small():
    feat = _feat(m5_buys=3, m5_sells=2, h1_buys=120, h1_sells=60)
    ok, fails = snipe_check(feat, _safety())
    assert ok, fails


def test_snipe_rejects_sell_dominated_tape():
    feat = _feat(m5_buys=10, m5_sells=40, h1_buys=40, h1_sells=120)
    ok, fails = snipe_check(feat, _safety())
    assert not ok and any("buy edge" in f for f in fails)


def test_snipe_rejects_dumping_out_of_gate():
    ok, fails = snipe_check(_feat(price_change_m5_pct=-12.0), _safety())
    assert not ok and any("dumping" in f for f in fails)


def test_snipe_rejects_thin_liquidity():
    ok, fails = snipe_check(_feat(liquidity_usd=5_000), _safety())
    assert not ok and any("LP" in f for f in fails)


def test_snipe_rejects_already_mooned():
    ok, fails = snipe_check(_feat(market_cap_usd=8_000_000), _safety())
    assert not ok and any("mooned" in f for f in fails)


def test_snipe_rejects_honeypot():
    ok, fails = snipe_check(_feat(), _safety(honeypot=True))
    assert not ok and any("honeypot" in f for f in fails)


def test_snipe_rejects_live_mint_authority():
    ok, fails = snipe_check(_feat(), _safety(mint_disabled=False))
    assert not ok and any("mint authority" in f for f in fails)


def test_snipe_rejects_high_sell_tax():
    ok, fails = snipe_check(_feat(), _safety(sell_tax_pct=25.0))
    assert not ok and any("sell tax" in f for f in fails)


def test_snipe_skips_safety_when_unknown():
    ok, fails = snipe_check(_feat(), None)
    assert ok, fails


# ── Alert format + state ─────────────────────────────────────────────


def test_snipe_alert_format():
    feat = dict(_feat(), symbol="TEST", name="Test Token", url="https://dexscreener.com/solana/abc")
    msg = format_snipe_alert("MINT123", feat, 12.4, 90)
    assert "`MINT123`" in msg  # tap-to-copy
    assert "TEST" in msg
    assert "graduated" in msg
    assert "12.4 SOL" in msg
    assert "[DexScreener](https://dexscreener.com/solana/abc)" in msg


def test_snipe_state_round_trip(tmp_path):
    path = str(tmp_path / "grad_snipe.json")
    _save_json({"MINT": {"alerted": True}}, path)
    assert _load_json(path) == {"MINT": {"alerted": True}}
    assert _load_json(str(tmp_path / "missing.json")) == {}
