"""Tests for tools/misfit_watch.py — the gate-rejected cohort tracker."""

import importlib.util
import os
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "misfit_watch",
    os.path.join(os.path.dirname(__file__), "..", "tools", "misfit_watch.py"),
)
assert SPEC is not None and SPEC.loader is not None
mw = importlib.util.module_from_spec(SPEC)
sys.modules["misfit_watch"] = mw
SPEC.loader.exec_module(mw)


def _misfit(addr="0xabc", **kw):
    c = {
        "address": addr,
        "symbol": "FOO",
        "name": "Foo Token",
        "chain": "solana",
        "price_usd": 0.001,
        "market_cap_usd": 10_000.0,
        "liquidity_usd": 8_000.0,
        "passed_filters": [],
        "score": {"overall": 74.0},
        "source": "gecko_new",
        "dexscreener": "https://dexscreener.com/solana/0xabc",
        "safety_unknown": False,
        "misfit": True,
    }
    c.update(kw)
    return c


def test_record_first_sighting_wins():
    state: dict = {}
    rec = mw.record_misfits([_misfit()], state, ts=1000.0)
    assert rec == ["0xabc"]
    assert state["0xabc"]["first_seen"] == 1000.0
    assert state["0xabc"]["first_price"] == 0.001
    assert state["0xabc"]["score"] == 74.0
    # re-recording the same address never re-stamps
    rec2 = mw.record_misfits([_misfit(price_usd=0.002)], state, ts=2000.0)
    assert rec2 == []
    assert state["0xabc"]["first_price"] == 0.001


def test_record_missing_price_fetches(monkeypatch):
    monkeypatch.setattr(mw, "fetch_price", lambda addr: (0.005, 50_000.0))
    state: dict = {}
    mw.record_misfits([_misfit(price_usd=None, market_cap_usd=None)], state, ts=1000.0)
    assert state["0xabc"]["first_price"] == 0.005
    assert state["0xabc"]["first_mcap"] == 50_000.0


def test_tick_appends_and_detects_pump(monkeypatch):
    state: dict = {}
    mw.record_misfits([_misfit()], state, ts=1000.0)
    monkeypatch.setattr(mw, "fetch_price", lambda addr: (0.0025, 25_000.0))  # +150%
    out = mw.tick_state(state, ts=2000.0)
    assert out["ticked"] == 1
    assert len(out["movers"]) == 1
    assert out["movers"][0]["kind"] == "pump_100"
    assert out["movers"][0]["chg_pct"] == pytest.approx(150.0)
    # one-time key: second tick does not re-flag
    out2 = mw.tick_state(state, ts=3000.0)
    assert out2["movers"] == []


def test_tick_detects_dump(monkeypatch):
    state: dict = {}
    mw.record_misfits([_misfit()], state, ts=1000.0)
    monkeypatch.setattr(mw, "fetch_price", lambda addr: (0.0004, 4_000.0))  # -60%
    out = mw.tick_state(state, ts=2000.0)
    assert out["movers"][0]["kind"] == "dump_50"
    assert out["movers"][0]["chg_pct"] == pytest.approx(-60.0)


def test_tick_no_pair_data_is_not_a_mover(monkeypatch):
    state: dict = {}
    mw.record_misfits([_misfit()], state, ts=1000.0)
    monkeypatch.setattr(mw, "fetch_price", lambda addr: (None, None))
    out = mw.tick_state(state, ts=2000.0)
    assert out["movers"] == []
    assert state["0xabc"]["ticks"][-1]["note"] == "no pair data"


def test_format_mover():
    text = mw.format_mover(
        {
            "symbol": "TERMINAL",
            "kind": "pump_100",
            "chg_pct": 1055.0,
            "dexscreener": "https://dexscreener.com/solana/xyz",
        }
    )
    assert "TERMINAL" in text and "+1055.0%" in text
    assert "https://dexscreener.com/solana/xyz" in text


def test_load_candidates_accepts_scout_output(tmp_path):
    p = tmp_path / "scout.json"
    p.write_text('{"misfits": [{"address": "0x1"}], "candidates": []}')
    assert mw._load_candidates(str(p)) == [{"address": "0x1"}]
    p2 = tmp_path / "list.json"
    p2.write_text('[{"address": "0x2"}]')
    assert mw._load_candidates(str(p2)) == [{"address": "0x2"}]


def test_summarize_move_stats():
    state: dict = {}
    mw.record_misfits([_misfit()], state, ts=1000.0)
    rec = state["0xabc"]
    rec["ticks"] = [
        {"ts": 2000.0, "price": 0.002, "mcap": 20_000.0, "chg_pct": 100.0},
        {"ts": 3000.0, "price": 0.0015, "mcap": 15_000.0, "chg_pct": 50.0},
    ]
    s = mw.summarize(rec, now=4000.0)
    assert s["chg_pct"] == pytest.approx(50.0)
    assert s["peak_pct"] == pytest.approx(100.0)
    assert s["trough_pct"] == pytest.approx(50.0)


def test_record_normalizes_evm_address_case():
    # 2026-10-04: a checksummed 0x address and its lowercase form double-
    # tracked CHFUND and double-fired a Telegram flag. Never again.
    state: dict = {}
    mw.record_misfits(
        [_misfit(addr="0x0821df61b58195D21C0C969212C89a2CFFDdb1b2")], state, ts=1000.0
    )
    rec = mw.record_misfits(
        [_misfit(addr="0x0821df61b58195d21c0c969212c89a2cffddb1b2")], state, ts=2000.0
    )
    assert rec == []
    assert len(state) == 1
    assert "0x0821df61b58195d21c0c969212c89a2cffddb1b2" in state


def test_record_keeps_solana_case_sensitive():
    state: dict = {}
    mw.record_misfits(
        [_misfit(addr="9ZmkKpVR3NdUcCmG5NByMHBzvjuqGnYpCPYu4zSSBtJv")], state, ts=1000.0
    )
    assert "9ZmkKpVR3NdUcCmG5NByMHBzvjuqGnYpCPYu4zSSBtJv" in state
