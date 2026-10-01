"""Tests for tools/gate_tracker.py — the scout's gate-clearance feedback loop."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gate_tracker",
    os.path.join(os.path.dirname(__file__), "..", "tools", "gate_tracker.py"),
)
assert SPEC is not None and SPEC.loader is not None
gt = importlib.util.module_from_spec(SPEC)
sys.modules["gate_tracker"] = gt
SPEC.loader.exec_module(gt)


def _cand(addr="0xabc", **kw):
    c = {
        "address": addr,
        "symbol": "FOO",
        "name": "Foo Token",
        "chain": "robinhood",
        "price_usd": 0.001,
        "market_cap_usd": 10_000.0,
        "liquidity_usd": 8_000.0,
        "passed_filters": ["degen_launch"],
        "filter_warnings": [],
        "playbooks": {"tags": {"Degen Ignition": 0.99}},
        "score": {"overall": 72.5},
        "source": "gecko_new",
        "dexscreener": "https://dexscreener.com/robinhood/0xabc",
        "safety_unknown": False,
    }
    c.update(kw)
    return c


def test_pct_change():
    assert gt.pct_change(0.002, 0.001) == pytest.approx(100.0)
    assert gt.pct_change(0.0005, 0.001) == pytest.approx(-50.0)
    assert gt.pct_change(None, 0.001) is None
    assert gt.pct_change(0.001, 0) is None
    assert gt.pct_change(0.001, None) is None


def test_record_first_clearance_wins():
    state: dict = {}
    rec = gt.record_candidates([_cand()], state, ts=1000.0)
    assert rec == ["0xabc"]
    assert state["0xabc"]["cleared_at"] == 1000.0
    assert state["0xabc"]["clearance_price"] == 0.001
    assert state["0xabc"]["filters"] == ["degen_launch"]
    assert state["0xabc"]["playbooks"] == {"Degen Ignition": 0.99}
    assert state["0xabc"]["score"] == 72.5
    # re-recording the same address never re-stamps
    rec2 = gt.record_candidates([_cand(price_usd=0.002)], state, ts=2000.0)
    assert rec2 == []
    assert state["0xabc"]["cleared_at"] == 1000.0
    assert state["0xabc"]["clearance_price"] == 0.001


def test_record_missing_price_fetches(monkeypatch):
    monkeypatch.setattr(gt, "fetch_price", lambda addr, timeout=20: (0.005, 50_000.0))
    state: dict = {}
    gt.record_candidates([_cand(price_usd=None, market_cap_usd=None)], state, ts=1.0)
    assert state["0xabc"]["clearance_price"] == 0.005
    assert state["0xabc"]["clearance_mcap"] == 50_000.0


def test_tick_appends_and_detects_movers(monkeypatch):
    state: dict = {}
    gt.record_candidates([_cand()], state, ts=1000.0)
    # +150% -> pump_100 fires once
    monkeypatch.setattr(gt, "fetch_price", lambda addr, timeout=20: (0.0025, 25_000.0))
    out = gt.tick_state(state, ts=2000.0)
    assert out["ticked"] == 1
    assert out["movers"] == [
        {"address": "0xabc", "symbol": "FOO", "kind": "pump_100", "chg_pct": 150.0}
    ]
    assert state["0xabc"]["alerted_moves"] == ["pump_100"]
    assert state["0xabc"]["ticks"][-1]["chg_pct"] == pytest.approx(150.0)
    # second tick at same level -> no duplicate mover
    out2 = gt.tick_state(state, ts=3000.0)
    assert out2["movers"] == []


def test_tick_detects_dump_and_dead(monkeypatch):
    state: dict = {}
    gt.record_candidates([_cand(addr="0xdown"), _cand(addr="0xdead")], state, ts=1.0)

    def fake(addr, timeout=20):
        if addr == "0xdown":
            return (0.0004, 4_000.0)  # -60%
        return (None, None)  # no pair data

    monkeypatch.setattr(gt, "fetch_price", fake)
    out = gt.tick_state(state, ts=2.0)
    assert out["movers"] == [
        {"address": "0xdown", "symbol": "FOO", "kind": "dump_50", "chg_pct": -60.0}
    ]
    dead = gt.summarize(state["0xdead"], now=3.0)
    assert dead["dead"] is True
    assert dead["chg_pct"] is None


def test_report_sorts_winners_first():
    state: dict = {}
    gt.record_candidates([_cand(addr="0xup"), _cand(addr="0xdn")], state, ts=1.0)
    state["0xup"]["ticks"] = [{"ts": 2, "price": 0.002, "mcap": 1, "chg_pct": 100.0}]
    state["0xdn"]["ticks"] = [{"ts": 2, "price": 0.0005, "mcap": 1, "chg_pct": -50.0}]
    rows = [gt.summarize(r, now=3.0) for r in state.values()]
    rows.sort(key=lambda r: (r["chg_pct"] is None, -(r["chg_pct"] or 0)))
    assert [r["symbol"] for r in rows] == ["FOO", "FOO"]
    assert rows[0]["chg_pct"] == pytest.approx(100.0)
    assert rows[1]["chg_pct"] == pytest.approx(-50.0)
    assert rows[0]["peak_pct"] == pytest.approx(100.0)
    assert rows[1]["trough_pct"] == pytest.approx(-50.0)


def test_state_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tracked.json")
        state: dict = {}
        gt.record_candidates([_cand()], state, ts=1.0)
        gt.save_state(p, state)
        loaded = gt.load_state(p)
        assert loaded["0xabc"]["symbol"] == "FOO"
        assert gt.load_state(os.path.join(d, "missing.json")) == {}


def test_record_cli_end_to_end(tmp_path):
    cands_file = tmp_path / "cands.json"
    cands_file.write_text(json.dumps({"candidates": [_cand(addr="0xcli")]}))
    state_file = str(tmp_path / "tracked.json")
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    proc = subprocess.run(
        [
            sys.executable,
            "tools/gate_tracker.py",
            "--state",
            state_file,
            "record",
            "--candidates",
            str(cands_file),
            "--ts",
            "1234.0",
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.returncode == 0
    state = json.loads(open(state_file).read())
    assert state["0xcli"]["cleared_at"] == 1234.0
