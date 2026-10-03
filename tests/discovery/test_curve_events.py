"""Tests for the event-driven curve ignition lane."""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import time

import base58

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fenrir.discovery.providers.pumpfun_events import (  # noqa: E402
    extract_create_candidates,
    notification_has_create,
)
from fenrir.protocol.pumpfun import (  # noqa: E402
    CREATE_DISCRIMINATORS,
    TokenLaunchDetector,
)
from tools.curve_events import (  # noqa: E402
    ignition_check,
    implied_mcap_sol,
    should_prune,
)


def _make_create_ix_data(name="Test", symbol="TST", uri="https://x") -> bytes:
    disc = hashlib.sha256(b"global:create").digest()[:8]
    assert disc in CREATE_DISCRIMINATORS
    out = bytearray(disc)
    for s in (name, symbol, uri):
        b = s.encode()
        out += struct.pack("<I", len(b)) + b
    return bytes(out)


def test_is_create_instruction():
    d = TokenLaunchDetector()
    assert d.is_create_instruction(_make_create_ix_data())
    assert not d.is_create_instruction(b"\x00" * 8)
    assert not d.is_create_instruction(b"short")


def test_parse_create_event():
    d = TokenLaunchDetector()
    data = _make_create_ix_data("MyCoin", "MC", "https://example.com/m.json")
    accounts = [f"acct{i}" for i in range(10)]
    parsed = d.parse_create_event(data, accounts)
    assert parsed is not None
    assert parsed["token_mint"] == "acct0"
    assert parsed["bonding_curve"] == "acct2"
    assert parsed["creator"] == "acct7"
    assert parsed["name"] == "MyCoin"
    assert parsed["symbol"] == "MC"
    assert parsed["uri"] == "https://example.com/m.json"


def test_parse_create_event_bad_data():
    d = TokenLaunchDetector()
    assert d.parse_create_event(b"\x00" * 8, []) is None


def test_notification_has_create():
    assert notification_has_create(
        ["Program log: Instruction: Create", "Program consumed 1234 units"]
    )
    assert notification_has_create(["program log: instruction: createv2"])
    assert not notification_has_create(["Program log: Instruction: Buy"])
    assert not notification_has_create([])


def _notif(logs, sig="sig123", slot=42):
    return {
        "jsonrpc": "2.0",
        "method": "logsNotification",
        "params": {
            "result": {
                "context": {"slot": slot},
                "value": {"signature": sig, "logs": logs, "err": None},
            },
            "subscription": 1,
        },
    }


def test_extract_create_candidates():
    cands = extract_create_candidates(_notif(["Program log: Instruction: Create"]))
    assert cands == [("sig123", 42, ["Program log: Instruction: Create"])]
    assert extract_create_candidates(_notif(["Program log: Instruction: Buy"])) == []
    assert extract_create_candidates({"id": 1, "result": True}) == []  # subscribe ack
    assert extract_create_candidates({}) == []
    assert extract_create_candidates(_notif([], sig="")) == []


def _entry(**kw):
    base = {
        "first_seen": time.time() - 300,
        "last_check": time.time(),
        "last_progress": 25.0,
        "last_sol": 21.0,
        "inflow_sol": 2.5,
        "prev_inflow_sol": 1.0,
        "alerted": False,
    }
    base.update(kw)
    return base


def test_ignition_check_fires():
    fire, reasons = ignition_check(_entry(), time.time())
    assert fire, reasons


def test_ignition_check_gates():
    now = time.time()
    fire, _ = ignition_check(_entry(last_progress=5.0), now)
    assert not fire  # below min progress
    fire, _ = ignition_check(_entry(last_progress=60.0), now)
    assert not fire  # above max progress
    fire, _ = ignition_check(_entry(inflow_sol=0.2), now)
    assert not fire  # inflow too small
    fire, reasons = ignition_check(_entry(inflow_sol=2.5, prev_inflow_sol=5.0), now)
    assert not fire and any("accelerat" in r for r in reasons)  # decelerating
    fire, _ = ignition_check(_entry(alerted=True), now)
    assert not fire  # already alerted
    fire, _ = ignition_check(_entry(inflow_sol=None), now)
    assert not fire  # no reading yet
    # first strong window with no history still fires
    fire, reasons = ignition_check(_entry(prev_inflow_sol=None), now)
    assert fire, reasons


def test_should_prune():
    now = time.time()
    assert should_prune(_entry(), now) is None
    assert should_prune(_entry(first_seen=now - 3600), now) is not None  # old
    assert should_prune(_entry(complete=True), now) is not None  # migrated
    assert should_prune(_entry(last_progress=97.0), now) is not None  # handoff
    stale = _entry(first_seen=now - 900, last_inflow_at=now - 900, inflow_sol=0.0)
    assert should_prune(stale, now) is not None  # dead 10m


class _FakeState:
    def __init__(self, v_sol, v_tok, supply):
        self.virtual_sol_reserves = v_sol
        self.virtual_token_reserves = v_tok
        self.token_total_supply = supply


def test_implied_mcap_sol():
    mcap = implied_mcap_sol(_FakeState(30_000_000_000, 800_000_000_000_000, 1_000_000_000))
    assert mcap is not None and mcap > 0
    assert implied_mcap_sol(_FakeState(1, 0, 1)) is None


def test_base58_roundtrip_for_ix_data():
    data = _make_create_ix_data()
    assert base58.b58decode(base58.b58encode(data)) == data
