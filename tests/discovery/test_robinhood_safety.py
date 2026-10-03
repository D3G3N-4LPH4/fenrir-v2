"""Tests for the on-chain Robinhood safety reader (Perceptor replacement)."""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fenrir.discovery.models import SafetySignals  # noqa: E402
from fenrir.discovery.providers.robinhood_safety import (  # noqa: E402
    RobinhoodSafetyProvider,
    _band_for,
    _decode_address_word,
    read_contract_powers,
)


def _word(addr: str) -> bytes:
    return bytes.fromhex("00" * 12 + addr.removeprefix("0x").lower())


def _owner_return(addr: str) -> str:
    return "0x" + _word(addr).hex()


def test_decode_address_word():
    addr = "0x1234567890abcdef1234567890abcdef12345678"
    assert _decode_address_word(_word(addr)) == addr.lower()
    assert _decode_address_word(b"\x00" * 31) is None
    assert _decode_address_word(b"") is None


class _FakeRpc:
    """Fake JSON-RPC keyed by (method, first-param-string)."""

    def __init__(self):
        self.calls: dict = {}

    def _key(self, method: str, params: list) -> tuple:
        p0 = params[0] if params else ""
        if isinstance(p0, dict):
            p0 = p0.get("to", "") + "|" + p0.get("data", "")
        return (method, str(p0))

    async def __call__(self, method: str, params: list):
        return self.calls.get(self._key(method, params))


def _powers_rpc(token: str, owner_ret: str | None, code: str) -> _FakeRpc:
    rpc = _FakeRpc()
    rpc.calls[("eth_call", f"{token}|0x8da5cb5b")] = owner_ret
    rpc.calls[("eth_getCode", token)] = code
    rpc.calls[("eth_getStorageAt", token)] = "0x" + "00" * 32
    return rpc


async def test_contract_powers_renounced_no_mint():
    token = "0x" + "aa" * 20
    rpc = _powers_rpc(
        token,
        _owner_return("0x0000000000000000000000000000000000000000"),
        "0x60806040",
    )
    out = await read_contract_powers(rpc, token)
    assert out["ownership_renounced"] is True
    assert out["mint_disabled"] is True  # renounced => nobody can mint
    assert out["blacklist_present"] is False
    assert out["freeze_disabled"] is True


async def test_contract_powers_live_owner_with_mint():
    token = "0x" + "bb" * 20
    owner = "0x" + "11" * 20
    rpc = _powers_rpc(token, _owner_return(owner), "0x60806040" + "40c10f19")
    out = await read_contract_powers(rpc, token)
    assert out["ownership_renounced"] is False
    assert out["owner_live"] is True
    assert out["mint_disabled"] is False  # live owner + mint fn
    assert out["blacklist_present"] is False


async def test_contract_powers_live_owner_no_mint_selector():
    # Live owner, no recognizable mint selector => mint UNKNOWN (not "disabled"):
    # custom mint functions slip past the bytecode scan.
    token = "0x" + "ee" * 20
    owner = "0x" + "22" * 20
    rpc = _powers_rpc(token, _owner_return(owner), "0x60806040")
    out = await read_contract_powers(rpc, token)
    assert out["ownership_renounced"] is False
    assert out["owner_live"] is True
    assert out["mint_disabled"] is None  # unknown, never asserted safe


async def test_contract_powers_blacklist_and_pause():
    token = "0x" + "cc" * 20
    rpc = _powers_rpc(token, None, "0x60806040" + "f9f92be4" + "8456cb59")
    out = await read_contract_powers(rpc, token)
    assert "ownership_renounced" not in out  # unknown stays unknown
    assert out["blacklist_present"] is True
    assert out["freeze_disabled"] is False


def test_band_for():
    assert _band_for(SafetySignals(honeypot=True))[0] == "high"
    assert _band_for(SafetySignals(risk_flags=["dev can pull liquidity"]))[0] == "high"
    assert _band_for(SafetySignals(blacklist_present=True))[0] == "high"
    assert _band_for(SafetySignals(mint_disabled=False))[0] == "medium"
    assert _band_for(SafetySignals(risk_flags=["pausable"]))[0] == "medium"
    assert _band_for(SafetySignals(lp_locked_or_burned=True)) == ("low", "Clean")


def test_provider_cache_roundtrip(tmp_path):
    cache = str(tmp_path / "safety.json")
    p = RobinhoodSafetyProvider(cache_path=cache)
    addr = "0x" + "dd" * 20
    assert p.cached_report(addr) is None
    s = SafetySignals(mint_disabled=True, lp_locked_or_burned=True, risk_flags=["x"])
    p._save_cache(
        {
            addr: {
                "ts": time.time(),
                "safety": {k: getattr(s, k) for k in SafetySignals.__dataclass_fields__},
                "band": "low",
                "band_label": "Clean",
                "headline": None,
                "signals": [],
                "investigation_id": "local:test",
            }
        }
    )
    hit = p.cached_report("0x" + "DD" * 20)  # case-insensitive
    assert hit is not None
    assert hit.band == "low"
    assert hit.safety.mint_disabled is True
    assert hit.safety.risk_flags == ["x"]
    # expired entries are ignored
    data = json.load(open(cache))
    data[addr]["ts"] = time.time() - 100_000
    json.dump(data, open(cache, "w"))
    assert p.cached_report(addr) is None
