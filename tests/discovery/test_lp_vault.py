"""Tests for fenrir.discovery.lp_vault (platform-vault LP detection)."""

import asyncio

import pytest

from fenrir.discovery.lp_vault import VaultCheck, check_lp_platform_vault


class FakeSession:
    """Minimal stand-in for aiohttp.ClientSession honoring _rpc's interface."""

    def __init__(self, handlers):
        self.handlers = handlers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, json=None, timeout=None):
        assert json is not None
        method = json["method"]
        return _FakePost(self.handlers[method](*json["params"]))


class _FakePost:
    def __init__(self, result):
        self._result = result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return {"result": self._result}


def run(coro):
    return asyncio.run(coro)


def _handlers(vault=True):
    owner = "VaultWallet1111111111111111111111111111111"
    top_acct = "TopAcct1111111111111111111111111111111111"
    n = 22020 if vault else 3
    return {
        "getTokenLargestAccounts": lambda mint: {
            "value": [{"address": top_acct, "amount": "6693000000"}]
        },
        "getTokenSupply": lambda mint: {"value": {"amount": "6693000000"}},
        "getAccountInfo": lambda addr, enc: {
            "value": {"data": {"parsed": {"info": {"owner": owner}}}}
        },
        "getTokenAccountsByOwner": lambda *a: {"value": [{}] * n},
    }


def test_platform_vault_detected(monkeypatch):
    import aiohttp

    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda trust_env=True: FakeSession(_handlers(vault=True)),
    )
    chk = run(check_lp_platform_vault("LPmint", "http://x", set()))
    assert isinstance(chk, VaultCheck)
    assert chk.is_platform_vault is True
    assert chk.holder == "VaultWallet1111111111111111111111111111111"
    assert chk.holder_share_pct == pytest.approx(100.0)
    assert chk.token_account_count == 22020


def test_distributed_lp_not_vault(monkeypatch):
    import aiohttp

    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda trust_env=True: FakeSession(_handlers(vault=False)),
    )
    chk = run(check_lp_platform_vault("LPmint", "http://x", set()))
    assert chk.is_platform_vault is False


def test_cached_holder_skips_account_scan(monkeypatch):
    import aiohttp

    owner = "VaultWallet1111111111111111111111111111111"
    top_acct = "TopAcct1111111111111111111111111111111111"
    calls: list[int] = []

    def _empty_accounts(*a) -> dict:
        calls.append(1)
        return {"value": []}

    def handlers():
        return {
            "getTokenLargestAccounts": lambda mint: {
                "value": [{"address": top_acct, "amount": "6693000000"}]
            },
            "getTokenSupply": lambda mint: {"value": {"amount": "6693000000"}},
            "getAccountInfo": lambda addr, enc: {
                "value": {"data": {"parsed": {"info": {"owner": owner}}}}
            },
            "getTokenAccountsByOwner": _empty_accounts,
        }

    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda trust_env=True: FakeSession(handlers()),
    )
    chk = run(check_lp_platform_vault("LPmint", "http://x", {owner}))
    assert chk.is_platform_vault is True
    assert chk.cached is True
    assert calls == []  # no account scan needed


def test_rpc_failure_fails_open(monkeypatch):
    import aiohttp

    def boom(trust_env=True):
        raise RuntimeError("down")

    monkeypatch.setattr(aiohttp, "ClientSession", boom)
    chk = run(check_lp_platform_vault("LPmint", "http://x", set()))
    assert chk.is_platform_vault is False
