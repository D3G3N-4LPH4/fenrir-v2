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


def test_account_scan_uses_count_only_slice(monkeypatch):
    """getTokenAccountsByOwner must request a dataSlice, not full account data."""
    import aiohttp

    seen_params: list = []

    def handlers():
        h = _handlers(vault=True)
        orig = h["getTokenAccountsByOwner"]

        def wrapped(*a):
            seen_params.append(a)
            return orig(*a)

        h["getTokenAccountsByOwner"] = wrapped
        return h

    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda trust_env=True: FakeSession(handlers()),
    )
    run(check_lp_platform_vault("LPmint", "http://x", set()))
    assert len(seen_params) == 1
    assert seen_params[0][2]["dataSlice"] == {"offset": 0, "length": 0}


def test_pool_check_caches_positive_result(monkeypatch, tmp_path):
    """Second check_pool_lp_vault call for a pool must not touch the network."""
    import aiohttp

    from fenrir.discovery import lp_vault

    monkeypatch.setattr(lp_vault, "DEFAULT_CHECK_CACHE_PATH", tmp_path / "checks.json")
    monkeypatch.setattr(lp_vault, "load_known_vaults", lambda: set())
    monkeypatch.setattr(lp_vault, "save_known_vaults", lambda v: None)

    async def _fake_resolve(pool):
        return "LPmint"

    monkeypatch.setattr(lp_vault, "resolve_lp_mint", _fake_resolve)
    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda trust_env=True: FakeSession(_handlers(vault=True)),
    )
    first = run(lp_vault.check_pool_lp_vault("PoolA", "http://x"))
    assert first.is_platform_vault is True
    assert first.cached is False

    def boom(trust_env=True):
        raise AssertionError("network should not be touched on cache hit")

    async def _boom_resolve(pool):
        raise AssertionError("resolve should not be touched on cache hit")

    monkeypatch.setattr(aiohttp, "ClientSession", boom)
    monkeypatch.setattr(lp_vault, "resolve_lp_mint", _boom_resolve)
    second = run(lp_vault.check_pool_lp_vault("PoolA", "http://x"))
    assert second.is_platform_vault is True
    assert second.cached is True
    assert second.holder == first.holder


def test_pool_check_caches_raydium_miss(monkeypatch, tmp_path):
    """A pool with no LP mint (pre-graduation) resolves Raydium only once."""
    from fenrir.discovery import lp_vault

    monkeypatch.setattr(lp_vault, "DEFAULT_CHECK_CACHE_PATH", tmp_path / "checks.json")

    calls: list = []

    async def _no_mint(pool):
        calls.append(pool)
        return None

    monkeypatch.setattr(lp_vault, "resolve_lp_mint", _no_mint)
    first = run(lp_vault.check_pool_lp_vault("PoolB", "http://x"))
    assert first.is_platform_vault is False
    second = run(lp_vault.check_pool_lp_vault("PoolB", "http://x"))
    assert second.is_platform_vault is False
    assert second.cached is True
    assert calls == ["PoolB"]


def test_burned_lp_supply_zero(monkeypatch):
    """LP mint with zero supply => burned => treated as locked."""
    import aiohttp

    handlers = dict(_handlers(vault=False))
    handlers["getTokenSupply"] = lambda mint: {"value": {"amount": "0"}}
    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda trust_env=True: FakeSession(handlers),
    )
    chk = run(check_lp_platform_vault("LPmint", "http://x", set()))
    assert chk.burned is True
    assert chk.is_platform_vault is False


def test_resolve_lp_mint_guards_none_pool(monkeypatch):
    """Raydium API returning data:[None] must not crash (live bug 2026-10-01)."""
    import aiohttp

    from fenrir.discovery import lp_vault

    class _Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"data": [None]}

    class _Session:
        def get(self, *a, **k):
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(aiohttp, "ClientSession", lambda trust_env=True: _Session())
    assert run(lp_vault.resolve_lp_mint("PoolX")) is None


def test_resolve_pumpswap_lp_mint(monkeypatch):
    """PumpSwap pool account parses to its lp_mint (offset verified live)."""
    import base64

    import aiohttp

    from fenrir.discovery import lp_vault

    # Build a synthetic PumpSwap Pool account: discriminator(8) + bump(1) +
    # index(2) + creator(32) + base(32) + quote(32) + lp_mint(32).
    want = bytes(range(32))
    raw = b"\x00" * 8 + b"\x01" + b"\x02\x03" + b"\x11" * 32 + b"\x22" * 32 + b"\x33" * 32 + want
    assert len(raw) >= lp_vault.PUMPSWAP_LP_MINT_OFFSET + 32

    acct = {
        "value": {
            "owner": lp_vault.PUMPSWAP_PROGRAM,
            "data": [base64.b64encode(raw).decode(), "base64"],
        }
    }
    handlers = {"getAccountInfo": lambda *a: acct}
    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda trust_env=True: FakeSession(handlers),
    )

    async def _go():
        async with aiohttp.ClientSession(trust_env=True) as s:
            return await lp_vault.resolve_pumpswap_lp_mint("PoolP", "http://x", s)

    assert run(_go()) == lp_vault._b58encode(want)


def test_resolve_pumpswap_rejects_non_pumpswap_owner(monkeypatch):
    import base64

    import aiohttp

    from fenrir.discovery import lp_vault

    raw = b"\x00" * (lp_vault.PUMPSWAP_LP_MINT_OFFSET + 32)
    acct = {
        "value": {
            "owner": "SomeOtherProgram111111111111111111111111111",
            "data": [base64.b64encode(raw).decode(), "base64"],
        }
    }
    handlers = {"getAccountInfo": lambda *a: acct}
    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda trust_env=True: FakeSession(handlers),
    )

    async def _go():
        async with aiohttp.ClientSession(trust_env=True) as s:
            return await lp_vault.resolve_pumpswap_lp_mint("PoolP", "http://x", s)

    assert run(_go()) is None
