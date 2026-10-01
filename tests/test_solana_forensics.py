"""Tests for fenrir.discovery.solana_forensics (Solana distribution forensics).

Covers: owner aggregation across ATAs, vault/infrastructure exclusion,
top-1/top-10 math, fail-open on RPC failure, cache round-trip, and the
volatility_breakout fail-closed rule (require_distribution_known).
"""

from __future__ import annotations

import base64
import sys
from typing import Any

import pytest

sys.path.insert(0, "/home/hatch/workspace/fenrir-v2")

from solders.pubkey import Pubkey  # noqa: E402

from fenrir.discovery import solana_forensics as sf  # noqa: E402
from fenrir.discovery.filters import FilterEngine, FilterName  # noqa: E402
from fenrir.discovery.models import Chain, TokenSnapshot  # noqa: E402

SYSTEM = "11111111111111111111111111111111"
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


def _wallet(seed: int) -> Pubkey:
    from solders.keypair import Keypair

    return Keypair.from_seed(bytes([seed]) * 32).pubkey()


def base58_addr(seed: int) -> str:
    return str(_wallet(seed))


def _token_account_data(mint: Pubkey, owner: Pubkey, amount: int) -> str:
    raw = bytes(mint) + bytes(owner) + amount.to_bytes(8, "little") + bytes(93)
    return base64.b64encode(raw).decode()


class FakeTransport:
    """Injectable transport: method -> canned result."""

    def __init__(
        self,
        largest: list[dict[str, Any]],
        supply: int,
        owners: dict[str, dict[str, Any] | None],
    ):
        self.largest = largest
        self.supply = supply
        self.owners = owners  # token-account address -> account info
        self.owner_accounts: dict[str, dict[str, Any] | None] = {}  # owner -> account info

    async def __call__(self, method: str, params: list):
        if method == "getTokenLargestAccounts":
            return {"value": self.largest}
        if method == "getTokenSupply":
            return {"value": {"amount": str(self.supply), "decimals": 6}}
        if method == "getMultipleAccounts":
            addrs = params[0]
            if addrs and addrs[0] in self.owners:
                return {"value": [self.owners.get(a) for a in addrs]}
            return {"value": [self.owner_accounts.get(a) for a in addrs]}
        raise AssertionError(method)


def _system_owner_info() -> dict:
    return {"executable": False, "owner": SYSTEM, "lamports": 1}


def _program_owner_info(program: str) -> dict:
    return {"executable": False, "owner": program, "lamports": 1}


def _make_setup():
    mint = Pubkey.from_string(base58_addr(1))
    whale = Pubkey.from_string(base58_addr(2))
    retail1 = Pubkey.from_string(base58_addr(3))
    retail2 = Pubkey.from_string(base58_addr(4))
    vault = Pubkey.from_string(base58_addr(5))  # e.g. pool vault PDA
    # Whale holds TWO token accounts (must aggregate by owner, not account).
    largest = [
        {"address": base58_addr(10), "amount": "700000000"},  # vault 70%
        {"address": base58_addr(11), "amount": "100000000"},  # whale ATA 1
        {"address": base58_addr(12), "amount": "50000000"},  # whale ATA 2
        {"address": base58_addr(13), "amount": "30000000"},  # retail1
        {"address": base58_addr(14), "amount": "20000000"},  # retail2
    ]
    owners: dict[str, dict[str, Any] | None] = {
        base58_addr(10): {"data": [_token_account_data(mint, vault, 700_000_000), "base64"]},
        base58_addr(11): {"data": [_token_account_data(mint, whale, 100_000_000), "base64"]},
        base58_addr(12): {"data": [_token_account_data(mint, whale, 50_000_000), "base64"]},
        base58_addr(13): {"data": [_token_account_data(mint, retail1, 30_000_000), "base64"]},
        base58_addr(14): {"data": [_token_account_data(mint, retail2, 20_000_000), "base64"]},
    }
    t = FakeTransport(largest, 1_000_000_000, owners)
    t.owner_accounts = {
        str(vault): _program_owner_info(PUMP_PROGRAM),
        str(whale): _system_owner_info(),
        str(retail1): _system_owner_info(),
        str(retail2): _system_owner_info(),
    }
    return t


@pytest.mark.asyncio
async def test_aggregates_by_owner_and_excludes_vault() -> None:
    t = _make_setup()
    report = await sf.check_solana_distribution("MINT", transport=t)
    assert report is not None
    # Whale: (100M + 50M) / 1B = 15% top holder; top-10 = 15+3+2 = 20%.
    assert report.top_holder_pct == pytest.approx(15.0)
    assert report.top10_holder_pct == pytest.approx(20.0)
    # Vault (70%, owned by pump program) excluded from holders, counted separately.
    assert report.excluded_vault_pct == pytest.approx(70.0)
    assert report.holders_measured == 3


@pytest.mark.asyncio
async def test_executable_owner_counts_as_vault() -> None:
    t = _make_setup()
    prog = Pubkey.from_string(base58_addr(6))
    t.owner_accounts[str(prog)] = {"executable": True, "owner": SYSTEM, "lamports": 1}
    # Move retail2's tokens under a program-owned account.
    mint = Pubkey.from_string(base58_addr(1))
    t.owners[base58_addr(14)] = {"data": [_token_account_data(mint, prog, 20_000_000), "base64"]}
    report = await sf.check_solana_distribution("MINT", transport=t)
    assert report is not None
    assert report.top10_holder_pct == pytest.approx(18.0)  # 15 + 3
    assert report.excluded_vault_pct == pytest.approx(72.0)  # 70 + 2


@pytest.mark.asyncio
async def test_null_owner_account_on_curve_counts_as_holder() -> None:
    t = _make_setup()
    # retail2's wallet closed its SOL account (null info) but is on-curve:
    # still a real holder, not infrastructure.
    t.owner_accounts[str(Pubkey.from_string(base58_addr(4)))] = None
    report = await sf.check_solana_distribution("MINT", transport=t)
    assert report is not None
    assert report.top10_holder_pct == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_null_owner_account_off_curve_counts_as_vault() -> None:
    t = _make_setup()
    # PDA owner (off-curve) with no account, e.g. a burn address: program-
    # controlled, can't dump like a wallet → vault.
    pda, _ = Pubkey.find_program_address([b"burn"], Pubkey.from_string(base58_addr(9)))
    assert not pda.is_on_curve()
    mint = Pubkey.from_string(base58_addr(1))
    t.owners[base58_addr(14)] = {"data": [_token_account_data(mint, pda, 20_000_000), "base64"]}
    t.owner_accounts[str(pda)] = None
    report = await sf.check_solana_distribution("MINT", transport=t)
    assert report is not None
    assert report.top10_holder_pct == pytest.approx(18.0)  # retail2 excluded
    assert report.excluded_vault_pct == pytest.approx(72.0)


@pytest.mark.asyncio
async def test_fail_open_on_rpc_failure() -> None:
    async def boom(method: str, params: list):
        raise ConnectionError("down")

    assert await sf.check_solana_distribution("MINT", transport=boom) is None


@pytest.mark.asyncio
async def test_fail_open_on_empty_supply() -> None:
    t = _make_setup()
    t.supply = 0
    assert await sf.check_solana_distribution("MINT", transport=t) is None


def test_cache_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sf, "_CACHE_PATH", str(tmp_path / "forensics.json"))
    assert sf.get_cached_forensics("AbC") is None
    report = sf.SolanaForensicsReport(
        top_holder_pct=1.5,
        top10_holder_pct=12.0,
        holders_measured=9,
        excluded_vault_pct=80.0,
        detail="x",
    )
    sf.save_cached_forensics("AbC", report.as_dict())
    cached = sf.get_cached_forensics("abc")  # case-normalized
    assert cached is not None
    assert sf.SolanaForensicsReport.from_dict(cached).top10_holder_pct == 12.0
    # Inconclusive marker round-trips as a dict (caller checks the flag).
    sf.save_cached_forensics("zzz", {"_inconclusive": True}, ttl_seconds=60)
    assert sf.get_cached_forensics("zzz") == {"_inconclusive": True}


def _vb_snap(**overrides: Any) -> TokenSnapshot:
    base: dict[str, Any] = dict(
        chain=Chain.SOLANA,
        token_address="VB",
        market_cap_usd=300_000,
        liquidity_usd=40_000,
        volume_24h_usd=300_000,
        volume_1h_usd=60_000,
        age_minutes=300,
        holder_count=800,
        txns_24h_buys=500,
        txns_24h_sells=200,
        txns_1h_buys=575,
        txns_1h_sells=100,
        price_change_5m_pct=3.0,
        price_change_1h_pct=60.1,
        price_change_24h_pct=120.0,
        top_holder_pct=8.0,
        top10_holder_pct=45.0,
        dev_wallet_pct=5.0,
    )
    base.update(overrides)
    return TokenSnapshot(**base)


def test_vb_fails_closed_when_distribution_unknown() -> None:
    s = _vb_snap(
        top_holder_pct=None,
        top10_holder_pct=None,
        bundle_pct=None,
        sniper_pct=None,
    )
    s.safety.bundled_supply_pct = None
    r = FilterEngine().evaluate(s, FilterName.VOLATILITY_BREAKOUT)
    assert not r.passed
    assert any("distribution unknown" in f for f in r.failures)


def test_vb_passes_when_top10_known() -> None:
    r = FilterEngine().evaluate(_vb_snap(), FilterName.VOLATILITY_BREAKOUT)
    assert r.passed, r.failures


def test_vb_passes_on_bundle_data_alone() -> None:
    s = _vb_snap(top_holder_pct=None, top10_holder_pct=None, sniper_pct=None)
    s.bundle_pct = 12.0
    r = FilterEngine().evaluate(s, FilterName.VOLATILITY_BREAKOUT)
    assert r.passed, r.failures


def test_other_filters_still_fail_open_on_unknown_distribution() -> None:
    from fenrir.discovery.filters import FilterName as FN

    s = _vb_snap(
        top_holder_pct=None,
        top10_holder_pct=None,
        bundle_pct=None,
        sniper_pct=None,
    )
    # degen_launch does not set require_distribution_known: unknown
    # distribution must not fail it on that axis.
    r = FilterEngine().evaluate(s, FN.DEGEN_LAUNCH)
    assert not any("distribution unknown" in f for f in r.failures)


def test_only_vb_sets_require_distribution_known() -> None:
    from fenrir.discovery.filters import DEFAULT_THRESHOLDS
    from fenrir.discovery.filters import FilterName as FN

    for name, thr in DEFAULT_THRESHOLDS.items():
        if name == FN.VOLATILITY_BREAKOUT:
            assert thr.require_distribution_known is True
        else:
            assert thr.require_distribution_known is False
