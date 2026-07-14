#!/usr/bin/env python3
"""
FENRIR - EVM discovery adapter + GoPlus provider tests

Pure GoPlus mapper + EvmAdapter discover/enrich with fakes — no network.
"""

from __future__ import annotations

import pytest

from fenrir.discovery.chains.evm import EvmAdapter
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.discovery.providers.goplus import (
    GOPLUS_CHAIN_IDS,
    GoPlusSecurity,
    parse_goplus,
)

# A GoPlus result modelled on the live PEPE/Ethereum response, plus a burned LP
# holder so LP-lock is exercised.
PEPE_LIKE = {
    "is_honeypot": "0",
    "buy_tax": "0",
    "sell_tax": "0.05",  # 5%
    "is_open_source": "1",
    "is_mintable": "0",
    "owner_address": "0x0000000000000000000000000000000000000000",
    "can_take_back_ownership": "0",
    "hidden_owner": "0",
    "transfer_pausable": "0",
    "is_blacklisted": "1",
    "holder_count": "570976",
    "creator_percent": "0.0",
    "holders": [{"address": "0xabc", "percent": "0.0855"}],
    "lp_holders": [
        {
            "address": "0x000000000000000000000000000000000000dead",
            "is_locked": 0,
            "percent": "0.95",
        },
        {"address": "0xpool", "is_locked": 0, "percent": "0.05"},
    ],
}


class TestParseGoPlus:
    def test_maps_core_signals(self) -> None:
        g = parse_goplus(PEPE_LIKE)
        s = g.safety
        assert s.honeypot is False
        assert s.buy_tax_pct == 0.0
        assert s.sell_tax_pct == 5.0  # 0.05 → 5%
        assert s.contract_verified is True
        assert s.mint_disabled is True  # is_mintable "0"
        assert s.freeze_disabled is True  # transfer_pausable "0"
        assert s.ownership_renounced is True  # owner is zero address
        assert s.blacklist_present is True
        assert g.holder_count == 570976
        assert g.top_holder_pct == pytest.approx(8.55)
        assert g.dev_wallet_pct == 0.0

    def test_lp_locked_from_burn_holder(self) -> None:
        g = parse_goplus(PEPE_LIKE)
        # 95% of LP in the burn address → locked/burned.
        assert g.safety.lp_locked_pct == pytest.approx(95.0)
        assert g.safety.lp_locked_or_burned is True

    def test_renounced_false_when_owner_present(self) -> None:
        res = dict(PEPE_LIKE, owner_address="0xdeadbeef00000000000000000000000000000001")
        assert parse_goplus(res).safety.ownership_renounced is False

    def test_unknowns_stay_none(self) -> None:
        g = parse_goplus({})  # empty result
        assert g.safety.honeypot is None
        assert g.safety.mint_disabled is None
        assert g.holder_count is None

    def test_chain_ids(self) -> None:
        assert GOPLUS_CHAIN_IDS[Chain.ETHEREUM] == "1"
        assert GOPLUS_CHAIN_IDS[Chain.BNB] == "56"
        assert GOPLUS_CHAIN_IDS[Chain.BASE] == "8453"


class _FakeDex:
    def __init__(self, chain: Chain) -> None:
        self.chain = chain

    async def fetch_boosted_addresses(self, chain: Chain) -> list[str]:
        assert chain is self.chain
        return ["0xtoken1", "0xtoken2"]

    async def fetch_snapshot(self, addr: str, chain: Chain) -> TokenSnapshot:
        return TokenSnapshot(
            chain=chain, token_address=addr, symbol="EVM", market_cap_usd=2_000_000
        )


class _FakeGoPlus:
    async def token_security(self, chain: Chain, address: str) -> GoPlusSecurity:
        return GoPlusSecurity(
            safety=SafetySignals(honeypot=False, contract_verified=True, mint_disabled=True),
            holder_count=4_000,
            top_holder_pct=3.0,
            dev_wallet_pct=1.0,
        )

    async def close(self) -> None:
        pass


class TestEvmAdapter:
    @pytest.mark.asyncio
    async def test_discover_maps_boosts_to_snapshots(self) -> None:
        adapter = EvmAdapter(Chain.ETHEREUM, _FakeDex(Chain.ETHEREUM), _FakeGoPlus())  # type: ignore[arg-type]
        snaps = await adapter.discover()
        assert [s.token_address for s in snaps] == ["0xtoken1", "0xtoken2"]
        assert all(s.chain is Chain.ETHEREUM for s in snaps)

    @pytest.mark.asyncio
    async def test_enrich_applies_goplus(self) -> None:
        adapter = EvmAdapter(Chain.BASE, _FakeDex(Chain.BASE), _FakeGoPlus())  # type: ignore[arg-type]
        snap = TokenSnapshot(chain=Chain.BASE, token_address="0xt")
        await adapter.enrich(snap)
        assert snap.safety.contract_verified is True
        assert snap.safety.mint_disabled is True
        assert snap.holder_count == 4_000
        assert snap.top_holder_pct == 3.0
        assert snap.dev_wallet_pct == 1.0
