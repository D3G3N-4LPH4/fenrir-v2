#!/usr/bin/env python3
"""Tests for the Robinhood v4 on-chain LP-lock check (lp_lock_v4).

All RPC traffic is faked via the ``transport`` seam — no network.
"""

from __future__ import annotations

import pytest

from fenrir.discovery.lp_lock_v4 import (
    DEAD_ADDRESS,
    KNOWN_LOCKERS,
    V4_POSITION_MANAGER,
    _addr_word,
    _decode_pool_key,
    _parse_mint_token_ids,
    _selector,
    _transfer_topic0,
    inspect_v4_lp_lock,
    pool_id_from_key,
)

TRANSFER_TOPIC0 = _transfer_topic0()
ZERO_TOPIC = "0x" + "00" * 32
GETINFO_SEL = _selector("getPoolAndPositionInfo(uint256)").hex()
OWNEROF_SEL = _selector("ownerOf(uint256)").hex()

C0 = "0x0b7d41fb08045A971D0f51b9beCE63cf24f3Ef2E"
C1 = "0x0Bd7D308f8E1639FAb988df18A8011f41EAcAD73"  # WETH
HOOKS = "0x0000000000000000000000000000000000000000"
EOA = "0x1111111111111111111111111111111111111111"
CONTRACT = "0x2222222222222222222222222222222222222222"


def _encode_pool_key_response(c0: str, c1: str, fee: int, tick_spacing: int, hooks: str) -> str:
    words = [
        _addr_word(c0),
        _addr_word(c1),
        fee.to_bytes(32, "big"),
        tick_spacing.to_bytes(32, "big", signed=True),
        _addr_word(hooks),
        (0).to_bytes(32, "big"),  # PositionInfo (unused)
    ]
    return "0x" + b"".join(words).hex()


def _mint_log(token_id: int) -> dict:
    return {
        "topics": [
            TRANSFER_TOPIC0,
            ZERO_TOPIC,
            ZERO_TOPIC,
            "0x" + token_id.to_bytes(32, "big").hex(),
        ],
        "blockNumber": hex(0xFFFF0),
    }


def _make_transport(
    *,
    owner: str = DEAD_ADDRESS,
    code: str = "0x",
    pool_key: tuple[str, str, int, int, str] | None = None,
    logs: list[dict] | None = None,
    fail_block: bool = False,
):
    """Fake JSON-RPC transport dispatching on method + calldata selector."""
    key_response = _encode_pool_key_response(*(pool_key or (C0, C1, 3000, 60, HOOKS)))
    owner_word = "0x" + _addr_word(owner).hex()
    log_list = logs if logs is not None else [_mint_log(7)]

    async def transport(method: str, params: list):
        if method == "eth_blockNumber":
            return None if fail_block else "0x100000"
        if method == "eth_getLogs":
            assert params[0]["address"] == V4_POSITION_MANAGER
            return log_list
        if method == "eth_call":
            data = params[0]["data"]
            assert params[0]["to"] == V4_POSITION_MANAGER
            if data[2:10] == GETINFO_SEL:
                return key_response
            if data[2:10] == OWNEROF_SEL:
                return owner_word
            raise AssertionError(f"unexpected eth_call data {data[:14]}")
        if method == "eth_getCode":
            return code
        raise AssertionError(f"unexpected method {method}")

    return transport


def _target_pool_id() -> str:
    return pool_id_from_key(C0, C1, 3000, 60, HOOKS)


class TestPoolId:
    def test_shape_and_determinism(self) -> None:
        pid = _target_pool_id()
        assert pid.startswith("0x") and len(pid) == 66
        assert pool_id_from_key(C0, C1, 3000, 60, HOOKS) == pid
        # Fee is part of the id.
        assert pool_id_from_key(C0, C1, 500, 60, HOOKS) != pid

    def test_decode_pool_key_roundtrip(self) -> None:
        raw = bytes.fromhex(_encode_pool_key_response(C0, C1, 3000, 60, HOOKS)[2:])
        decoded = _decode_pool_key(raw)
        assert decoded is not None
        c0, c1, fee, tick_spacing, hooks = decoded
        assert c0.lower() == C0.lower()
        assert c1.lower() == C1.lower()
        assert fee == 3000
        assert tick_spacing == 60
        assert int(hooks, 16) == 0

    def test_decode_rejects_short_data(self) -> None:
        assert _decode_pool_key(b"\x00" * 32) is None

    def test_parse_mint_token_ids(self) -> None:
        # A burn (Transfer TO zero, FROM a holder) is not a mint: skip it.
        burn_log = {
            "topics": [
                TRANSFER_TOPIC0,
                "0x" + _addr_word(EOA).hex(),
                ZERO_TOPIC,
                "0x" + (9).to_bytes(32, "big").hex(),
            ],
            "blockNumber": hex(0xFFFF0),
        }
        assert _parse_mint_token_ids([_mint_log(7), _mint_log(9), burn_log]) == [7, 9]
        assert _parse_mint_token_ids([{"topics": []}]) == []


class TestInspect:
    @pytest.mark.asyncio
    async def test_locked_when_burned(self) -> None:
        res = await inspect_v4_lp_lock(
            _target_pool_id(),
            age_minutes=21,
            transport=_make_transport(owner=DEAD_ADDRESS),
        )
        assert res.locked is True
        assert res.positions_checked == 1
        assert "locked" in res.detail

    @pytest.mark.asyncio
    async def test_locked_when_locker_holds(self) -> None:
        locker = next(iter(KNOWN_LOCKERS))
        res = await inspect_v4_lp_lock(
            _target_pool_id(),
            age_minutes=21,
            transport=_make_transport(owner=locker),
        )
        assert res.locked is True

    @pytest.mark.asyncio
    async def test_unlocked_when_eoa_holds(self) -> None:
        res = await inspect_v4_lp_lock(
            _target_pool_id(),
            age_minutes=21,
            transport=_make_transport(owner=EOA, code="0x"),
        )
        assert res.locked is False
        assert "pullable" in res.detail

    @pytest.mark.asyncio
    async def test_unknown_when_contract_holds(self) -> None:
        res = await inspect_v4_lp_lock(
            _target_pool_id(),
            age_minutes=21,
            transport=_make_transport(owner=CONTRACT, code="0x60006000"),
        )
        assert res.locked is None
        assert "unknown" in res.detail

    @pytest.mark.asyncio
    async def test_none_when_pool_mismatch(self) -> None:
        # Minted positions belong to a different pool → cannot determine.
        other_key = (EOA, C1, 3000, 60, HOOKS)
        res = await inspect_v4_lp_lock(
            _target_pool_id(),
            age_minutes=21,
            transport=_make_transport(pool_key=other_key),
        )
        assert res.locked is None
        assert "no positions found" in res.detail

    @pytest.mark.asyncio
    async def test_none_when_no_mints(self) -> None:
        res = await inspect_v4_lp_lock(
            _target_pool_id(),
            age_minutes=21,
            transport=_make_transport(logs=[]),
        )
        assert res.locked is None

    @pytest.mark.asyncio
    async def test_none_on_rpc_failure(self) -> None:
        res = await inspect_v4_lp_lock(
            _target_pool_id(),
            age_minutes=21,
            transport=_make_transport(fail_block=True),
        )
        assert res.locked is None

    @pytest.mark.asyncio
    async def test_rejects_non_pool_id(self) -> None:
        res = await inspect_v4_lp_lock(
            "0x1234",
            age_minutes=21,
            transport=_make_transport(),
        )
        assert res.locked is None
        assert "not a v4 pool id" in res.detail
