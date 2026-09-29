#!/usr/bin/env python3
"""Tests for the Robinhood on-chain new-pair provider (rh_onchain)."""

import asyncio
import json
import os
import tempfile
import time

import pytest

from fenrir.discovery.providers.rh_onchain import (
    INITIALIZE_TOPIC0,
    NATIVE_ETH,
    USDG_ROBINHOOD,
    WETH_ROBINHOOD,
    RobinhoodPairMonitor,
    parse_initialize_log,
    to_checksum,
)

# A fixed "new token" address (lower than WETH so it sorts as currency0).
TOKEN = "0x1234567890abcdef1234567890abcdef12345678"


def _topic(addr: str) -> str:
    return "0x" + "0" * 24 + addr.lower().removeprefix("0x")


def _log(currency0: str, currency1: str, block: int = 100, topic0: str = INITIALIZE_TOPIC0) -> dict:
    return {
        "topics": [topic0, "0x" + "ab" * 32, _topic(currency0), _topic(currency1)],
        "blockNumber": hex(block),
        "transactionHash": "0x" + "cd" * 32,
    }


class TestChecksum:
    def test_eip55_vector(self):
        # Canonical EIP-55 test vector.
        assert (
            to_checksum("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed")
            == "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
        )

    def test_lowercase_passthrough_shape(self):
        out = to_checksum(TOKEN)
        assert out.startswith("0x") and len(out) == 42


class TestParseInitializeLog:
    def test_weth_pair_token_is_currency0(self):
        pool = parse_initialize_log(_log(TOKEN, WETH_ROBINHOOD))
        assert pool is not None
        assert pool.token_address == to_checksum(TOKEN)
        assert pool.base_address == to_checksum(WETH_ROBINHOOD)
        assert pool.block_number == 100

    def test_weth_pair_token_is_currency1(self):
        # Token address higher than WETH sorts second.
        high = "0xfff4567890abcdef1234567890abcdef12345678"
        pool = parse_initialize_log(_log(WETH_ROBINHOOD, high))
        assert pool is not None
        assert pool.token_address == to_checksum(high)
        assert pool.base_address == to_checksum(WETH_ROBINHOOD)

    def test_native_eth_pair(self):
        pool = parse_initialize_log(_log(NATIVE_ETH, TOKEN))
        assert pool is not None
        assert pool.token_address == to_checksum(TOKEN)
        assert pool.base_address == to_checksum(NATIVE_ETH)

    def test_usdg_pair(self):
        pool = parse_initialize_log(_log(TOKEN, USDG_ROBINHOOD))
        assert pool is not None
        assert pool.base_address == to_checksum(USDG_ROBINHOOD)

    def test_meme_meme_pair_skipped(self):
        other = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        assert parse_initialize_log(_log(TOKEN, other)) is None

    def test_base_base_pair_skipped(self):
        assert parse_initialize_log(_log(WETH_ROBINHOOD, USDG_ROBINHOOD)) is None

    def test_wrong_topic_skipped(self):
        assert parse_initialize_log(_log(TOKEN, WETH_ROBINHOOD, topic0="0x" + "00" * 32)) is None

    def test_malformed_log_skipped(self):
        assert parse_initialize_log({"topics": ["0x1234"]}) is None
        assert parse_initialize_log({}) is None


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def state_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _monitor_with_fake_rpc(state_path, latest, logs_by_range):
    mon = RobinhoodPairMonitor(state_path=state_path)

    async def fake_rpc(method, params):
        if method == "eth_blockNumber":
            return latest
        if method == "eth_getLogs":
            return logs_by_range
        return None

    mon._rpc = fake_rpc  # type: ignore[method-assign]
    return mon


class TestSync:
    def test_first_run_backfills_and_records(self, state_path):
        logs = [_log(TOKEN, WETH_ROBINHOOD, block=99990)]
        mon = _monitor_with_fake_rpc(state_path, hex(100_000), logs)
        result = _run(mon.sync())
        assert result["new"] == [to_checksum(TOKEN)]
        assert result["latest_block"] == 100_000
        with open(state_path) as f:
            store = json.load(f)
        assert store["last_block"] == 100_000
        entry = store["first_seen"][to_checksum(TOKEN)]
        assert entry["block"] == 99990
        assert entry["base"] == to_checksum(WETH_ROBINHOOD)
        _run(mon.close())

    def test_second_run_dedups_and_advances(self, state_path):
        logs = [_log(TOKEN, WETH_ROBINHOOD, block=99990)]
        mon = _monitor_with_fake_rpc(state_path, hex(100_000), logs)
        _run(mon.sync())
        mon2 = _monitor_with_fake_rpc(state_path, hex(100_010), logs)
        result = _run(mon2.sync())
        assert result["new"] == []  # already in registry
        assert result["latest_block"] == 100_010
        _run(mon.close())
        _run(mon2.close())

    def test_rpc_failure_keeps_cursor(self, state_path):
        mon = _monitor_with_fake_rpc(state_path, None, [])
        result = _run(mon.sync())
        assert result["new"] == []
        assert result["error"] == "rpc_unreachable"
        assert not os.path.exists(state_path)
        _run(mon.close())

    def test_fresh_addresses_window(self, state_path):
        now = time.time()
        store = {
            "last_block": 100,
            "first_seen": {
                "0xAAA": {"block": 90, "ts": now - 60, "base": "0xBBB"},
                "0xCCC": {"block": 80, "ts": now - 10 * 3600, "base": "0xBBB"},
            },
        }
        with open(state_path, "w") as f:
            json.dump(store, f)
        mon = RobinhoodPairMonitor(state_path=state_path)
        fresh = mon.fresh_addresses(max_age_hours=6.0)
        assert fresh == ["0xAAA"]
        _run(mon.close())

    def test_prune_state(self, state_path):
        now = time.time()
        store = {
            "last_block": 100,
            "first_seen": {
                "0xAAA": {"block": 90, "ts": now - 60, "base": "0xBBB"},
                "0xCCC": {"block": 80, "ts": now - 30 * 3600, "base": "0xBBB"},
            },
        }
        with open(state_path, "w") as f:
            json.dump(store, f)
        pruned = RobinhoodPairMonitor.prune_state(max_age_hours=24.0, now=now, path=state_path)
        assert pruned == 1
        with open(state_path) as f:
            reloaded = json.load(f)
        assert list(reloaded["first_seen"]) == ["0xAAA"]
