#!/usr/bin/env python3
"""Tests for the Robinhood bundle + deployer-cluster check (bundle_check).

All RPC traffic is faked via the ``transport`` seam and funder lookups are
injected — no network.
"""

from __future__ import annotations

import pytest

from fenrir.discovery.bundle_check import (
    INITIALIZE_TOPIC0,
    V4_POOL_MANAGER,
    _merge_clusters,
    _timing_clusters,
    check_bundle_and_deployer,
)
from fenrir.discovery.filters import FilterEngine, FilterName
from fenrir.discovery.lp_lock_v4 import _addr_word, _selector
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.discovery.scoring import ScoringEngine

TRANSFER_T0 = "0x" + ("ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")
INIT_BLOCK = 0xFFF00
POOL_ID = "0x" + "ab" * 32
TOKEN = "0x" + "11" * 20
DEV = "0x" + "dd" * 20
FUNDER = "0x" + "f0" * 20
CEX = "0x" + "ce" * 20
SUPPLY = 1_000_000 * 10**18
E18 = 10**18


def _word(addr: str) -> str:
    return "0x" + addr.lower().removeprefix("0x").rjust(64, "0")


def _transfer_log(frm: str, to: str, amount: int, block: int) -> dict:
    return {
        "topics": [TRANSFER_T0, _word(frm), _word(to)],
        "data": hex(amount),
        "blockNumber": hex(block),
    }


def _init_log() -> dict:
    return {
        "topics": [INITIALIZE_TOPIC0, _word(POOL_ID)],
        "data": "0x",
        "blockNumber": hex(INIT_BLOCK),
    }


def _wallet(i: int) -> str:
    return "0x" + f"{i:040x}"


def _make_transport(
    *,
    buyer_logs: list[dict],
    seller_logs: list[dict] | None = None,
    deployer: str = DEV,
    deployer_balance: int = 0,
    deployer_flows: list[dict] | None = None,
    total_supply: int = SUPPLY,
    fail: bool = False,
    no_init: bool = False,
):
    """Fake JSON-RPC transport dispatching on address + topic/selector."""
    total_hex = "0x" + total_supply.to_bytes(32, "big").hex()
    bal_hex = "0x" + deployer_balance.to_bytes(32, "big").hex()
    dev_word = "0x" + _addr_word(deployer).hex()
    flows = deployer_flows if deployer_flows is not None else []
    sells = seller_logs if seller_logs is not None else []

    async def transport(method: str, params: list):
        if fail:
            return None
        if method == "eth_blockNumber":
            return hex(INIT_BLOCK + 10_000)
        if method == "eth_getLogs":
            f = params[0]
            addr = f["address"].lower()
            topics = f["topics"]
            if addr == V4_POOL_MANAGER.lower():
                if topics[0] == INITIALIZE_TOPIC0 and not no_init:
                    return [_init_log()]
                return []
            if addr == TOKEN.lower():
                if topics[0] == TRANSFER_T0 and topics[1] == _word(V4_POOL_MANAGER):
                    lo = int(f["fromBlock"], 16)
                    hi = int(f["toBlock"], 16)
                    return [lg for lg in buyer_logs if lo <= int(lg["blockNumber"], 16) <= hi]
                if (
                    topics[0] == TRANSFER_T0
                    and topics[1] is None
                    and len(topics) > 2
                    and topics[2] == _word(V4_POOL_MANAGER)
                ):
                    lo = int(f["fromBlock"], 16)
                    hi = int(f["toBlock"], 16)
                    return [lg for lg in sells if lo <= int(lg["blockNumber"], 16) <= hi]
                if topics[0] == TRANSFER_T0 and topics[1] == _word(deployer):
                    lo = int(f["fromBlock"], 16)
                    hi = int(f["toBlock"], 16)
                    return [lg for lg in flows if lo <= int(lg["blockNumber"], 16) <= hi]
            return []
        if method == "eth_call":
            data = params[0]["data"]
            sel = data[2:10]
            if sel == _selector("totalSupply()").hex():
                return total_hex
            if sel == _selector("deployer()").hex():
                return dev_word
            if sel == _selector("balanceOf(address)").hex():
                return bal_hex
            raise AssertionError(f"unexpected eth_call {data[:14]}")
        raise AssertionError(f"unexpected method {method}")

    return transport


def _lookup(funders: dict[str, str]):
    async def lookup(session, wallet: str, before_block: int | None):
        return funders.get(wallet.lower())

    return lookup


def _run(**kw):
    return check_bundle_and_deployer(TOKEN, POOL_ID, age_minutes=30, **kw)


class TestTimingClusters:
    def test_same_block_group_forms_cluster(self) -> None:
        buys = {f"0x{i:040x}": (100, 10) for i in range(5)}
        clusters = _timing_clusters(buys)
        assert len(clusters) == 1 and len(clusters[0]) == 5

    def test_spread_buys_form_no_cluster(self) -> None:
        buys = {f"0x{i:040x}": (100 + i * 5, 10) for i in range(20)}
        assert _timing_clusters(buys) == []

    def test_three_block_run_forms_cluster(self) -> None:
        buys = {f"0x{i:040x}": (100 + (i % 3), 10) for i in range(6)}
        clusters = _timing_clusters(buys)
        assert len(clusters) == 1

    def test_merge_unions_overlapping(self) -> None:
        merged = _merge_clusters([{"a", "b"}, {"b", "c"}, {"x", "y"}])
        assert len(merged) == 2
        assert {"a", "b", "c"} in merged


class TestBundleDetection:
    async def test_bundled_cluster_detected(self) -> None:
        # 8 wallets buy in the same 2 blocks, all funded by one EOA.
        wallets = [_wallet(1 + i) for i in range(8)]
        logs = [
            _transfer_log(V4_POOL_MANAGER, w, 50_000 * E18, INIT_BLOCK + (i % 2))
            for i, w in enumerate(wallets)
        ]
        funders = {w: FUNDER for w in wallets}
        report = await _run(
            transport=_make_transport(buyer_logs=logs),
            funder_lookup=_lookup(funders),
        )
        assert report is not None
        assert report.bundled_supply_pct == pytest.approx(40.0)
        assert report.cluster_count == 1
        assert report.largest_cluster_pct == pytest.approx(40.0)

    async def test_organic_distribution_passes(self) -> None:
        # 60 wallets, one buy every 5 blocks, distinct funders, varied sizes —
        # no timing run (gap > 1), no shared funder.
        wallets = [_wallet(100 + i) for i in range(60)]
        logs = [
            _transfer_log(V4_POOL_MANAGER, w, (100 + i * 7) * E18, INIT_BLOCK + i * 5)
            for i, w in enumerate(wallets)
        ]
        funders = {w: _wallet(900 + i) for i, w in enumerate(wallets)}
        report = await _run(
            transport=_make_transport(buyer_logs=logs),
            funder_lookup=_lookup(funders),
        )
        assert report is not None
        assert report.bundled_supply_pct == pytest.approx(0.0)
        assert report.cluster_count == 0

    async def test_timing_without_shared_funder(self) -> None:
        # Same-block buys, but all funded by a denylisted CEX hot wallet:
        # timing cluster forms, funder merge is suppressed.
        wallets = [_wallet(1 + i) for i in range(6)]
        logs = [_transfer_log(V4_POOL_MANAGER, w, 10_000 * E18, INIT_BLOCK) for w in wallets]
        funders = {w: CEX for w in wallets}
        report = await _run(
            transport=_make_transport(buyer_logs=logs),
            funder_lookup=_lookup(funders),
            funder_denylist=frozenset({CEX.lower()}),
        )
        assert report is not None
        assert report.cluster_count == 1  # timing only
        assert report.bundled_supply_pct == pytest.approx(6.0)

    async def test_wash_trading_nets_to_zero(self) -> None:
        # A bot that buys and sells the same tokens in the window nets ~0:
        # gross first-buy accounting would double-count recycled supply.
        wallets = [_wallet(1 + i) for i in range(5)]
        buys = [_transfer_log(V4_POOL_MANAGER, w, 50_000 * E18, INIT_BLOCK) for w in wallets]
        sells = [_transfer_log(w, V4_POOL_MANAGER, 50_000 * E18, INIT_BLOCK + 1) for w in wallets]
        report = await _run(
            transport=_make_transport(buyer_logs=buys, seller_logs=sells),
            funder_lookup=_lookup({}),
        )
        assert report is not None
        assert report.bundled_supply_pct == pytest.approx(0.0)

    async def test_rpc_failure_fail_open(self) -> None:
        report = await _run(
            transport=_make_transport(buyer_logs=[], fail=True),
            funder_lookup=_lookup({}),
        )
        assert report is None

    async def test_rejects_non_pool_id(self) -> None:
        report = await check_bundle_and_deployer(
            TOKEN, "0x1234", transport=_make_transport(buyer_logs=[])
        )
        assert report is None


class TestDeployerTracing:
    async def test_deployer_distributed_supply_flagged(self) -> None:
        # Deployer spreads to 15 wallets, holds nothing — the classic pattern.
        recipients = [_wallet(200 + i) for i in range(15)]
        flows = [_transfer_log(DEV, r, 20_000 * E18, INIT_BLOCK + 5) for r in recipients]
        report = await _run(
            transport=_make_transport(buyer_logs=[], deployer_flows=flows),
            funder_lookup=_lookup({}),
        )
        assert report is not None
        assert report.deployer_address == DEV
        assert report.deployer_holding_pct == pytest.approx(0.0)
        assert report.deployer_distributed_wallets == 15
        assert report.deployer_distributed_pct == pytest.approx(30.0)
        assert report.deployer_cluster_pct == pytest.approx(30.0)

    async def test_serial_launcher_funder_flagged(self) -> None:
        # Deployer funded by the known FARTDOG serial-launcher dev.
        serial = "0xea1a19ed166097b3574a68f363292ef006700c90"
        report = await _run(
            transport=_make_transport(buyer_logs=[]),
            funder_lookup=_lookup({DEV.lower(): serial}),
        )
        assert report is not None
        assert report.deployer_funder == serial
        assert report.deployer_funder_is_serial_launcher is True

    async def test_clean_deployer_no_flags(self) -> None:
        report = await _run(
            transport=_make_transport(buyer_logs=[], deployer_balance=0),
            funder_lookup=_lookup({DEV.lower(): _wallet(999)}),
        )
        assert report is not None
        assert report.deployer_funder_is_serial_launcher is False
        assert report.deployer_distributed_wallets == 0


def _snap_with_bundle(bundled: float | None, deployer_cluster: float | None) -> TokenSnapshot:
    return TokenSnapshot(
        chain=Chain.ROBINHOOD,
        token_address="0x" + "ab" * 20,
        market_cap_usd=20_000,
        liquidity_usd=5_000,
        volume_24h_usd=10_000,
        volume_1h_usd=2_000,
        age_minutes=21,
        holder_count=100,
        txns_24h_buys=50,
        txns_24h_sells=10,
        txns_1h_buys=30,
        txns_1h_sells=10,
        price_change_1h_pct=10.0,
        price_change_24h_pct=25.0,
        top_holder_pct=8.0,
        top10_holder_pct=40.0,
        bundle_pct=bundled,
        safety=SafetySignals(
            mint_disabled=True,
            freeze_disabled=True,
            lp_locked_or_burned=True,
            honeypot=False,
            buy_tax_pct=0.0,
            sell_tax_pct=0.0,
            blacklist_present=False,
            bundled_supply_pct=bundled,
            deployer_cluster_pct=deployer_cluster,
        ),
    )


class TestBundleFilterGates:
    """Bubblemaps bands in the filter engine: >30% fails the risk-on filters,
    5-15% warns, unknown stays fail-open."""

    def setup_method(self) -> None:
        self.engine = FilterEngine()

    def test_bundled_over_30_fails_low_cap_alpha(self) -> None:
        res = self.engine.evaluate(_snap_with_bundle(35.0, None), FilterName.LOW_CAP_ALPHA)
        assert not res.passed
        assert any("Bundled" in f for f in res.failures)

    def test_deployer_cluster_over_30_fails(self) -> None:
        res = self.engine.evaluate(_snap_with_bundle(None, 35.0), FilterName.LOW_CAP_ALPHA)
        assert not res.passed
        assert any("Deployer cluster" in f for f in res.failures)

    def test_bundled_moderate_warns_not_fails(self) -> None:
        res = self.engine.evaluate(_snap_with_bundle(10.0, None), FilterName.LOW_CAP_ALPHA)
        assert not any("Bundled" in f for f in res.failures)
        assert any("moderate" in w for w in res.warnings)

    def test_unknown_bundle_fail_open(self) -> None:
        res = self.engine.evaluate(_snap_with_bundle(None, None), FilterName.LOW_CAP_ALPHA)
        assert not any("Bundled" in f or "Deployer cluster" in f for f in res.failures)

    def test_serial_launcher_warns(self) -> None:
        snap = _snap_with_bundle(None, None)
        snap.safety.deployer_funder_is_serial_launcher = True
        res = self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA)
        assert any("serial launcher" in w for w in res.warnings)


class TestBundleScoring:
    def test_penalties_and_cap(self) -> None:
        snap = _snap_with_bundle(20.0, 20.0)
        snap.safety.deployer_holding_pct = 8.0
        score = ScoringEngine().score(snap).safety
        # 15 + 10 + 10 = 35, capped at 25.
        base = _snap_with_bundle(None, None)
        base_score = ScoringEngine().score(base).safety
        assert score == pytest.approx(base_score - 25.0)

    def test_no_penalty_when_clean(self) -> None:
        snap = _snap_with_bundle(3.0, 4.0)
        snap.safety.deployer_holding_pct = 0.5
        base = _snap_with_bundle(None, None)
        assert ScoringEngine().score(snap).safety == pytest.approx(
            ScoringEngine().score(base).safety
        )


class TestBundleCacheTTL:
    def test_inconclusive_uses_short_ttl(self, tmp_path, monkeypatch) -> None:
        import fenrir.discovery.bundle_check as bc

        monkeypatch.setattr(bc, "_CACHE_PATH", str(tmp_path / "bundle.json"))
        bc.save_cached_bundle_report(
            "0xabc", {"_inconclusive": True}, ttl_seconds=bc.INCONCLUSIVE_TTL_S
        )
        entry = bc._load_cache()["0xabc"]
        assert entry["ttl"] == bc.INCONCLUSIVE_TTL_S
        assert bc.get_cached_bundle_report("0xABC") == {"_inconclusive": True}

    def test_short_ttl_expires(self, tmp_path, monkeypatch) -> None:
        import time

        import fenrir.discovery.bundle_check as bc

        monkeypatch.setattr(bc, "_CACHE_PATH", str(tmp_path / "bundle.json"))
        # Margins kept wide enough to be deterministic on slow filesystems
        # (a 10ms TTL raced the save->read round-trip on Windows).
        bc.save_cached_bundle_report("0xabc", {"_inconclusive": True}, ttl_seconds=0.5)
        assert bc.get_cached_bundle_report("0xabc") == {"_inconclusive": True}
        time.sleep(0.6)
        assert bc.get_cached_bundle_report("0xabc") is None

    def test_default_ttl_stays_24h(self, tmp_path, monkeypatch) -> None:
        import fenrir.discovery.bundle_check as bc

        monkeypatch.setattr(bc, "_CACHE_PATH", str(tmp_path / "bundle.json"))
        bc.save_cached_bundle_report("0xdef", {"bundled_supply_pct": 3.0})
        entry = bc._load_cache()["0xdef"]
        assert entry["ttl"] == 24 * 3600
