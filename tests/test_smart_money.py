#!/usr/bin/env python3
"""
FENRIR - Smart-Money Tracker Test Suite

Covers venue-agnostic buy detection (token-balance deltas), SOL-spent estimation,
per-wallet tiers, candidate shaping, the routing curve check, per-mint cooldown,
and seen-signature dedup. Network is fully mocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fenrir.config import BotConfig
from fenrir.protocol.pumpfun import BondingCurveState
from fenrir.trading.smart_money import WSOL_MINT, SmartMoneyTracker

T0 = datetime(2020, 1, 1, 12, 0, 0)
WALLET = "So11111111111111111111111111111111111111112"
A_WALLET = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
MINT = "9Gtngjfj2rsivH5ZhYZLRLD8K6jqeYft3bNAAmtfpump"
OTHER = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"


def _bal(owner: str, mint: str, amt: float):
    return SimpleNamespace(owner=owner, mint=mint, ui_token_amount=SimpleNamespace(ui_amount=amt))


def _tx(pre, post, keys=None, pre_lamports=None, post_lamports=None):
    meta = SimpleNamespace(
        pre_token_balances=pre,
        post_token_balances=post,
        pre_balances=pre_lamports or [],
        post_balances=post_lamports or [],
    )
    inner = SimpleNamespace(message=SimpleNamespace(account_keys=keys or []))
    return SimpleNamespace(transaction=SimpleNamespace(meta=meta, transaction=inner))


@pytest.fixture
def tracker():
    c = BotConfig()
    c.smart_money_wallets = [WALLET]
    c.smart_money_priority_wallets = [A_WALLET]
    c.smart_money_cooldown_minutes = 60.0
    c.smart_money_max_candidates_per_cycle = 3
    return SmartMoneyTracker(c, AsyncMock(), MagicMock(), MagicMock())


class TestDetectBuys:
    def test_new_buy_detected(self, tracker):
        tx = _tx([], [_bal(WALLET, MINT, 100.0)])
        assert tracker._detect_buys(tx, WALLET) == [(MINT, 0.0)]  # no keys → SOL unknown

    def test_increase_detected(self, tracker):
        tx = _tx([_bal(WALLET, MINT, 10.0)], [_bal(WALLET, MINT, 50.0)])
        assert tracker._detect_buys(tx, WALLET) == [(MINT, 0.0)]

    def test_no_change_ignored(self, tracker):
        tx = _tx([_bal(WALLET, MINT, 50.0)], [_bal(WALLET, MINT, 50.0)])
        assert tracker._detect_buys(tx, WALLET) == []

    def test_decrease_ignored(self, tracker):
        tx = _tx([_bal(WALLET, MINT, 50.0)], [_bal(WALLET, MINT, 10.0)])  # sold
        assert tracker._detect_buys(tx, WALLET) == []

    def test_other_owner_ignored(self, tracker):
        tx = _tx([], [_bal(OTHER, MINT, 100.0)])
        assert tracker._detect_buys(tx, WALLET) == []

    def test_wsol_excluded(self, tracker):
        tx = _tx([], [_bal(WALLET, WSOL_MINT, 5.0)])
        assert tracker._detect_buys(tx, WALLET) == []

    def test_no_meta_safe(self, tracker):
        tx = SimpleNamespace(transaction=SimpleNamespace(meta=None))
        assert tracker._detect_buys(tx, WALLET) == []

    def test_sol_spent_from_native_delta(self, tracker):
        # Wallet is fee payer (index 0); spent 2 SOL (5 → 3).
        tx = _tx(
            [],
            [_bal(WALLET, MINT, 100.0)],
            keys=[WALLET],
            pre_lamports=[5_000_000_000],
            post_lamports=[3_000_000_000],
        )
        assert tracker._detect_buys(tx, WALLET) == [(MINT, 2.0)]


class TestTiers:
    def test_priority_is_a_tier(self, tracker):
        assert tracker._tier(A_WALLET) == "A"
        assert tracker._tier(WALLET) == "B"

    def test_tracked_wallets_union_deduped(self, tracker):
        tracker.config.smart_money_priority_wallets = [A_WALLET, WALLET]  # overlap
        assert tracker._tracked_wallets() == [WALLET, A_WALLET]


def _curve(complete: bool = False, real_sol_lamports: int = 5_000_000_000):
    return BondingCurveState(
        virtual_token_reserves=1_073_000_000_000_000,
        virtual_sol_reserves=30_000_000_000,
        real_token_reserves=793_100_000_000_000,
        real_sol_reserves=real_sol_lamports,
        token_total_supply=1_000_000_000_000_000,
        complete=complete,
    )


class TestBuildCandidate:
    @pytest.mark.asyncio
    async def test_fresh_candidate_enriched_from_curve(self, tracker):
        tracker._curve_state = AsyncMock(return_value=_curve())  # type: ignore[method-assign]
        c = await tracker._build_candidate(A_WALLET, MINT, 2.5, T0)
        assert c is not None
        assert c["source"] == "smart_money"
        assert c["smart_money_wallet"] == A_WALLET
        assert c["smart_money_tier"] == "A"
        assert c["smart_money_sol"] == 2.5
        assert c["migrated"] is False
        assert c["tier"] is None
        # Enriched with real on-chain numbers (not the bare zero defaults).
        assert c["bonding_curve_state"] is not None
        assert c["initial_liquidity_sol"] == 5.0  # 5e9 lamports
        assert c["market_cap_sol"] > 0

    @pytest.mark.asyncio
    async def test_migrated_candidate(self, tracker):
        tracker._curve_state = AsyncMock(return_value=None)  # type: ignore[method-assign]
        c = await tracker._build_candidate(WALLET, MINT, 0.0, T0)
        assert c is not None
        assert c["smart_money_tier"] == "B"
        assert c["migrated"] is True
        assert c["tier"] == "mid"
        assert c["bonding_curve_state"] is None
        assert "initial_liquidity_sol" not in c  # no curve → no curve-derived numbers

    @pytest.mark.asyncio
    async def test_cooldown_blocks_then_allows(self, tracker):
        tracker._curve_state = AsyncMock(return_value=_curve())  # type: ignore[method-assign]
        tracker._cooldown[MINT] = T0
        assert await tracker._build_candidate(WALLET, MINT, 0.0, T0 + timedelta(minutes=5)) is None
        assert (
            await tracker._build_candidate(WALLET, MINT, 0.0, T0 + timedelta(minutes=61))
            is not None
        )


class TestCurveState:
    @pytest.mark.asyncio
    async def test_no_account_returns_none(self, tracker):
        tracker.pumpfun.derive_bonding_curve_address.return_value = (MagicMock(), 0)
        tracker.client.get_account_info = AsyncMock(return_value=None)
        assert await tracker._curve_state(MINT) is None

    @pytest.mark.asyncio
    async def test_complete_curve_returns_none(self, tracker):
        tracker.pumpfun.derive_bonding_curve_address.return_value = (MagicMock(), 0)
        tracker.client.get_account_info = AsyncMock(return_value=b"x" * 80)
        tracker.pumpfun.decode_bonding_curve.return_value = _curve(complete=True)
        assert await tracker._curve_state(MINT) is None

    @pytest.mark.asyncio
    async def test_live_curve_returned(self, tracker):
        tracker.pumpfun.derive_bonding_curve_address.return_value = (MagicMock(), 0)
        tracker.client.get_account_info = AsyncMock(return_value=b"x" * 80)
        live = _curve(complete=False)
        tracker.pumpfun.decode_bonding_curve.return_value = live
        assert await tracker._curve_state(MINT) is live


class TestDetectSells:
    def test_sell_detected(self, tracker):
        tx = _tx([_bal(WALLET, MINT, 100.0)], [_bal(WALLET, MINT, 10.0)])  # reduced
        assert tracker._detect_sells(tx, WALLET) == [MINT]

    def test_full_dump_detected(self, tracker):
        tx = _tx([_bal(WALLET, MINT, 100.0)], [])  # gone to zero
        assert tracker._detect_sells(tx, WALLET) == [MINT]

    def test_buy_is_not_a_sell(self, tracker):
        tx = _tx([], [_bal(WALLET, MINT, 100.0)])  # increase
        assert tracker._detect_sells(tx, WALLET) == []

    def test_wsol_excluded(self, tracker):
        tx = _tx([_bal(WALLET, WSOL_MINT, 5.0)], [_bal(WALLET, WSOL_MINT, 0.0)])
        assert tracker._detect_sells(tx, WALLET) == []


class TestNewBuys:
    @pytest.mark.asyncio
    async def test_dedup_seen_signatures(self, tracker):
        sig = SimpleNamespace(signature="SIG1", err=None)
        tracker.client.get_recent_signatures = AsyncMock(return_value=[sig])
        tracker.client.get_transaction = AsyncMock(return_value=_tx([], [_bal(WALLET, MINT, 10.0)]))

        buys, sells = await tracker._new_buys(WALLET)
        assert buys == [(MINT, 0.0)]
        assert sells == []
        buys2, _ = await tracker._new_buys(WALLET)  # same sig already seen
        assert buys2 == []

    @pytest.mark.asyncio
    async def test_failed_tx_skipped(self, tracker):
        sig = SimpleNamespace(signature="SIGERR", err={"e": 1})
        tracker.client.get_recent_signatures = AsyncMock(return_value=[sig])
        tracker.client.get_transaction = AsyncMock(return_value=_tx([], [_bal(WALLET, MINT, 10.0)]))
        assert await tracker._new_buys(WALLET) == ([], [])
        tracker.client.get_transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_sell_surfaced_via_on_sell(self, tracker):
        sig = SimpleNamespace(signature="SELL1", err=None)
        tracker.client.get_recent_signatures = AsyncMock(return_value=[sig])
        tracker.client.get_transaction = AsyncMock(
            return_value=_tx([_bal(WALLET, MINT, 100.0)], [_bal(WALLET, MINT, 0.0)])
        )
        sold: list = []

        async def on_sell(mint, wallet):
            sold.append((mint, wallet))

        tracker._on_sell = on_sell
        await tracker._poll_once(AsyncMock())
        assert sold == [(MINT, WALLET)]


class TestPollOnce:
    @pytest.mark.asyncio
    async def test_emits_candidate_for_buy(self, tracker):
        sig = SimpleNamespace(signature="S1", err=None)
        tracker.client.get_recent_signatures = AsyncMock(return_value=[sig])
        tracker.client.get_transaction = AsyncMock(return_value=_tx([], [_bal(WALLET, MINT, 10.0)]))
        tracker._curve_state = AsyncMock(return_value=_curve())  # type: ignore[method-assign]
        got: list = []

        async def cb(c):
            got.append(c)

        await tracker._poll_once(cb)
        assert [c["token_address"] for c in got] == [MINT]
        assert got[0]["source"] == "smart_money"
        assert got[0]["smart_money_tier"] == "B"
