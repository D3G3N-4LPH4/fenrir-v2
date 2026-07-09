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


class TestBuildCandidate:
    @pytest.mark.asyncio
    async def test_candidate_carries_tier_and_sol(self, tracker):
        tracker._is_non_curve = AsyncMock(return_value=False)  # type: ignore[method-assign]
        c = await tracker._build_candidate(A_WALLET, MINT, 2.5, T0)
        assert c is not None
        assert c["source"] == "smart_money"
        assert c["smart_money_wallet"] == A_WALLET
        assert c["smart_money_tier"] == "A"
        assert c["smart_money_sol"] == 2.5
        assert c["migrated"] is False
        assert c["tier"] is None

    @pytest.mark.asyncio
    async def test_migrated_candidate(self, tracker):
        tracker._is_non_curve = AsyncMock(return_value=True)  # type: ignore[method-assign]
        c = await tracker._build_candidate(WALLET, MINT, 0.0, T0)
        assert c is not None
        assert c["smart_money_tier"] == "B"
        assert c["migrated"] is True
        assert c["tier"] == "mid"

    @pytest.mark.asyncio
    async def test_cooldown_blocks_then_allows(self, tracker):
        tracker._is_non_curve = AsyncMock(return_value=False)  # type: ignore[method-assign]
        tracker._cooldown[MINT] = T0
        assert await tracker._build_candidate(WALLET, MINT, 0.0, T0 + timedelta(minutes=5)) is None
        assert (
            await tracker._build_candidate(WALLET, MINT, 0.0, T0 + timedelta(minutes=61))
            is not None
        )


class TestIsNonCurve:
    @pytest.mark.asyncio
    async def test_no_account_is_migrated(self, tracker):
        tracker.pumpfun.derive_bonding_curve_address.return_value = (MagicMock(), 0)
        tracker.client.get_account_info = AsyncMock(return_value=None)
        assert await tracker._is_non_curve(MINT) is True

    @pytest.mark.asyncio
    async def test_complete_curve_is_migrated(self, tracker):
        tracker.pumpfun.derive_bonding_curve_address.return_value = (MagicMock(), 0)
        tracker.client.get_account_info = AsyncMock(return_value=b"x" * 80)
        tracker.pumpfun.decode_bonding_curve.return_value = SimpleNamespace(complete=True)
        assert await tracker._is_non_curve(MINT) is True

    @pytest.mark.asyncio
    async def test_live_curve_is_fresh(self, tracker):
        tracker.pumpfun.derive_bonding_curve_address.return_value = (MagicMock(), 0)
        tracker.client.get_account_info = AsyncMock(return_value=b"x" * 80)
        tracker.pumpfun.decode_bonding_curve.return_value = SimpleNamespace(complete=False)
        assert await tracker._is_non_curve(MINT) is False


class TestNewBuys:
    @pytest.mark.asyncio
    async def test_dedup_seen_signatures(self, tracker):
        sig = SimpleNamespace(signature="SIG1", err=None)
        tracker.client.get_recent_signatures = AsyncMock(return_value=[sig])
        tracker.client.get_transaction = AsyncMock(return_value=_tx([], [_bal(WALLET, MINT, 10.0)]))

        first = await tracker._new_buys(WALLET)
        assert first == [(MINT, 0.0)]
        second = await tracker._new_buys(WALLET)  # same sig already seen
        assert second == []

    @pytest.mark.asyncio
    async def test_failed_tx_skipped(self, tracker):
        sig = SimpleNamespace(signature="SIGERR", err={"e": 1})
        tracker.client.get_recent_signatures = AsyncMock(return_value=[sig])
        tracker.client.get_transaction = AsyncMock(return_value=_tx([], [_bal(WALLET, MINT, 10.0)]))
        assert await tracker._new_buys(WALLET) == []
        tracker.client.get_transaction.assert_not_called()


class TestPollOnce:
    @pytest.mark.asyncio
    async def test_emits_candidate_for_buy(self, tracker):
        sig = SimpleNamespace(signature="S1", err=None)
        tracker.client.get_recent_signatures = AsyncMock(return_value=[sig])
        tracker.client.get_transaction = AsyncMock(return_value=_tx([], [_bal(WALLET, MINT, 10.0)]))
        tracker._is_non_curve = AsyncMock(return_value=False)  # type: ignore[method-assign]
        got: list = []

        async def cb(c):
            got.append(c)

        await tracker._poll_once(cb)
        assert [c["token_address"] for c in got] == [MINT]
        assert got[0]["source"] == "smart_money"
        assert got[0]["smart_money_tier"] == "B"
