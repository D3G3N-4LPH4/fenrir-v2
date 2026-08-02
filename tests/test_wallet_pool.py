#!/usr/bin/env python3
"""
FENRIR - WalletPool tests (Phase 1.3)

Rotation, balance-gated selection, per-strategy funding, and the acceptance
criterion: concurrent position requests never get the same wallet (no
double-spend). No network.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.core.wallet import WalletPool


def _sim_pool(size: int = 5, base: float = 0.5, **kw) -> WalletPool:
    return WalletPool(
        private_keys=[], simulation_mode=True, pool_size=size, base_funding_sol=base, **kw
    )


class TestConstruction:
    def test_sim_fills_to_pool_size_with_throwaway_keys(self):
        pool = _sim_pool(size=5)
        assert pool.size == 5
        # Distinct keypairs.
        assert len({w.address for w in pool.wallets}) == 5
        assert pool.total_balance_sol() == pytest.approx(2.5)  # 5 x 0.5

    def test_primary_is_backward_compatible_single_wallet(self):
        pool = _sim_pool(size=1)
        assert pool.size == 1
        assert pool.primary.get_address() == pool.wallets[0].address

    def test_live_requires_a_key(self):
        with pytest.raises(ValueError, match="required for live"):
            WalletPool(private_keys=[], simulation_mode=False)


class TestSelection:
    @pytest.mark.asyncio
    async def test_rotates_across_wallets(self):
        pool = _sim_pool(size=3)
        seen = []
        for _ in range(3):
            w = await pool.acquire(0.1)
            assert w is not None
            seen.append(w.address)
            pool.release(w)
        assert len(set(seen)) == 3  # round-robin visits each

    @pytest.mark.asyncio
    async def test_skips_underfunded_wallets(self):
        pool = _sim_pool(size=2, base=0.5)
        # A trade larger than any wallet's balance (+buffer) can't be funded.
        assert await pool.acquire(1.0) is None
        # Within funding, it can.
        assert await pool.acquire(0.4) is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_all_in_flight(self):
        pool = _sim_pool(size=2)
        a = await pool.acquire(0.1)
        b = await pool.acquire(0.1)
        assert a is not None and b is not None
        assert await pool.acquire(0.1) is None  # both reserved
        pool.release(a)
        assert await pool.acquire(0.1) is a  # freed one reusable

    @pytest.mark.asyncio
    async def test_release_increments_trade_count(self):
        pool = _sim_pool(size=1)
        w = await pool.acquire(0.1)
        assert w is not None
        pool.release(w)
        assert w.trades == 1
        assert w.in_flight is False


class TestConcurrencyNoDoubleSpend:
    """Acceptance criterion: concurrent requests never share a wallet."""

    @pytest.mark.asyncio
    async def test_concurrent_acquires_are_all_distinct(self):
        pool = _sim_pool(size=5)
        # Fire 5 acquires concurrently; each must get a different wallet.
        results = await asyncio.gather(*(pool.acquire(0.1) for _ in range(5)))
        assert all(w is not None for w in results)
        addrs = [w.address for w in results]  # type: ignore[union-attr]
        assert len(set(addrs)) == 5  # no wallet handed out twice

    @pytest.mark.asyncio
    async def test_more_requests_than_wallets_some_get_none(self):
        pool = _sim_pool(size=3)
        results = await asyncio.gather(*(pool.acquire(0.1) for _ in range(6)))
        granted = [w for w in results if w is not None]
        assert len(granted) == 3  # exactly the pool size
        assert len({w.address for w in granted}) == 3  # all distinct


class TestFunding:
    def test_base_funding_when_no_override(self):
        pool = _sim_pool(base=0.5)
        assert pool.target_funding("sniper") == 0.5
        assert pool.target_funding(None) == 0.5

    def test_per_strategy_funding_variation(self):
        pool = _sim_pool(base=0.5, strategy_funding={"sniper": 0.75, "graduation": 0.25})
        assert pool.target_funding("sniper") == 0.75
        assert pool.target_funding("graduation") == 0.25
        assert pool.target_funding("other") == 0.5  # falls back to base


class TestBalanceRefresh:
    @pytest.mark.asyncio
    async def test_refresh_updates_live_balances(self):
        pool = WalletPool(private_keys=["k"], simulation_mode=True, pool_size=2)
        # Force it to act "live" for the refresh path.
        pool.simulation_mode = False
        client = AsyncMock()
        client.get_balance = AsyncMock(return_value=1.23)
        await pool.refresh_balances(client)
        assert all(w.balance_sol == pytest.approx(1.23) for w in pool.wallets)

    @pytest.mark.asyncio
    async def test_refresh_is_noop_in_simulation(self):
        pool = _sim_pool(size=2)
        client = AsyncMock()
        await pool.refresh_balances(client)
        client.get_balance.assert_not_called()


class TestStatusAndConfig:
    def test_status_never_exposes_keys(self):
        pool = _sim_pool(size=2)
        for row in pool.status():
            assert set(row) == {"address", "balance_sol", "in_flight", "trades"}

    def test_build_from_config(self, monkeypatch):
        for v in ("WALLET_PRIVATE_KEYS", "WALLET_POOL_SIZE", "WALLET_BASE_FUNDING_SOL"):
            monkeypatch.delenv(v, raising=False)
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        pool = cfg.build_wallet_pool()
        assert pool.size == 5
        assert pool.base_funding_sol == 0.5

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("WALLET_POOL_SIZE", "3")
        monkeypatch.setenv("WALLET_BASE_FUNDING_SOL", "0.75")
        monkeypatch.setenv("WALLET_STRATEGY_FUNDING", "sniper:1.0,swing:0.5")
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.wallet_pool_size == 3
        assert cfg.wallet_base_funding_sol == 0.75
        assert cfg.wallet_strategy_funding == {"sniper": 1.0, "swing": 0.5}
