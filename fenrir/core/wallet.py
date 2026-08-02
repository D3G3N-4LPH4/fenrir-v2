#!/usr/bin/env python3
"""
FENRIR - Wallet Management

Your keys, your crypto. Handle with the reverence they deserve.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import base58
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction


class WalletManager:
    """
    Your keys, your crypto. Handle with the reverence they deserve.
    Never logs private keys. Never stores them unencrypted.
    """

    def __init__(self, private_key_b58: str, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode

        if simulation_mode:
            # Generate a throwaway keypair for testing
            self.keypair = Keypair()
            self.pubkey = self.keypair.pubkey()
        else:
            if not private_key_b58:
                raise ValueError("Private key required for live trading")

            try:
                private_key_bytes = base58.b58decode(private_key_b58)
                self.keypair = Keypair.from_bytes(private_key_bytes)
                self.pubkey = self.keypair.pubkey()
            except Exception as e:
                raise ValueError(f"Invalid private key format: {e}") from e

    def get_address(self) -> str:
        """Return the wallet's public address."""
        return str(self.pubkey)

    def sign_transaction(self, transaction: Transaction) -> Transaction:
        """Sign with the elegance of a digital signature."""
        if self.simulation_mode:
            return transaction  # Don't actually sign in sim
        transaction.sign([self.keypair], transaction.message.recent_blockhash)
        return transaction


# ── Multi-wallet pool ──────────────────────────────────────────────────

# Keep a little SOL beyond the trade for fees/rent so a wallet selected for a
# 0.5 SOL trade isn't left unable to pay its own transaction.
_FEE_BUFFER_SOL = 0.01


@dataclass
class PooledWallet:
    """One wallet in the rotation pool, with its monitored balance + in-flight lock."""

    manager: WalletManager
    balance_sol: float = 0.0
    in_flight: bool = False
    trades: int = 0

    @property
    def pubkey(self) -> Pubkey:
        return self.manager.pubkey

    @property
    def address(self) -> str:
        return self.manager.get_address()


class WalletPool:
    """A rotating pool of funded wallets.

    Position sizing is otherwise capped by a single wallet's per-wallet buy limits
    and its balance; a pool spreads concurrent positions across wallets and lets the
    audit chain attribute each trade to the wallet that executed it.

    Selection is concurrency-safe: ``acquire`` holds a lock while it picks a free
    wallet with enough balance and marks it in-flight, so two simultaneous position
    requests can never be handed the same wallet (no double-spend). ``release``
    frees it after the trade settles.

    Backward compatible: a single-key pool behaves like one WalletManager, exposed
    via :attr:`primary`.
    """

    def __init__(
        self,
        private_keys: list[str],
        simulation_mode: bool = True,
        pool_size: int = 5,
        base_funding_sol: float = 0.5,
        strategy_funding: dict[str, float] | None = None,
        logger: Any = None,
    ) -> None:
        self.simulation_mode = simulation_mode
        self.base_funding_sol = base_funding_sol
        self._strategy_funding = dict(strategy_funding or {})
        self._logger = logger
        self._lock = asyncio.Lock()
        self._cursor = 0  # round-robin start, for fair distribution

        keys = [k for k in private_keys if k]
        managers: list[WalletManager] = []
        if simulation_mode:
            # Throwaway keypairs; fill to pool_size so rotation is exercisable with
            # no real funds (simulation must work for every component).
            for _ in range(max(1, pool_size)):
                managers.append(WalletManager("", simulation_mode=True))
        else:
            if not keys:
                raise ValueError("At least one WALLET_PRIVATE_KEY(S) required for live trading")
            for k in keys[: max(1, pool_size)]:
                managers.append(WalletManager(k, simulation_mode=False))

        # In simulation, seed each wallet with its base funding so selection logic
        # (which gates on balance) has something to work with.
        seed = base_funding_sol if simulation_mode else 0.0
        self.wallets: list[PooledWallet] = [
            PooledWallet(manager=m, balance_sol=seed) for m in managers
        ]

    @property
    def primary(self) -> WalletManager:
        """First wallet — the single-wallet backward-compatible view."""
        return self.wallets[0].manager

    @property
    def size(self) -> int:
        return len(self.wallets)

    def target_funding(self, strategy_id: str | None) -> float:
        """Desired funding for a strategy's wallets — base, varied per strategy."""
        if strategy_id and strategy_id in self._strategy_funding:
            return self._strategy_funding[strategy_id]
        return self.base_funding_sol

    async def acquire(
        self, amount_sol: float, strategy_id: str | None = None
    ) -> PooledWallet | None:
        """Reserve a free wallet that can fund ``amount_sol`` (+ fee buffer).

        Round-robins across eligible wallets for even wear. Returns None when every
        wallet is either in-flight or under-funded — the caller should skip/queue
        rather than force a trade the pool can't back.
        """
        needed = amount_sol + _FEE_BUFFER_SOL
        async with self._lock:
            n = len(self.wallets)
            for i in range(n):
                w = self.wallets[(self._cursor + i) % n]
                if not w.in_flight and w.balance_sol >= needed:
                    w.in_flight = True
                    self._cursor = (self._cursor + i + 1) % n
                    return w
            return None

    def release(self, wallet: PooledWallet) -> None:
        """Free a wallet after its trade settles."""
        wallet.in_flight = False
        wallet.trades += 1

    async def refresh_balances(self, client: Any) -> None:
        """Update each wallet's SOL balance from chain (skipped in simulation)."""
        if self.simulation_mode:
            return
        for w in self.wallets:
            try:
                w.balance_sol = await client.get_balance(w.pubkey)
            except Exception as e:  # noqa: BLE001 - a balance hiccup must not wedge selection
                if self._logger is not None:
                    self._logger.warning(f"Balance refresh failed for {w.address[:8]}...: {e}")

    def total_balance_sol(self) -> float:
        return sum(w.balance_sol for w in self.wallets)

    def status(self) -> list[dict[str, Any]]:
        """Per-wallet snapshot for monitoring / audit (never exposes keys)."""
        return [
            {
                "address": w.address,
                "balance_sol": round(w.balance_sol, 6),
                "in_flight": w.in_flight,
                "trades": w.trades,
            }
            for w in self.wallets
        ]
