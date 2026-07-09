#!/usr/bin/env python3
"""
FENRIR - Smart-Money / Whale Wallet Tracker

Follows a curated list of wallets (proven early buyers). Polls each wallet's
recent transactions and, when a tracked wallet BUYS a token, surfaces that token
as a candidate carrying a strong-signal AI context — you follow smart money into
a position early.

Buy detection is venue-agnostic: it compares the wallet's pre/post SPL-token
balances for the transaction (from tx meta), so it catches pump.fun bonding-curve
buys, AMM swaps, and aggregator/Jupiter routes alike — without decoding any
specific instruction or resolving address-lookup tables.

A per-candidate on-chain curve check tags the token so the engine routes it
correctly (fresh pump.fun curve buy vs. migrated/AMM Jupiter buy).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from solders.pubkey import Pubkey

from fenrir.config import BotConfig
from fenrir.logger import FenrirLogger

__all__ = ["SmartMoneyTracker"]

# Wrapped SOL — a buy wraps/unwraps SOL (net ~0), never the target token.
WSOL_MINT = "So11111111111111111111111111111111111111112"
_SEEN_CAP = 300  # bound the per-wallet seen-signature set


class SmartMoneyTracker:
    """Periodically follow tracked wallets and emit tokens they buy as candidates."""

    def __init__(self, config: BotConfig, client: Any, pumpfun: Any, logger: FenrirLogger):
        self.config = config
        self.client = client  # SolanaClient
        self.pumpfun = pumpfun  # PumpFunProtocol (for the routing curve check)
        self.logger = logger
        self.running = False
        self._seen: dict[str, set[str]] = {}  # wallet -> seen signatures
        self._cooldown: dict[str, datetime] = {}  # mint -> last emitted

    async def start_tracking(self, on_candidate: Callable[[dict], Awaitable[None]]) -> None:
        """Loop: poll tracked wallets on a cadence and emit their new buys."""
        self.running = True
        wallets = self.config.smart_money_wallets
        self.logger.info(
            f"Smart-money tracker active ({len(wallets)} wallet(s), "
            f"every {self.config.smart_money_poll_seconds:.0f}s)"
        )
        # Seed the seen-set with current history so we only fire on buys that
        # happen AFTER startup (not a backlog of old positions).
        await self._seed_seen()
        while self.running:
            try:
                await self._poll_once(on_candidate)
            except Exception as e:
                self.logger.error("Smart-money poll error", e)
            await asyncio.sleep(self.config.smart_money_poll_seconds)

    async def stop(self) -> None:
        self.running = False

    async def _seed_seen(self) -> None:
        for w in self.config.smart_money_wallets:
            try:
                sigs = await self.client.get_recent_signatures(Pubkey.from_string(w), limit=20)
                self._seen[w] = {str(s.signature) for s in sigs}
            except Exception:  # noqa: BLE001 - a bad address shouldn't kill startup
                self._seen[w] = set()

    async def _poll_once(self, on_candidate: Callable[[dict], Awaitable[None]]) -> None:
        now = datetime.now()
        emitted = 0
        cap = self.config.smart_money_max_candidates_per_cycle
        for wallet in self.config.smart_money_wallets:
            if emitted >= cap:
                break
            for mint in await self._new_buy_mints(wallet):
                if emitted >= cap:
                    break
                candidate = await self._build_candidate(wallet, mint, now)
                if candidate is None:
                    continue
                self._cooldown[mint] = now
                self.logger.info(f"Smart money {wallet[:6]}… bought {mint[:8]}… → candidate")
                await on_candidate(candidate)
                emitted += 1

    async def _new_buy_mints(self, wallet: str) -> list[str]:
        """Return mints the wallet newly BOUGHT since we last checked."""
        seen = self._seen.setdefault(wallet, set())
        try:
            sigs = await self.client.get_recent_signatures(Pubkey.from_string(wallet), limit=15)
        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"smart-money sigs failed for {wallet[:6]}…: {e}")
            return []

        # Oldest-first so we emit in buy order; skip already-seen and failed txs.
        fresh = [
            s
            for s in reversed(sigs)
            if str(s.signature) not in seen and getattr(s, "err", None) is None
        ]
        mints: list[str] = []
        for s in fresh:
            sig = str(s.signature)
            seen.add(sig)
            try:
                tx = await self.client.get_transaction(sig)
            except Exception:  # noqa: BLE001, S112 - one bad fetch shouldn't stop the poll
                continue
            if tx is None:
                continue
            mints.extend(self._detect_buys(tx, wallet))

        if len(seen) > _SEEN_CAP:  # keep the most recent signatures only
            self._seen[wallet] = set(list(seen)[-_SEEN_CAP:])
        # De-dup within this cycle, preserve order.
        return list(dict.fromkeys(mints))

    @staticmethod
    def _detect_buys(tx: Any, wallet: str) -> list[str]:
        """Mints whose balance INCREASED for `wallet` in this tx (a buy/receive).

        Uses the transaction's pre/post SPL-token balances (venue-agnostic). WSOL
        is excluded (wrapping SOL nets ~0 and is never the target token).
        """
        meta = getattr(getattr(tx, "transaction", None), "meta", None) or getattr(tx, "meta", None)
        if meta is None:
            return []

        def by_owner_mint(entries: Any) -> dict[tuple[str, str], float]:
            out: dict[tuple[str, str], float] = {}
            for b in entries or []:
                owner = str(getattr(b, "owner", "") or "")
                mint = str(getattr(b, "mint", "") or "")
                if not owner or not mint:
                    continue
                amt = getattr(getattr(b, "ui_token_amount", None), "ui_amount", None) or 0.0
                out[(owner, mint)] = float(amt)
            return out

        pre = by_owner_mint(getattr(meta, "pre_token_balances", None))
        post = by_owner_mint(getattr(meta, "post_token_balances", None))
        bought: list[str] = []
        for (owner, mint), amt in post.items():
            if owner != wallet or mint == WSOL_MINT:
                continue
            if amt > pre.get((owner, mint), 0.0) + 1e-9:
                bought.append(mint)
        return bought

    async def _build_candidate(self, wallet: str, mint: str, now: datetime) -> dict | None:
        """Apply cooldown, tag routing via an on-chain curve check, shape token_data."""
        last = self._cooldown.get(mint)
        if last and (now - last).total_seconds() < self.config.smart_money_cooldown_minutes * 60:
            return None

        migrated = await self._is_non_curve(mint)
        return {
            "token_address": mint,
            "symbol": "???",
            "name": "Unknown",
            "source": "smart_money",
            "smart_money_wallet": wallet,
            # Routing hint for the engine: migrated/AMM → Jupiter, else pump curve.
            "migrated": migrated,
            "tier": "mid" if migrated else None,
            "bonding_curve_state": None,
        }

    async def _is_non_curve(self, mint: str) -> bool:
        """True if the token has no live pump.fun bonding curve (migrated / AMM).

        Best-effort: on any error we assume a fresh curve (False) so the engine
        tries the pump path first — a wrong guess just fails routing, never spends.
        """
        try:
            bonding_curve, _ = self.pumpfun.derive_bonding_curve_address(Pubkey.from_string(mint))
            data = await self.client.get_account_info(bonding_curve)
            if not data:
                return True  # no curve account → migrated / not a pump token
            curve = self.pumpfun.decode_bonding_curve(data)
            return bool(curve is None or curve.complete)
        except Exception:  # noqa: BLE001
            return False
