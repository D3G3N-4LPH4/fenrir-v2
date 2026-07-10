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
        self._on_sell: Callable[[str, str], Awaitable[None]] | None = None

    def _tracked_wallets(self) -> list[str]:
        """All wallets we follow: standard + A-tier priority, de-duped, order-stable."""
        return list(
            dict.fromkeys(
                self.config.smart_money_wallets + self.config.smart_money_priority_wallets
            )
        )

    def _tier(self, wallet: str) -> str:
        """'A' for a priority wallet (higher conviction), else 'B'."""
        return "A" if wallet in set(self.config.smart_money_priority_wallets) else "B"

    async def start_tracking(
        self,
        on_candidate: Callable[[dict], Awaitable[None]],
        on_sell: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        """Loop: poll tracked wallets and emit their new buys (candidates) + sells.

        on_candidate(token_data) fires when a tracked wallet BUYS; on_sell(mint,
        wallet) fires when one SELLS/reduces a position (the bot decides whether we
        hold it and route it to the AI exit evaluator).
        """
        self.running = True
        self._on_sell = on_sell
        wallets = self._tracked_wallets()
        self.logger.info(
            f"Smart-money tracker active ({len(wallets)} wallet(s), "
            f"every {self.config.smart_money_poll_seconds:.0f}s)"
        )
        # Seed the seen-set with current history so we only fire on buys/sells that
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
        for w in self._tracked_wallets():
            try:
                sigs = await self.client.get_recent_signatures(Pubkey.from_string(w), limit=20)
                self._seen[w] = {str(s.signature) for s in sigs}
            except Exception:  # noqa: BLE001 - a bad address shouldn't kill startup
                self._seen[w] = set()

    async def _poll_once(self, on_candidate: Callable[[dict], Awaitable[None]]) -> None:
        now = datetime.now()
        emitted = 0
        cap = self.config.smart_money_max_candidates_per_cycle
        for wallet in self._tracked_wallets():
            buys, sells = await self._new_buys(wallet)
            # Sells first: surface exit signals for tokens we may hold (bot decides).
            if self._on_sell is not None:
                for mint in sells:
                    await self._on_sell(mint, wallet)
            for mint, sol_spent in buys:
                if emitted >= cap:
                    break
                candidate = await self._build_candidate(wallet, mint, sol_spent, now)
                if candidate is None:
                    continue
                self._cooldown[mint] = now
                self.logger.info(
                    f"Smart money [{self._tier(wallet)}] {wallet[:6]}… bought "
                    f"{mint[:8]}… (~{sol_spent:.2f} SOL) → candidate"
                )
                await on_candidate(candidate)
                emitted += 1

    async def _new_buys(self, wallet: str) -> tuple[list[tuple[str, float]], list[str]]:
        """Scan a wallet's new txs → (buys[(mint, sol_spent)], sells[mint])."""
        seen = self._seen.setdefault(wallet, set())
        try:
            sigs = await self.client.get_recent_signatures(Pubkey.from_string(wallet), limit=15)
        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"smart-money sigs failed for {wallet[:6]}…: {e}")
            return [], []

        # Oldest-first so we emit in buy order; skip already-seen and failed txs.
        fresh = [
            s
            for s in reversed(sigs)
            if str(s.signature) not in seen and getattr(s, "err", None) is None
        ]
        buys: list[tuple[str, float]] = []
        sells: list[str] = []
        for s in fresh:
            sig = str(s.signature)
            seen.add(sig)
            try:
                tx = await self.client.get_transaction(sig)
            except Exception:  # noqa: BLE001, S112 - one bad fetch shouldn't stop the poll
                continue
            if tx is None:
                continue
            buys.extend(self._detect_buys(tx, wallet))
            sells.extend(self._detect_sells(tx, wallet))

        if len(seen) > _SEEN_CAP:  # keep the most recent signatures only
            self._seen[wallet] = set(list(seen)[-_SEEN_CAP:])
        # De-dup by mint within this cycle.
        dedup_buys: dict[str, float] = {}
        for mint, sol in buys:
            dedup_buys.setdefault(mint, sol)
        return list(dedup_buys.items()), list(dict.fromkeys(sells))

    @staticmethod
    def _balances(entries: Any) -> dict[tuple[str, str], float]:
        """{(owner, mint): ui_amount} from a tx's token-balance entries."""
        out: dict[tuple[str, str], float] = {}
        for b in entries or []:
            owner = str(getattr(b, "owner", "") or "")
            mint = str(getattr(b, "mint", "") or "")
            if not owner or not mint:
                continue
            amt = getattr(getattr(b, "ui_token_amount", None), "ui_amount", None) or 0.0
            out[(owner, mint)] = float(amt)
        return out

    @staticmethod
    def _detect_buys(tx: Any, wallet: str) -> list[tuple[str, float]]:
        """(mint, sol_spent) for mints whose balance INCREASED for `wallet` in this tx.

        Uses the transaction's pre/post SPL-token balances (venue-agnostic). WSOL
        is excluded (wrapping SOL nets ~0 and is never the target token). sol_spent
        is the wallet's native lamport decrease (best-effort; 0.0 if unresolvable).
        """
        meta = getattr(getattr(tx, "transaction", None), "meta", None) or getattr(tx, "meta", None)
        if meta is None:
            return []
        pre = SmartMoneyTracker._balances(getattr(meta, "pre_token_balances", None))
        post = SmartMoneyTracker._balances(getattr(meta, "post_token_balances", None))
        bought = [
            mint
            for (owner, mint), amt in post.items()
            if owner == wallet and mint != WSOL_MINT and amt > pre.get((owner, mint), 0.0) + 1e-9
        ]
        if not bought:
            return []
        sol_spent = SmartMoneyTracker._sol_spent(tx, wallet, meta)
        return [(mint, sol_spent) for mint in bought]

    @staticmethod
    def _detect_sells(tx: Any, wallet: str) -> list[str]:
        """Mints the wallet SOLD/reduced in this tx (token balance decreased).

        A tracked wallet trimming or dumping a position is an exit signal for any
        token we hold. WSOL excluded. Requires a prior balance (pre > 0).
        """
        meta = getattr(getattr(tx, "transaction", None), "meta", None) or getattr(tx, "meta", None)
        if meta is None:
            return []
        pre = SmartMoneyTracker._balances(getattr(meta, "pre_token_balances", None))
        post = SmartMoneyTracker._balances(getattr(meta, "post_token_balances", None))
        return [
            mint
            for (owner, mint), amt in pre.items()
            if owner == wallet
            and mint != WSOL_MINT
            and amt > 1e-9
            and post.get((owner, mint), 0.0) < amt - 1e-9
        ]

    @staticmethod
    def _sol_spent(tx: Any, wallet: str, meta: Any) -> float:
        """Best-effort native SOL the wallet spent in this tx (lamport decrease)."""
        try:
            msg = tx.transaction.transaction.message
            keys = [str(k) for k in msg.account_keys]
            idx = keys.index(wallet)
            pre = int(meta.pre_balances[idx])
            post = int(meta.post_balances[idx])
            return max(0.0, (pre - post) / 1e9)
        except Exception:  # noqa: BLE001 - context flavor only; unknown → 0.0
            return 0.0

    async def _build_candidate(
        self, wallet: str, mint: str, sol_spent: float, now: datetime
    ) -> dict | None:
        """Apply cooldown, then enrich the candidate with real on-chain token data.

        A bare candidate (unknown symbol / zero liquidity) makes the AI reject on
        artifact red flags, drowning out the smart-money signal. So we read the
        pump.fun bonding curve and populate the real price / liquidity / market cap
        (and bonding_curve_state, which the AI metadata builder consumes) — or tag
        it migrated for the Jupiter route when there's no live curve.
        """
        last = self._cooldown.get(mint)
        if last and (now - last).total_seconds() < self.config.smart_money_cooldown_minutes * 60:
            return None

        curve = await self._curve_state(mint)
        migrated = curve is None
        candidate: dict = {
            "token_address": mint,
            "symbol": "???",
            "name": "Unknown",
            "source": "smart_money",
            "smart_money_wallet": wallet,
            "smart_money_tier": self._tier(wallet),
            "smart_money_sol": round(sol_spent, 4),
            # Routing hint for the engine: migrated/AMM → Jupiter, else pump curve.
            "migrated": migrated,
            "tier": "mid" if migrated else None,
            "bonding_curve_state": curve,
        }
        if curve is not None:
            # Real numbers from the curve so the AI evaluates the actual token
            # instead of "unknown with zero liquidity".
            candidate["initial_liquidity_sol"] = round(curve.real_sol_reserves / 1e9, 4)
            candidate["market_cap_sol"] = round(curve.get_market_cap_sol(), 4)
        return candidate

    async def _curve_state(self, mint: str) -> Any:
        """Return the token's live pump.fun BondingCurveState, or None if migrated.

        Best-effort: on any error return None → the engine's Jupiter route is tried,
        which fails safe (never spends on a bad guess).
        """
        try:
            bonding_curve, _ = self.pumpfun.derive_bonding_curve_address(Pubkey.from_string(mint))
            data = await self.client.get_account_info(bonding_curve)
            if not data:
                return None  # no curve account → migrated / not a pump token
            curve = self.pumpfun.decode_bonding_curve(data)
            if curve is None or curve.complete:
                return None  # decode failed or migrated
            return curve
        except Exception:  # noqa: BLE001
            return None
