#!/usr/bin/env python3
"""
FENRIR - pump.fun bonding-curve provider for discovery.

Reads live bonding-curve state straight off Solana RPC so the scout can see
where a pump.fun token sits on its curve *before* DexScreener momentum shows
up. This is the data behind the ``graduation_watch`` entry filter: tokens at
50-85% of the curve with fresh SOL inflow are approaching Raydium graduation,
which is exactly the "catch it lower" window the DexScreener-only filters miss.

Curve math (decoding, PDA derivation) lives in :mod:`fenrir.protocol.pumpfun`;
this module only handles RPC transport, batching, and the velocity state file.

State file (JSON, one entry per mint)::
    {"<mint>": {"first_seen": <epoch>, "last_check": <epoch>,
                "last_progress": 62.1, "last_sol": 52.7,
                "prev_check": <epoch>, "prev_sol": 51.9}}

``record_reading`` shifts last_* -> prev_* before writing the new reading, so
``inflow_sol`` (SOL into the curve between the two most recent checks) is
always computable without a second RPC round-trip.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time

import aiohttp

from fenrir.protocol.pumpfun import (
    MIGRATION_THRESHOLD_SOL,
    BondingCurveState,
    PumpFunProgram,
)

logger = logging.getLogger(__name__)

# Default location of the velocity state file. Overridable via env for tests.
DEFAULT_STATE_PATH = os.path.expanduser(
    os.getenv(
        "PUMPFUN_STATE_PATH",
        "~/workspace/goals/token-scout-watch/hidden_files/pumpfun_curves.json",
    )
)

# getMultipleAccounts cap per JSON-RPC call.
_RPC_BATCH_SIZE = 100
# A previous reading older than this is too stale to compute inflow from.
_MAX_INFLOW_GAP_SECONDS = 30 * 60


class PumpFunProvider:
    """Live pump.fun bonding-curve reads over Solana JSON-RPC."""

    def __init__(self, rpc_url: str | None = None, timeout_seconds: float = 15.0) -> None:
        self.rpc_url = (
            rpc_url or os.getenv("SOLANA_RPC_URL") or "https://api.mainnet-beta.solana.com"
        )
        self.timeout = timeout_seconds
        self.program = PumpFunProgram()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # trust_env=True: the sandbox proxies outbound traffic.
            # NOTE: use aiohttp, not httpx — httpx crashes on the bracketed
            # IPv6 literals in this sandbox's NO_PROXY (see AGENTS.md).
            self._session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "FENRIR/2.0 discovery"},
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _rpc(self, method: str, params: list) -> dict | None:
        """One raw JSON-RPC call. Fail-open: None on any error."""
        try:
            session = await self._get_session()
            async with session.post(
                self.rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            ) as resp:
                if resp.status != 200:
                    logger.debug("pumpfun RPC %s -> HTTP %s", method, resp.status)
                    return None
                payload = await resp.json()
            if "error" in payload:
                logger.debug("pumpfun RPC %s error: %s", method, payload["error"])
                return None
            result: dict | None = payload.get("result")
            return result
        except Exception as e:  # noqa: BLE001 - discovery is fail-open
            logger.debug("pumpfun RPC %s failed: %s", method, e)
            return None

    def _decode_account(self, value: dict | None) -> BondingCurveState | None:
        if not value or not value.get("data"):
            return None
        try:
            raw = base64.b64decode(value["data"][0])
        except Exception:  # noqa: BLE001
            return None
        return self.program.decode_bonding_curve(raw)

    async def curve_state(self, mint: str) -> BondingCurveState | None:
        """Fetch one token's bonding-curve state (None when not a pump.fun token)."""
        states = await self.curve_states([mint])
        return states.get(mint)

    async def curve_states(self, mints: list[str]) -> dict[str, BondingCurveState]:
        """Batch-fetch curve states. Mints without a curve are simply absent."""
        out: dict[str, BondingCurveState] = {}
        if not mints:
            return out
        from solders.pubkey import Pubkey

        pairs: list[tuple[str, str]] = []  # (mint, curve_pda)
        for mint in mints:
            try:
                pda, _ = self.program.derive_bonding_curve_address(Pubkey.from_string(mint))
                pairs.append((mint, str(pda)))
            except Exception:  # noqa: BLE001, S112 - bad mint string, skip
                continue
        for i in range(0, len(pairs), _RPC_BATCH_SIZE):
            chunk = pairs[i : i + _RPC_BATCH_SIZE]
            addrs = [pda for _, pda in chunk]
            result = await self._rpc(
                "getMultipleAccounts",
                [addrs, {"encoding": "base64", "commitment": "confirmed"}],
            )
            if not result or not isinstance(result.get("value"), list):
                continue
            for (mint, _), value in zip(chunk, result["value"], strict=False):
                state = self._decode_account(value)
                if state is not None:
                    out[mint] = state
        return out

    # ── Velocity state file ────────────────────────────────────────────

    @staticmethod
    def _load_state(path: str = DEFAULT_STATE_PATH) -> dict:
        try:
            with open(path) as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _save_state(state: dict, path: str = DEFAULT_STATE_PATH) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, path)
        except OSError as e:  # noqa: BLE001 - state is best-effort
            logger.debug("pumpfun state save failed: %s", e)

    def record_reading(
        self,
        mint: str,
        state: BondingCurveState,
        now: float | None = None,
        path: str = DEFAULT_STATE_PATH,
    ) -> dict:
        """Record a curve reading; returns the entry with prev_* shifted."""
        now = now if now is not None else time.time()
        store = self._load_state(path)
        prev = store.get(mint, {})
        entry = {
            "first_seen": prev.get("first_seen", now),
            "last_check": now,
            "last_progress": round(state.get_migration_progress(), 2),
            "last_sol": round(state.real_sol_reserves / 1e9, 4),
            "complete": bool(state.complete),
            "prev_check": prev.get("last_check"),
            "prev_sol": prev.get("last_sol"),
        }
        store[mint] = entry
        self._save_state(store, path)
        return entry

    @staticmethod
    def inflow_sol(entry: dict, now: float | None = None) -> float | None:
        """SOL into the curve since the previous reading, or None when unknown."""
        now = now if now is not None else time.time()
        prev_check = entry.get("prev_check")
        prev_sol = entry.get("prev_sol")
        last_sol = entry.get("last_sol")
        if prev_check is None or prev_sol is None or last_sol is None:
            return None
        if now - prev_check > _MAX_INFLOW_GAP_SECONDS:
            return None
        inflow: float = round(last_sol - prev_sol, 4)
        return inflow

    @staticmethod
    def prune_state(
        max_age_hours: float = 6.0,
        now: float | None = None,
        path: str = DEFAULT_STATE_PATH,
    ) -> int:
        """Drop entries untouched for a while. Returns number pruned."""
        now = now if now is not None else time.time()
        store = PumpFunProvider._load_state(path)
        cutoff = now - max_age_hours * 3600
        pruned = [m for m, e in store.items() if e.get("last_check", 0) < cutoff]
        for m in pruned:
            del store[m]
        if pruned:
            PumpFunProvider._save_state(store, path)
        return len(pruned)

    @staticmethod
    def sol_to_graduation(state: BondingCurveState) -> float:
        """SOL still needed before the 85 SOL migration trigger."""
        return max(0.0, (MIGRATION_THRESHOLD_SOL - state.real_sol_reserves) / 1e9)


async def annotate_bond_curve(
    snap, provider: PumpFunProvider | None = None, path: str = DEFAULT_STATE_PATH
) -> bool:
    """Populate a Solana snapshot's bond fields from RPC + the state file.

    Reuses this run's source-sweep reading when fresh (<120s, no extra RPC);
    otherwise does one getAccountInfo and records the reading for velocity.
    Returns True when the token has a pump.fun curve.
    """
    own = provider is None
    provider = provider or PumpFunProvider()
    try:
        now = time.time()
        store = PumpFunProvider._load_state(path)
        entry = store.get(snap.token_address)
        if entry and now - entry.get("last_check", 0) < 120:
            snap.bond_progress_pct = entry.get("last_progress")
            snap.bond_inflow_sol = PumpFunProvider.inflow_sol(entry, now)
            last_sol = entry.get("last_sol")
            snap.bond_sol_remaining = (
                round(max(0.0, 85.0 - last_sol), 2) if last_sol is not None else None
            )
            snap.migrated = bool(entry.get("complete"))
            return True
        state = await provider.curve_state(snap.token_address)
        if state is None:
            return False
        entry = provider.record_reading(snap.token_address, state, now, path)
        snap.bond_progress_pct = round(state.get_migration_progress(), 2)
        snap.bond_inflow_sol = PumpFunProvider.inflow_sol(entry, now)
        snap.bond_sol_remaining = round(PumpFunProvider.sol_to_graduation(state), 2)
        snap.migrated = bool(state.complete)
        return True
    finally:
        if own:
            await provider.close()
