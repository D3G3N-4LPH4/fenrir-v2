#!/usr/bin/env python3
"""
FENRIR - pump.fun creator-vault reader.

Reads a coin's creator fee vault straight off Solana RPC: the vault PDA
(``["creator-vault", creator]`` under the pump program) accrues the creator's
cut of trading fees until the creator sweeps it via ``collect_creator_fee``.

Empirical note (2026-10-04): every creator vault checked (SI, ANSEM, SAPLING,
Human) sat at exactly 650240 lamports (~0.00065 SOL) — creators sweep to the
rent minimum. So a *current* vault balance is almost always ~zero and is NOT
a useful live signal. The reader stays valuable as a primitive: an unswept
vault with a real balance is itself informative (creator asleep, or fees
accruing faster than sweeps), and the vault address derivation is needed for
any future fee-flow work.

Fail-open: returns None when the mint has no pump.fun bonding curve.
"""

from __future__ import annotations

import base64
import logging
import os

import aiohttp
from solders.pubkey import Pubkey

from fenrir.protocol.pumpfun import PumpFunProgram

logger = logging.getLogger(__name__)

# Below this the vault is just rent residue — the creator swept the fees.
SWEPT_THRESHOLD_LAMPORTS = 1_000_000  # 0.001 SOL


async def read_creator_vault(
    mint: str,
    rpc_url: str | None = None,
    timeout_seconds: float = 15.0,
) -> dict | None:
    """Return the creator-vault state for a pump.fun mint, or None.

    Result keys: ``mint``, ``creator``, ``vault``, ``lamports``, ``sol``,
    ``swept`` (True when the balance is rent residue — fees were collected).
    """
    program = PumpFunProgram()
    rpc = rpc_url or os.getenv("SOLANA_RPC_URL") or "https://api.mainnet-beta.solana.com"
    try:
        mint_pk = Pubkey.from_string(mint)
    except Exception:
        logger.debug("bad mint string %s", mint)
        return None

    async def _rpc(method: str, params: list) -> dict | None:
        try:
            # aiohttp, not httpx — httpx crashes on this sandbox's NO_PROXY
            # (see AGENTS.md).
            async with aiohttp.ClientSession(
                trust_env=True, timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as session:
                async with session.post(
                    rpc, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                ) as resp:
                    if resp.status != 200:
                        return None
                    payload = await resp.json()
                    return payload.get("result") if "error" not in payload else None
        except Exception as e:  # noqa: BLE001 - fail-open
            logger.debug("pump_vault RPC %s failed: %s", method, e)
            return None

    curve_pda, _ = program.derive_bonding_curve_address(mint_pk)
    acct = await _rpc(
        "getAccountInfo", [str(curve_pda), {"encoding": "base64", "commitment": "confirmed"}]
    )
    if not acct or not acct.get("value") or not acct["value"].get("data"):
        return None  # not a pump.fun coin (or curve closed)
    try:
        state = program.decode_bonding_curve(base64.b64decode(acct["value"]["data"][0]))
    except Exception as e:  # noqa: BLE001
        logger.debug("bonding curve decode failed for %s: %s", mint, e)
        return None
    if state is None or not state.creator:
        return None
    creator_pk = Pubkey.from_string(state.creator)
    vault = program.derive_creator_vault(creator_pk)
    vacc = await _rpc(
        "getAccountInfo", [str(vault), {"encoding": "base64", "commitment": "confirmed"}]
    )
    lamports = vacc["value"]["lamports"] if vacc and vacc.get("value") else 0
    return {
        "mint": mint,
        "creator": state.creator,
        "vault": str(vault),
        "lamports": lamports,
        "sol": lamports / 1e9,
        "swept": lamports <= SWEPT_THRESHOLD_LAMPORTS,
    }
