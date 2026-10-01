"""Platform-vault LP detection for Solana pools.

Some launchpads don't burn LP or use a recognised locker at graduation.
StonkFun (via Raydium LaunchLab) keeps 100% of the graduated pool's LP tokens
in a platform vault wallet instead of passing them to the creator. RugCheck-style
heuristics then report 0% locked, which reads as a live rug risk even though the
token creator cannot touch the LP.

Heuristic: if a single holder owns ~all of the LP mint AND that holder's wallet
controls a large number of token accounts (>= ``VAULT_MIN_TOKEN_ACCOUNTS``), it
is platform infrastructure, not the dev. Identified vaults are cached by address
so later tokens skip the expensive account scan.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)

RAYDIUM_POOL_INFO = "https://api-v3.raydium.io/pools/info/ids?ids={pool}"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"  # noqa: S105 - public Solana program ID, not a secret

# Share of LP supply in one holder that counts as "concentrated".
VAULT_MIN_SHARE = 0.95
# Token-account count above which a holder is treated as platform infrastructure.
VAULT_MIN_TOKEN_ACCOUNTS = 500

DEFAULT_CACHE_PATH = Path(
    os.environ.get(
        "LP_VAULT_CACHE",
        str(Path.home() / ".cache" / "fenrir" / "lp_vaults.json"),
    )
)

# Per-pool check results (positive AND negative), so a token re-scanned on the
# next 10-minute tick doesn't redo the RPC walk. LP custody rarely changes.
DEFAULT_CHECK_CACHE_PATH = Path(
    os.environ.get(
        "LP_VAULT_CHECK_CACHE",
        str(Path.home() / ".cache" / "fenrir" / "lp_vault_checks.json"),
    )
)
CHECK_CACHE_TTL_SECONDS = 24 * 3600


@dataclass
class VaultCheck:
    is_platform_vault: bool
    holder: str | None = None
    holder_share_pct: float | None = None
    token_account_count: int | None = None
    cached: bool = False


async def resolve_lp_mint(pool_address: str, timeout_seconds: float = 10.0) -> str | None:
    """Return the LP mint for a Raydium pool address, or None."""
    try:
        async with aiohttp.ClientSession(trust_env=True) as s:
            async with s.get(
                RAYDIUM_POOL_INFO.format(pool=pool_address),
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("raydium pool info failed for %s: %s", pool_address, e)
        return None
    pools = data.get("data") or []
    if not pools:
        return None
    return (pools[0].get("lpMint") or {}).get("address")


async def _rpc(
    session: aiohttp.ClientSession,
    rpc_url: str,
    method: str,
    params: list,
    timeout_seconds: float,
):
    async with session.post(
        rpc_url,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=aiohttp.ClientTimeout(total=timeout_seconds),
    ) as r:
        r.raise_for_status()
        body = await r.json()
    if body.get("error"):
        raise RuntimeError(body["error"])
    return body["result"]


async def check_lp_platform_vault(
    lp_mint: str,
    rpc_url: str,
    known_vaults: set[str] | None = None,
    min_token_accounts: int = VAULT_MIN_TOKEN_ACCOUNTS,
    timeout_seconds: float = 12.0,
) -> VaultCheck:
    """Check whether an LP mint is held by a platform vault wallet.

    ``timeout_seconds`` applies per RPC call (previously it was accepted but
    silently ignored while every call used a hardcoded 30s).
    """
    known_vaults = known_vaults or set()
    try:
        async with aiohttp.ClientSession(trust_env=True) as s:
            # Independent reads go out together, not sequentially.
            largest, supply = await asyncio.gather(
                _rpc(s, rpc_url, "getTokenLargestAccounts", [lp_mint], timeout_seconds),
                _rpc(s, rpc_url, "getTokenSupply", [lp_mint], timeout_seconds),
            )
            largest = largest["value"]
            if not largest:
                return VaultCheck(False)
            top = largest[0]
            top_addr = top["address"]
            top_amt = float(top.get("amount") or 0)
            if top_addr in known_vaults:
                return VaultCheck(True, holder=top_addr, cached=True)
            supply = float(supply["value"]["amount"])
            share = (top_amt / supply) if supply else 0.0
            if share < VAULT_MIN_SHARE:
                return VaultCheck(False, holder=top_addr, holder_share_pct=share * 100)
            acct = await _rpc(
                s,
                rpc_url,
                "getAccountInfo",
                [top_addr, {"encoding": "jsonParsed"}],
                timeout_seconds,
            )
            owner = acct["value"]["data"]["parsed"]["info"]["owner"]
            if owner in known_vaults:
                return VaultCheck(True, holder=owner, holder_share_pct=share * 100, cached=True)
            # Count only — dataSlice keeps the payload tiny (a platform vault
            # can hold 20k+ token accounts; we need the count, not the data).
            accounts = (
                await _rpc(
                    s,
                    rpc_url,
                    "getTokenAccountsByOwner",
                    [
                        owner,
                        {"programId": TOKEN_PROGRAM},
                        {"encoding": "base64", "dataSlice": {"offset": 0, "length": 0}},
                    ],
                    timeout_seconds,
                )
            )["value"]
            count = len(accounts)
            is_vault = count >= min_token_accounts
            return VaultCheck(
                is_vault, holder=owner, holder_share_pct=share * 100, token_account_count=count
            )
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("lp vault check failed for %s: %s", lp_mint, e)
        return VaultCheck(False)


def load_known_vaults(path: Path | None = None) -> set[str]:
    path = path if path is not None else DEFAULT_CACHE_PATH
    try:
        return set(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001 - missing/corrupt cache is fine
        return set()


def save_known_vaults(vaults: set[str], path: Path | None = None) -> None:
    path = path if path is not None else DEFAULT_CACHE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(vaults)))
    except Exception as e:  # noqa: BLE001 - cache is best-effort
        log.debug("could not save lp vault cache: %s", e)


def _load_check_cache(path: Path | None = None) -> dict:
    path = path if path is not None else DEFAULT_CHECK_CACHE_PATH
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - missing/corrupt cache is fine
        return {}


def _save_check_cache(store: dict, path: Path | None = None) -> None:
    path = path if path is not None else DEFAULT_CHECK_CACHE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store))
    except Exception as e:  # noqa: BLE001 - cache is best-effort
        log.debug("could not save lp vault check cache: %s", e)


async def check_pool_lp_vault(
    pool_address: str,
    rpc_url: str,
    min_token_accounts: int = VAULT_MIN_TOKEN_ACCOUNTS,
    timeout_seconds: float = 12.0,
) -> VaultCheck:
    """Resolve a pool's LP mint and run the platform-vault check, cached.

    Per-pool results (vault or not) are cached for 24h so a token re-scanned
    on the next 10-minute tick doesn't redo the Raydium + RPC walk.
    """
    store = _load_check_cache()
    hit = store.get(pool_address)
    if (
        isinstance(hit, dict)
        and time.time() - float(hit.get("checked_at", 0)) < CHECK_CACHE_TTL_SECONDS
    ):
        return VaultCheck(
            is_platform_vault=bool(hit.get("is_vault")),
            holder=hit.get("holder"),
            holder_share_pct=hit.get("holder_share_pct"),
            token_account_count=hit.get("token_account_count"),
            cached=True,
        )
    lp_mint = await resolve_lp_mint(pool_address)
    if not lp_mint:
        # Cache the miss too: a pre-graduation pool has no Raydium LP mint yet.
        store[pool_address] = {"is_vault": False, "lp_mint": None, "checked_at": time.time()}
        _save_check_cache(store)
        return VaultCheck(False)
    known = load_known_vaults()
    check = await check_lp_platform_vault(
        lp_mint, rpc_url, known, min_token_accounts, timeout_seconds
    )
    if check.is_platform_vault and check.holder and check.holder not in known:
        known.add(check.holder)
        save_known_vaults(known)
    store[pool_address] = {
        "is_vault": check.is_platform_vault,
        "holder": check.holder,
        "holder_share_pct": check.holder_share_pct,
        "token_account_count": check.token_account_count,
        "checked_at": time.time(),
    }
    _save_check_cache(store)
    return check
