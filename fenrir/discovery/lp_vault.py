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

import json
import logging
import os
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


async def _rpc(session: aiohttp.ClientSession, rpc_url: str, method: str, params: list):
    async with session.post(
        rpc_url,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=aiohttp.ClientTimeout(total=30),
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
    timeout_seconds: float = 30.0,
) -> VaultCheck:
    """Check whether an LP mint is held by a platform vault wallet."""
    known_vaults = known_vaults or set()
    try:
        async with aiohttp.ClientSession(trust_env=True) as s:
            largest = (await _rpc(s, rpc_url, "getTokenLargestAccounts", [lp_mint]))["value"]
            if not largest:
                return VaultCheck(False)
            top = largest[0]
            top_addr = top["address"]
            top_amt = float(top.get("amount") or 0)
            if top_addr in known_vaults:
                return VaultCheck(True, holder=top_addr, cached=True)
            supply = float((await _rpc(s, rpc_url, "getTokenSupply", [lp_mint]))["value"]["amount"])
            share = (top_amt / supply) if supply else 0.0
            if share < VAULT_MIN_SHARE:
                return VaultCheck(False, holder=top_addr, holder_share_pct=share * 100)
            acct = await _rpc(s, rpc_url, "getAccountInfo", [top_addr, {"encoding": "jsonParsed"}])
            owner = acct["value"]["data"]["parsed"]["info"]["owner"]
            if owner in known_vaults:
                return VaultCheck(True, holder=owner, holder_share_pct=share * 100, cached=True)
            # Count only — base64 keeps the payload small.
            accounts = (
                await _rpc(
                    s,
                    rpc_url,
                    "getTokenAccountsByOwner",
                    [owner, {"programId": TOKEN_PROGRAM}, {"encoding": "base64"}],
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


def load_known_vaults(path: Path = DEFAULT_CACHE_PATH) -> set[str]:
    try:
        return set(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001 - missing/corrupt cache is fine
        return set()


def save_known_vaults(vaults: set[str], path: Path = DEFAULT_CACHE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(vaults)))
    except Exception as e:  # noqa: BLE001 - cache is best-effort
        log.debug("could not save lp vault cache: %s", e)
