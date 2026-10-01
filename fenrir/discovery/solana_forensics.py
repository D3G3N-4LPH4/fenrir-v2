#!/usr/bin/env python3
"""Solana distribution forensics: who actually holds the supply.

The 2026-10-01 volatility_breakout blowups (s/acc -99%, SIC -98.5%,
JANE -98.9%, HIHI -90.7%) all cleared with "Top-10 holders % unavailable" /
"Bundled % unavailable" — the concentration caps on the filter were dead
code because no provider supplied the data. Jupiter's holder enrichment only
covers some tokens; this module reads distribution straight off Solana RPC so
the caps can actually bite.

On-chain reads (<=4 RPC calls, ~5-10s):
1. ``getTokenLargestAccounts`` — top-20 token accounts + amounts.
2. ``getTokenSupply`` — total supply, for percentages.
3. ``getMultipleAccounts`` (top-20 token accounts) — parse each account's
   owner; aggregate amounts by owner wallet (one wallet can hold several
   ATAs; counting ATAs separately understates whales).
4. ``getMultipleAccounts`` (top owners) — classify each owner: a plain
   system-owned wallet counts as a holder; anything else (executable
   program, or an account owned by a launchpad/AMM program — the pump.fun
   bonding-curve PDA, Raydium/PumpSwap pool vaults, ...) is infrastructure
   and is excluded from the concentration math. Pre-migration the curve
   vault routinely holds 80%+ of supply; counting it as "top holder" would
   fail every young coin.

Fail-open at module level: any RPC failure returns ``None`` (unknown, never
a pass). The ``volatility_breakout`` filter fails CLOSED on unknown
distribution (``require_distribution_known``) — a vertical Solana move with
no holder data is exactly the shape that just blew up four times.

Follows the patterns of :mod:`fenrir.discovery.bundle_check` (Robinhood
counterpart): aiohttp only (never httpx), bracketed-IPv6 NO_PROXY
sanitization, bounded RPC budget, JSON file cache with 24h conclusive TTL
and 1h inconclusive backoff.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import base58
from solders.pubkey import Pubkey

from fenrir.discovery.lp_lock_v4 import Transport, _http_transport

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
#: System program: owner of plain wallet accounts. Only these count as holders.
SYSTEM_PROGRAM = "11111111111111111111111111111111"
#: File cache (mirrors bundle_check.py's layout).
_CACHE_PATH = os.path.expanduser("~/.cache/fenrir/solana_forensics.json")
_CACHE_TTL_S = 24 * 3600
#: Backoff for inconclusive checks (RPC failed / unclassifiable holders).
INCONCLUSIVE_TTL_S = 3600
#: RPC budget for one forensics run (caller adds its own wait_for on top).
_RPC_TIMEOUT_S = 8.0
#: How many of the largest token accounts to resolve owners for.
_TOP_ACCOUNTS = 20

_SOLANA_RPC_URL_DEFAULT = "https://api.mainnet-beta.solana.com"


# ── Report ───────────────────────────────────────────────────────────────
@dataclass
class SolanaForensicsReport:
    """Holder concentration for a Solana token, vaults excluded."""

    #: % of supply held by the largest non-vault wallet.
    top_holder_pct: float | None = None
    #: % of supply held by the top-10 non-vault wallets combined.
    top10_holder_pct: float | None = None
    #: Distinct non-vault wallets measured in the top-20 accounts.
    holders_measured: int = 0
    #: % of supply sitting in excluded vault/infrastructure accounts.
    excluded_vault_pct: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolanaForensicsReport:
        return cls(
            top_holder_pct=data.get("top_holder_pct"),
            top10_holder_pct=data.get("top10_holder_pct"),
            holders_measured=int(data.get("holders_measured", 0)),
            excluded_vault_pct=float(data.get("excluded_vault_pct", 0.0)),
            detail=str(data.get("detail", "")),
        )


# ── Cache ────────────────────────────────────────────────────────────────
def _load_cache() -> dict[str, Any]:
    try:
        with open(_CACHE_PATH) as f:
            loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def get_cached_forensics(mint: str) -> dict[str, Any] | None:
    """Return the cached report dict when fresh, else None."""
    entry = _load_cache().get(mint.lower())
    if not isinstance(entry, dict):
        return None
    ttl = float(entry.get("ttl", _CACHE_TTL_S))
    if time.time() - float(entry.get("cached_at", 0)) > ttl:
        return None
    report = entry.get("report")
    return report if isinstance(report, dict) else None


def save_cached_forensics(
    mint: str, report: dict[str, Any], ttl_seconds: float = _CACHE_TTL_S
) -> None:
    cache = _load_cache()
    cache[mint.lower()] = {
        "cached_at": time.time(),
        "ttl": ttl_seconds,
        "report": report,
    }
    _save_cache(cache)


# ── RPC helpers ──────────────────────────────────────────────────────────
def _parse_token_account_owner(account_info: dict[str, Any] | None) -> str | None:
    """Extract the owner wallet from a token account's info (None if unknown).

    Token account layout: mint (0..32), owner (32..64), amount (64..72).
    """
    if not account_info:
        return None
    data = account_info.get("data")
    if not isinstance(data, list) or len(data) < 1:
        return None
    try:
        raw = base64.b64decode(data[0])
    except Exception:  # noqa: BLE001 - malformed payload, fail-open
        return None
    if len(raw) < 64:
        return None
    return base58.b58encode(raw[32:64]).decode()


async def _rpc(transport: Transport, method: str, params: list) -> Any:
    try:
        return await transport(method, params)
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("solana_forensics RPC %s failed: %s", method, e)
        return None


# ── Main check ───────────────────────────────────────────────────────────
async def check_solana_distribution(
    mint: str,
    transport: Transport | None = None,
    rpc_url: str | None = None,
) -> SolanaForensicsReport | None:
    """Measure holder concentration for a Solana token (None when unknown).

    Fail-open: returns None on any RPC/parse failure — the caller (filter)
    decides whether unknown distribution passes or fails.
    """
    owned_transport = transport is None
    if transport is None:
        transport = await _http_transport(
            rpc_url or os.getenv("SOLANA_RPC_URL") or _SOLANA_RPC_URL_DEFAULT,
            _RPC_TIMEOUT_S,
        )
    try:
        return await _check(mint, transport)
    finally:
        if owned_transport:
            session = getattr(transport, "_session", None)
            if session is not None:
                with contextlib.suppress(Exception):
                    await session.close()


async def _check(mint: str, transport: Transport) -> SolanaForensicsReport | None:
    largest = await _rpc(transport, "getTokenLargestAccounts", [mint])
    supply = await _rpc(transport, "getTokenSupply", [mint])
    if not largest or not supply:
        return None
    accounts = largest.get("value") or []
    supply_value = supply.get("value") or {}
    try:
        total = int(supply_value.get("amount", "0"))
    except (TypeError, ValueError):
        return None
    if total <= 0 or not accounts:
        return None

    # Resolve each top token account to its owner wallet (one batched call).
    infos = await _rpc(
        transport,
        "getMultipleAccounts",
        [[a["address"] for a in accounts[:_TOP_ACCOUNTS]], {"encoding": "base64"}],
    )
    if not infos or not isinstance(infos.get("value"), list):
        return None
    by_owner: dict[str, int] = {}
    for entry, info in zip(accounts[:_TOP_ACCOUNTS], infos["value"], strict=False):
        owner = _parse_token_account_owner(info)
        if owner is None:
            # Unresolvable holder in the measured set: concentration is
            # unknowable, not zero — fail-open (None), never a clean pass.
            return None
        try:
            amount = int(entry.get("amount", "0"))
        except (TypeError, ValueError):
            return None
        by_owner[owner] = by_owner.get(owner, 0) + amount

    # Classify the top owners: plain wallets count, infrastructure doesn't.
    ranked = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    owner_infos = await _rpc(
        transport,
        "getMultipleAccounts",
        [[o for o, _ in ranked[:12]], {"encoding": "jsonParsed"}],
    )
    if not owner_infos or not isinstance(owner_infos.get("value"), list):
        return None
    holder_amounts: list[int] = []
    vault_amount = 0
    for (owner, amount), info in zip(ranked[:12], owner_infos["value"], strict=False):
        if not info:
            # Owner pubkey with no account (closed wallet, burn address,
            # PDA): off-curve pubkeys are program-controlled → vault;
            # on-curve ones are wallets that just hold no SOL → holder.
            try:
                is_vault = not Pubkey.from_string(owner).is_on_curve()
            except Exception:  # noqa: BLE001 - malformed pubkey, fail-open
                return None
            if is_vault:
                vault_amount += amount
            else:
                holder_amounts.append(amount)
            continue
        program_owner = info.get("owner")
        if info.get("executable") or program_owner != SYSTEM_PROGRAM:
            vault_amount += amount
        else:
            holder_amounts.append(amount)
    # Owners beyond the top-12 are small; count them as holders (they can
    # only dilute concentration, and none individually moves the top-10).
    for _owner, amount in ranked[12:]:
        holder_amounts.append(amount)

    holder_amounts.sort(reverse=True)
    top1 = holder_amounts[0] / total * 100 if holder_amounts else 0.0
    top10 = sum(holder_amounts[:10]) / total * 100 if holder_amounts else 0.0
    vault_pct = vault_amount / total * 100
    detail = (
        f"top holder {top1:.1f}%, top-10 {top10:.1f}% "
        f"({len(holder_amounts)} wallets in top-{_TOP_ACCOUNTS} accounts, "
        f"{vault_pct:.1f}% in vaults)"
    )
    return SolanaForensicsReport(
        top_holder_pct=round(top1, 2),
        top10_holder_pct=round(top10, 2),
        holders_measured=len(holder_amounts),
        excluded_vault_pct=round(vault_pct, 2),
        detail=detail,
    )


async def main() -> None:  # pragma: no cover - manual smoke test
    import sys

    mint = sys.argv[1] if len(sys.argv) > 1 else ""
    if not mint:
        print("usage: solana_forensics.py <mint>")
        raise SystemExit(2)
    report = await check_solana_distribution(mint)
    print(report.as_dict() if report else "inconclusive")


if __name__ == "__main__":
    asyncio.run(main())
