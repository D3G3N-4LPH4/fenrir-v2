#!/usr/bin/env python3
"""
FENRIR - GoPlus Security provider (EVM safety)

The EVM analogue of RugCheck (Solana): a keyless multi-EVM token-security API that
supplies honeypot / buy-sell tax / LP-lock / verified / renounced / blacklist /
holder-distribution signals for Ethereum, BNB Chain and Base.

Endpoint (no key):
  GET https://api.gopluslabs.io/api/v1/token_security/{chain_id}?contract_addresses={addr}

The pure ``parse_goplus`` mapper is unit-testable without network. Percentages in
the GoPlus response are FRACTIONS (0.0855 = 8.55%) — normalized to % here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fenrir.discovery.models import Chain, SafetySignals

logger = logging.getLogger("FENRIR.GoPlus")

GOPLUS_API = "https://api.gopluslabs.io/api/v1/token_security"

# Chain → GoPlus chain_id.
GOPLUS_CHAIN_IDS: dict[Chain, str] = {
    Chain.ETHEREUM: "1",
    Chain.BNB: "56",
    Chain.BASE: "8453",
    # Robinhood Chain (Arbitrum Orbit L2, chain id 4663) — verified live on
    # GoPlus 2026-09-28. This closes the "safety unknown on Robinhood" gap.
    Chain.ROBINHOOD: "4663",
}

_ZERO = "0x0000000000000000000000000000000000000000"
_DEAD = "0x000000000000000000000000000000000000dead"
_BURN_ADDRS = frozenset({_ZERO, _DEAD})


def _b(value: Any) -> bool | None:
    """GoPlus booleans are '0'/'1' strings; None/missing → unknown."""
    if value is None or value == "":
        return None
    return str(value) == "1"


def _pct(value: Any) -> float | None:
    """GoPlus fraction string → percent; None when unparseable."""
    try:
        return float(value) * 100.0 if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


@dataclass
class GoPlusSecurity:
    """Parsed GoPlus result: safety signals + holder distribution."""

    safety: SafetySignals = field(default_factory=SafetySignals)
    holder_count: int | None = None
    top_holder_pct: float | None = None  # raw holders[0] (may be the pool — see below)
    dev_wallet_pct: float | None = None
    # Top-10 holders as (address, percent, is_contract, is_locked) tuples.
    # Callers compute wallet concentration via ``distribution_metrics`` —
    # holders[0] is very often AMM infrastructure (a v2/v3 pool contract, or
    # the Uniswap v4 PoolManager singleton), which is liquidity, not a whale.
    holders: list[tuple[str, float, int | None, int | None]] = field(default_factory=list)


def distribution_metrics(
    holders: list[tuple[str, float, int | None, int | None]],
    exclude_addresses: set[str] | None = None,
) -> tuple[float | None, float | None]:
    """(top_holder_pct, top10_holder_pct) over EOA-held supply — the dump risk.

    Pure helper shared by the EVM adapter and the scout's enrich path.
    Excludes, in order:

    1. Explicit ``exclude_addresses`` (a known pool/pair address).
    2. Any holder flagged ``is_contract`` — AMM infrastructure (v2/v3 pool
       contracts, the Uniswap v4 PoolManager singleton) custodies tokens but
       never market-sells them. Address-matching alone can't catch v4: the
       PoolManager serves every pool, and DexScreener reports v4 pools by
       PoolId (bytes32), so there is no pool address to match.
    3. Holders flagged ``is_locked`` (burned/locked supply can't dump).

    Exclusion needs an explicit positive flag (``== 1``); unknown (``None``)
    flags keep the holder, so missing data fails conservative instead of
    silently shrinking the metric.
    """
    exclude = {a.lower() for a in (exclude_addresses or set()) if a}
    wallet = [
        (a, p)
        for a, p, is_contract, is_locked in holders
        if a.lower() not in exclude and is_contract != 1 and is_locked != 1
    ]
    if not wallet:
        return None, None
    top = max(p for _, p in wallet)
    top10 = sum(p for _, p in sorted(wallet, key=lambda x: x[1], reverse=True)[:10])
    return top, top10


def _lp_locked_pct(res: dict[str, Any]) -> float | None:
    """Sum LP % held in locked positions or burn addresses (0.0–100.0)."""
    lp = res.get("lp_holders")
    if not lp:
        return None
    total = 0.0
    for h in lp:
        if not isinstance(h, dict):
            continue
        addr = str(h.get("address", "")).lower()
        locked = str(h.get("is_locked", "0")) == "1"
        tag = str(h.get("tag", "")).lower()
        if locked or addr in _BURN_ADDRS or "burn" in tag or "lock" in tag:
            try:
                total += float(h.get("percent") or 0) * 100.0
            except (TypeError, ValueError):
                continue
    return round(total, 2)


def parse_goplus(res: dict[str, Any]) -> GoPlusSecurity:
    """Map a GoPlus ``token_security`` result entry to safety + holder fields (pure)."""
    owner = str(res.get("owner_address", "")).lower()
    renounced: bool | None
    if owner == "":
        renounced = None
    else:
        renounced = (
            owner in _BURN_ADDRS
            and str(res.get("can_take_back_ownership", "0")) != "1"
            and str(res.get("hidden_owner", "0")) != "1"
        )

    lp_pct = _lp_locked_pct(res)
    holders = res.get("holders") or []
    top_pct = None
    pairs: list[tuple[str, float, int | None, int | None]] = []
    for h in holders:
        if not isinstance(h, dict):
            continue
        addr = str(h.get("address", ""))
        pct = _pct(h.get("percent"))
        if addr and pct is not None:
            is_contract = h.get("is_contract")
            is_locked = h.get("is_locked")
            pairs.append(
                (
                    addr,
                    pct,
                    int(is_contract) if isinstance(is_contract, int | bool) else None,
                    int(is_locked) if isinstance(is_locked, int | bool) else None,
                )
            )
    if pairs:
        top_pct = max(p for _, p, _, _ in pairs)

    safety = SafetySignals(
        # EVM has no mint/freeze authority; map the closest analogues.
        mint_disabled=(
            None if res.get("is_mintable") in (None, "") else not _b(res.get("is_mintable"))
        ),
        freeze_disabled=(
            None
            if res.get("transfer_pausable") in (None, "")
            else not _b(res.get("transfer_pausable"))
        ),
        lp_locked_or_burned=(lp_pct >= 90.0 if lp_pct is not None else None),
        lp_locked_pct=lp_pct,
        honeypot=_b(res.get("is_honeypot")),
        buy_tax_pct=_pct(res.get("buy_tax")),
        sell_tax_pct=_pct(res.get("sell_tax")),
        contract_verified=_b(res.get("is_open_source")),
        ownership_renounced=renounced,
        blacklist_present=_b(res.get("is_blacklisted")),
    )
    return GoPlusSecurity(
        safety=safety,
        holder_count=_int(res.get("holder_count")),
        top_holder_pct=top_pct,
        dev_wallet_pct=_pct(res.get("creator_percent")),
        holders=pairs,
    )


class GoPlusProvider:
    """Fetches EVM token-security signals from GoPlus (keyless, fail-open)."""

    def __init__(self, timeout_seconds: float = 6.0) -> None:
        self.timeout = timeout_seconds
        self._session: Any = None

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(
                trust_env=True, headers={"User-Agent": "FENRIR/2.0 discovery"}
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def token_security(self, chain: Chain, address: str) -> GoPlusSecurity | None:
        """Fetch + parse GoPlus security for ``address`` on ``chain`` (None on failure)."""
        chain_id = GOPLUS_CHAIN_IDS.get(chain)
        if chain_id is None:
            return None
        try:
            session = await self._get_session()
            url = f"{GOPLUS_API}/{chain_id}?contract_addresses={address}"
            async with session.get(url, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.warning("GoPlus HTTP %d for %s…", resp.status, address[:10])
                    return None
                data = await resp.json()
            result = (data.get("result") or {}).get(address.lower())
            return parse_goplus(result) if isinstance(result, dict) else None
        except TimeoutError:
            logger.warning("GoPlus timeout for %s…", address[:10])
            return None
        except Exception as e:  # noqa: BLE001 - fail-open like RugCheck
            logger.warning("GoPlus error for %s…: %s", address[:10], e)
            return None
