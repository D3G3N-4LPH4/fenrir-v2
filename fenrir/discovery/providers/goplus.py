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
    top_holder_pct: float | None = None
    dev_wallet_pct: float | None = None


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
    if holders and isinstance(holders[0], dict):
        top_pct = _pct(holders[0].get("percent"))

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
    )


class GoPlusProvider:
    """Fetches EVM token-security signals from GoPlus (keyless, fail-open)."""

    def __init__(self, timeout_seconds: float = 6.0) -> None:
        self.timeout = timeout_seconds
        self._session: Any = None

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(headers={"User-Agent": "FENRIR/2.0 discovery"})
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
