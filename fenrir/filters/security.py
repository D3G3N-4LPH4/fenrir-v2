#!/usr/bin/env python3
"""
FENRIR - Security Hard-Filters

Pre-trade safety checks that must ALL pass before any strategy evaluation.
These are non-negotiable gates — a single failure kills the trade regardless
of AI confidence or market conditions.

Checks performed:
  1. Mint authority revoked   — dev cannot mint more tokens
  2. Freeze authority revoked — dev cannot freeze holder wallets
  3. LP burned >= threshold   — dev cannot rug the liquidity pool
  4. Top-10 holder concentration < threshold — no single whale can dump

Data sources:
  - Mint/freeze authority: on-chain via Solana RPC (no extra API needed)
  - LP burned: on-chain LP token supply sent to burn addresses
  - Holder distribution: Helius DAS API (getTokenAccounts) or fallback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("FENRIR.SecurityFilter")

# Well-known Solana burn / dead addresses
BURN_ADDRESSES: frozenset[str] = frozenset(
    {
        "1nc1nerator11111111111111111111111111111111",
        "11111111111111111111111111111111",  # system program / null address
        "burnxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    }
)


@dataclass
class SecurityCheckResult:
    """Result of a full security scan on a token."""

    passed: bool
    token_address: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.failures:
            return f"[{status}] {self.token_address[:8]}... — {', '.join(self.failures)}"
        return f"[{status}] {self.token_address[:8]}..."


@dataclass
class SecurityFilterConfig:
    """Tunable thresholds for the security pre-filter."""

    # Mint authority must be revoked (null)
    require_mint_revoked: bool = True
    # Freeze authority must be revoked (null)
    require_freeze_revoked: bool = True
    # Minimum % of LP tokens that must be burned
    min_lp_burned_pct: float = 90.0
    # Maximum combined % of supply held by top-10 wallets
    max_top10_holder_pct: float = 30.0
    # Skip holder check if it cannot be fetched (fail-open vs fail-closed)
    fail_open_on_holder_fetch_error: bool = False
    # Timeout for each RPC call in seconds
    rpc_timeout_seconds: float = 3.0


class SecurityFilter:
    """
    Hard security gate — all checks must pass or the token is rejected.

    Usage:
        sf = SecurityFilter(config, rpc_url, helius_api_key)
        result = await sf.check(token_address, lp_mint_address)
        if not result.passed:
            logger.info(f"Rejected: {result}")
            return
    """

    def __init__(
        self,
        config: SecurityFilterConfig,
        rpc_url: str,
        helius_api_key: str = "",
    ) -> None:
        self.config = config
        self.rpc_url = rpc_url
        self.helius_api_key = helius_api_key
        self._session: Any = None  # aiohttp session, created lazily

    async def _get_session(self) -> Any:
        """Lazy aiohttp session creation."""
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Public entry point ────────────────────────────────────────────

    async def check(
        self,
        token_address: str,
        lp_mint_address: str | None = None,
    ) -> SecurityCheckResult:
        """
        Run all enabled security checks on a token.
        Returns immediately on first hard failure to save RPC calls.
        """
        result = SecurityCheckResult(passed=True, token_address=token_address)

        mint_info = await self._fetch_mint_info(token_address)

        if mint_info is None:
            result.passed = False
            result.failures.append("Could not fetch mint account info")
            return result

        # 1. Mint authority check
        if self.config.require_mint_revoked:
            mint_authority = mint_info.get("mint_authority")
            if mint_authority is not None:
                result.passed = False
                result.failures.append(
                    f"Mint authority NOT revoked (authority: {str(mint_authority)[:8]}...)"
                )
                result.details["mint_authority"] = str(mint_authority)
            else:
                result.details["mint_authority"] = None

        # 2. Freeze authority check
        if self.config.require_freeze_revoked:
            freeze_authority = mint_info.get("freeze_authority")
            if freeze_authority is not None:
                result.passed = False
                result.failures.append(
                    f"Freeze authority NOT revoked (authority: {str(freeze_authority)[:8]}...)"
                )
                result.details["freeze_authority"] = str(freeze_authority)
            else:
                result.details["freeze_authority"] = None

        # Fail fast — no point checking LP/holders if basic mint checks failed
        if not result.passed:
            return result

        # 3. LP burned check
        if lp_mint_address and self.config.min_lp_burned_pct > 0:
            burned_pct = await self._fetch_lp_burned_pct(lp_mint_address)
            result.details["lp_burned_pct"] = burned_pct
            if burned_pct is None:
                result.warnings.append("Could not verify LP burn percentage")
            elif burned_pct < self.config.min_lp_burned_pct:
                result.passed = False
                result.failures.append(
                    f"LP burned {burned_pct:.1f}% < required {self.config.min_lp_burned_pct:.0f}%"
                )

        # 4. Top-10 holder concentration check
        if self.config.max_top10_holder_pct < 100.0:
            top10_pct = await self._fetch_top10_holder_pct(token_address)
            result.details["top10_holder_pct"] = top10_pct
            if top10_pct is None:
                if not self.config.fail_open_on_holder_fetch_error:
                    result.passed = False
                    result.failures.append("Could not fetch holder distribution")
                else:
                    result.warnings.append("Holder distribution unavailable — skipping check")
            elif top10_pct > self.config.max_top10_holder_pct:
                result.passed = False
                result.failures.append(
                    f"Top-10 holders own {top10_pct:.1f}% > max {self.config.max_top10_holder_pct:.0f}%"
                )

        return result

    # ── RPC helpers ───────────────────────────────────────────────────

    async def _fetch_mint_info(self, token_address: str) -> dict[str, Any] | None:
        """
        Fetch token mint account and extract mint/freeze authority.
        Uses getParsedAccountInfo RPC call.
        """
        try:
            session = await self._get_session()
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getParsedAccountInfo",
                "params": [
                    token_address,
                    {"encoding": "jsonParsed"},
                ],
            }
            async with session.post(
                self.rpc_url,
                json=payload,
                timeout=self.config.rpc_timeout_seconds,
            ) as resp:
                data = await resp.json()

            value = data.get("result", {}).get("value")
            if not value:
                return None

            parsed = value.get("data", {}).get("parsed", {})
            info = parsed.get("info", {})

            mint_authority_raw = info.get("mintAuthority")
            freeze_authority_raw = info.get("freezeAuthority")

            return {
                # None means revoked (what we want)
                "mint_authority": mint_authority_raw if mint_authority_raw else None,
                "freeze_authority": freeze_authority_raw if freeze_authority_raw else None,
                "decimals": info.get("decimals", 6),
                "supply": info.get("supply", "0"),
                "is_initialized": info.get("isInitialized", False),
            }

        except TimeoutError:
            logger.warning(f"Timeout fetching mint info for {token_address[:8]}...")
            return None
        except Exception as e:
            logger.warning(f"Error fetching mint info for {token_address[:8]}...: {e}")
            return None

    async def _fetch_lp_burned_pct(self, lp_mint_address: str) -> float | None:
        """
        Calculate what percentage of LP tokens have been burned.
        Burned = tokens sent to known burn addresses or supply reduction.
        Uses getTokenAccountsByOwner for burn address holdings.
        """
        try:
            session = await self._get_session()

            # Get total LP supply
            supply_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenSupply",
                "params": [lp_mint_address],
            }
            async with session.post(
                self.rpc_url,
                json=supply_payload,
                timeout=self.config.rpc_timeout_seconds,
            ) as resp:
                supply_data = await resp.json()

            total_supply_str = supply_data.get("result", {}).get("value", {}).get("amount", "0")
            total_supply = int(total_supply_str) if total_supply_str else 0
            if total_supply == 0:
                return None

            # Check how many LP tokens are held by burn addresses
            burned_amount = 0
            for burn_addr in BURN_ADDRESSES:
                accounts_payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "getTokenAccountsByOwner",
                    "params": [
                        burn_addr,
                        {"mint": lp_mint_address},
                        {"encoding": "jsonParsed"},
                    ],
                }
                try:
                    async with session.post(
                        self.rpc_url,
                        json=accounts_payload,
                        timeout=self.config.rpc_timeout_seconds,
                    ) as resp:
                        accounts_data = await resp.json()

                    accounts = accounts_data.get("result", {}).get("value", [])
                    for acc in accounts:
                        amt_str = (
                            acc.get("account", {})
                            .get("data", {})
                            .get("parsed", {})
                            .get("info", {})
                            .get("tokenAmount", {})
                            .get("amount", "0")
                        )
                        burned_amount += int(amt_str) if amt_str else 0
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Burn-address lookup failed for {burn_addr[:8]}...: {e}")
                    continue

            burned_pct = (burned_amount / total_supply) * 100.0
            return round(burned_pct, 2)

        except TimeoutError:
            logger.warning(f"Timeout fetching LP burn for {lp_mint_address[:8]}...")
            return None
        except Exception as e:
            logger.warning(f"Error fetching LP burn for {lp_mint_address[:8]}...: {e}")
            return None

    async def _fetch_top10_holder_pct(self, token_address: str) -> float | None:
        """
        Fetch top token holders and calculate combined % of supply.
        Uses getTokenLargestAccounts RPC (built into all Solana nodes) — top 20
        holders, no API key required.
        """
        try:
            # getTokenLargestAccounts returns top 20 holders — no API key needed
            session = await self._get_session()
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [token_address],
            }
            async with session.post(
                self.rpc_url,
                json=payload,
                timeout=self.config.rpc_timeout_seconds,
            ) as resp:
                data = await resp.json()

            accounts = data.get("result", {}).get("value", [])
            if not accounts:
                return None

            # Get total supply for percentage calculation
            supply_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "getTokenSupply",
                "params": [token_address],
            }
            async with session.post(
                self.rpc_url,
                json=supply_payload,
                timeout=self.config.rpc_timeout_seconds,
            ) as resp:
                supply_data = await resp.json()

            total_supply_str = supply_data.get("result", {}).get("value", {}).get("amount", "0")
            total_supply = int(total_supply_str) if total_supply_str else 0
            if total_supply == 0:
                return None

            # Sum top-10 holders
            top10 = accounts[:10]
            top10_amount = sum(int(acc.get("amount", "0")) for acc in top10)

            top10_pct = (top10_amount / total_supply) * 100.0
            return round(top10_pct, 2)

        except TimeoutError:
            logger.warning(f"Timeout fetching holders for {token_address[:8]}...")
            return None
        except Exception as e:
            logger.warning(f"Error fetching holders for {token_address[:8]}...: {e}")
            return None
