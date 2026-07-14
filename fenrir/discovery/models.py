#!/usr/bin/env python3
"""
FENRIR - Discovery data models

Normalized, chain-agnostic snapshot of a token that the filter engine and scoring
engine operate on. Chain adapters populate these dataclasses from their own data
sources (DexScreener market data, RugCheck/GoPlus safety, on-chain holders, …), so
the shared components never see chain-specific shapes.

Design notes:
  - Every "safety" field is ``bool | float | None``. ``None`` means the provider
    did NOT supply the signal (unknown), which is distinct from ``False``/0.0. The
    universal-safety policy decides whether an unknown fails open or closed.
  - ``TokenSnapshot`` is a superset of ``fenrir/filters/market.py::MarketData`` — the
    DexScreener provider produces this and the market filter reads the same fields,
    so there is a single source of truth for market metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Chain(str, Enum):
    """Supported chains. ``str`` mixin so values serialize cleanly to JSON/env."""

    SOLANA = "solana"
    ETHEREUM = "ethereum"
    BNB = "bnb"
    BASE = "base"

    @classmethod
    def from_dexscreener(cls, chain_id: str | None) -> Chain | None:
        """Map a DexScreener ``chainId`` to a :class:`Chain` (None if unsupported)."""
        return _DEXSCREENER_CHAIN_IDS.get((chain_id or "").lower())


# DexScreener chainId → Chain. (DexScreener uses "bsc" for BNB Chain.)
_DEXSCREENER_CHAIN_IDS: dict[str, Chain] = {
    "solana": Chain.SOLANA,
    "ethereum": Chain.ETHEREUM,
    "bsc": Chain.BNB,
    "base": Chain.BASE,
}


@dataclass
class SafetySignals:
    """Contract/liquidity safety signals, provider-supplied.

    Each field is ``None`` when unknown (provider didn't return it) — NOT the same
    as a negative result. The universal-safety filter treats unknowns per its
    fail-open/closed policy.
    """

    mint_disabled: bool | None = None  # mint authority revoked (no new supply)
    freeze_disabled: bool | None = None  # freeze authority revoked
    lp_locked_or_burned: bool | None = None  # LP burned or locked
    lp_locked_pct: float | None = None  # % of LP burned/locked
    honeypot: bool | None = None  # True = cannot sell (EVM)
    buy_tax_pct: float | None = None  # EVM buy tax %
    sell_tax_pct: float | None = None  # EVM sell tax %
    contract_verified: bool | None = None
    ownership_renounced: bool | None = None
    blacklist_present: bool | None = None  # contract can blacklist wallets

    # Provider risk score (RugCheck score_normalised / GoPlus-derived), lower=safer.
    risk_score: float | None = None
    # Free-form risk labels surfaced by the provider (e.g. "mint live").
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class TokenSnapshot:
    """Normalized, chain-agnostic snapshot of a token.

    Superset of the DexScreener market metrics plus holder distribution, chain
    extras (Solana bond/sniper/bundle), socials and a nested :class:`SafetySignals`.
    Filters/scoring read ONLY this object.
    """

    # ── Identity ──────────────────────────────────────────────────────
    chain: Chain
    token_address: str
    symbol: str = "???"
    name: str = "Unknown"
    pair_address: str | None = None
    dex_id: str | None = None  # "raydium", "pumpfun", "uniswap", "aerodrome", …

    # ── Pricing / market cap ──────────────────────────────────────────
    price_usd: float = 0.0
    market_cap_usd: float = 0.0
    fdv_usd: float = 0.0

    # ── Liquidity ─────────────────────────────────────────────────────
    liquidity_usd: float = 0.0
    liquidity_sol: float | None = None  # Solana post-migration LP in SOL

    # ── Volume ────────────────────────────────────────────────────────
    volume_5m_usd: float = 0.0
    volume_1h_usd: float = 0.0
    volume_6h_usd: float = 0.0
    volume_24h_usd: float = 0.0

    # ── Transactions (buy pressure) ───────────────────────────────────
    txns_5m_buys: int = 0
    txns_5m_sells: int = 0
    txns_1h_buys: int = 0
    txns_1h_sells: int = 0
    txns_24h_buys: int = 0
    txns_24h_sells: int = 0

    # ── Price change ──────────────────────────────────────────────────
    price_change_5m_pct: float = 0.0
    price_change_1h_pct: float = 0.0
    price_change_24h_pct: float = 0.0

    # ── Age ───────────────────────────────────────────────────────────
    created_at: datetime | None = None
    age_minutes: float = 0.0

    # ── Holders / distribution ────────────────────────────────────────
    holder_count: int | None = None
    top_holder_pct: float | None = None  # single largest holder %
    dev_wallet_pct: float | None = None  # creator/deployer holdings %

    # ── Solana-specific launch extras (None off Solana) ───────────────
    bond_progress_pct: float | None = None  # pump.fun bonding-curve progress
    migrated: bool | None = None  # graduated off the curve to an AMM
    creator_pct: float | None = None
    sniper_pct: float | None = None  # % held by launch snipers
    insider_pct: float | None = None  # % held by insider wallets
    bundle_pct: float | None = None  # % bought in bundled txs

    # ── Community ─────────────────────────────────────────────────────
    twitter: str | None = None
    telegram: str | None = None
    website: str | None = None
    organic_score: float | None = None  # provider organic-activity score

    # ── Safety ────────────────────────────────────────────────────────
    safety: SafetySignals = field(default_factory=SafetySignals)

    # Raw provider payloads for debugging / downstream context.
    raw: dict[str, Any] = field(default_factory=dict)

    # ── Derived helpers ───────────────────────────────────────────────

    @property
    def txns_24h_total(self) -> int:
        return self.txns_24h_buys + self.txns_24h_sells

    @property
    def buys_exceed_sells(self) -> bool:
        """More 24h buys than sells (buy pressure)."""
        return self.txns_24h_buys > self.txns_24h_sells

    @property
    def buy_pressure_24h(self) -> float:
        """Buy fraction over 24h. 0.5 when no data (neutral)."""
        total = self.txns_24h_total
        return self.txns_24h_buys / total if total > 0 else 0.5

    @property
    def liquidity_to_mcap(self) -> float:
        """Liquidity depth relative to market cap (0.0 when mcap unknown)."""
        return self.liquidity_usd / self.market_cap_usd if self.market_cap_usd > 0 else 0.0


@dataclass
class FilterResult:
    """Outcome of evaluating one filter against a snapshot.

    Mirrors ``fenrir/filters/market.py::MarketFilterResult`` (failures/warnings) so
    the two subsystems read alike.
    """

    passed: bool
    filter_name: str
    token_address: str
    chain: Chain
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        head = f"[{status}][{self.filter_name}] {self.token_address[:8]}…"
        return f"{head} — {', '.join(self.failures)}" if self.failures else head


@dataclass
class ScoreBreakdown:
    """Weighted 0–100 scores for a token."""

    overall: float = 0.0
    momentum: float = 0.0
    safety: float = 0.0
    liquidity: float = 0.0
    holder: float = 0.0
    community: float = 0.0
    risk: float = 0.0  # 0 = safe, 100 = risky (inverse of the others)

    def as_dict(self) -> dict[str, float]:
        return {
            "overall": round(self.overall, 1),
            "momentum": round(self.momentum, 1),
            "safety": round(self.safety, 1),
            "liquidity": round(self.liquidity, 1),
            "holder": round(self.holder, 1),
            "community": round(self.community, 1),
            "risk": round(self.risk, 1),
        }
