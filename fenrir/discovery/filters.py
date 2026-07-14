#!/usr/bin/env python3
"""
FENRIR - Discovery filter engine

Three declarative trading filters plus a universal-safety gate, all evaluated
against the chain-agnostic :class:`TokenSnapshot`. Chain-specific criteria (Solana
bond %/sniper/bundle, EVM taxes/honeypot/LP-lock) are only checked when the
relevant snapshot fields are populated — so no chain logic leaks in here.

Filters (spec):
  - LOW_CAP_ALPHA     — very early launches before migration.
  - MID_CAP_MOMENTUM  — approaching / just past migration.
  - HIGH_CAP          — established meme coins.

Policy:
  - Numeric market fields (mcap/liquidity/volume) come from DexScreener and are
    always present → enforced.
  - Optional fields (holders, bond %, sniper/bundle, safety flags) are ``None`` when
    the provider didn't supply them → the check is skipped with a warning
    (fail-open) unless ``strict`` is set. This keeps discovery from silently
    dropping tokens just because one provider was unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fenrir.discovery.models import Chain, FilterResult, TokenSnapshot


class FilterName(str, Enum):
    LOW_CAP_ALPHA = "low_cap_alpha"
    MID_CAP_MOMENTUM = "mid_cap_momentum"
    HIGH_CAP = "high_cap"


@dataclass
class FilterThresholds:
    """Threshold set for one filter. ``None`` means 'do not check this dimension'."""

    # Market cap (USD)
    min_market_cap_usd: float | None = None
    max_market_cap_usd: float | None = None
    # Age (minutes)
    min_age_minutes: float | None = None
    max_age_minutes: float | None = None
    # Liquidity / volume (USD)
    min_liquidity_usd: float | None = None
    min_volume_24h_usd: float | None = None
    # Holders
    min_holder_count: int | None = None
    max_holder_count: int | None = None
    min_buys_24h: int | None = None
    # Distribution caps (%)
    max_top_holder_pct: float | None = None
    max_dev_wallet_pct: float | None = None
    # Solana launch extras (%)
    max_bond_progress_pct: float | None = None
    min_bond_progress_pct: float | None = None
    max_sniper_pct: float | None = None
    max_bundle_pct: float | None = None
    # Boolean requirements
    require_buys_exceed_sells: bool = False
    require_migrated_or_bond: bool = False  # migrated OR bond >= min_bond_progress_pct
    require_verified: bool = False  # soft: warn when unverifiable
    require_lp_locked: bool = False


# ── Filter defaults (exact spec values) ───────────────────────────────

LOW_CAP_ALPHA = FilterThresholds(
    min_market_cap_usd=3_000.0,
    max_market_cap_usd=75_000.0,
    max_age_minutes=120.0,  # ideal 0–30m, hard cap 2h
    min_liquidity_usd=1_000.0,
    min_volume_24h_usd=2_000.0,
    min_holder_count=25,
    max_holder_count=250,
    min_buys_24h=15,
    max_top_holder_pct=12.0,
    max_dev_wallet_pct=10.0,
    max_bond_progress_pct=40.0,
    max_sniper_pct=20.0,
    max_bundle_pct=15.0,
    require_verified=True,
)

MID_CAP_MOMENTUM = FilterThresholds(
    min_market_cap_usd=80_000.0,
    max_market_cap_usd=900_000.0,
    min_age_minutes=30.0,
    max_age_minutes=7 * 24 * 60.0,  # 7 days
    min_liquidity_usd=35_000.0,
    min_volume_24h_usd=100_000.0,
    min_holder_count=400,
    max_holder_count=4_000,
    max_top_holder_pct=8.0,
    max_dev_wallet_pct=5.0,
    min_bond_progress_pct=65.0,
    require_migrated_or_bond=True,  # bond 65–100% OR migrated
    require_buys_exceed_sells=True,
    require_verified=True,
    require_lp_locked=True,
)

HIGH_CAP = FilterThresholds(
    min_market_cap_usd=1_000_000.0,
    min_age_minutes=24 * 60.0,  # > 1 day
    min_liquidity_usd=250_000.0,
    min_volume_24h_usd=1_000_000.0,
    min_holder_count=3_000,
    require_verified=True,
    require_lp_locked=True,
)

DEFAULT_THRESHOLDS: dict[FilterName, FilterThresholds] = {
    FilterName.LOW_CAP_ALPHA: LOW_CAP_ALPHA,
    FilterName.MID_CAP_MOMENTUM: MID_CAP_MOMENTUM,
    FilterName.HIGH_CAP: HIGH_CAP,
}


@dataclass
class UniversalSafety:
    """Baseline contract-safety gate applied on top of a filter (when data present).

    Numeric distribution caps stay per-filter; this covers the boolean contract
    gates + a transfer-tax ceiling. Every check is skipped when its snapshot field
    is ``None`` (provider didn't supply it) unless ``strict``.
    """

    enabled: bool = True
    require_not_honeypot: bool = True
    require_mint_disabled: bool = True
    require_freeze_disabled: bool = True
    # LP-lock is NOT universal: pre-migration launches hold liquidity in the
    # bonding curve (no lockable LP), so Low Cap Alpha must not be gated on it.
    # Mid/High Cap enforce LP-lock per-filter (require_lp_locked=True) instead.
    require_lp_locked: bool = False
    require_no_blacklist: bool = True
    max_transfer_tax_pct: float = 10.0
    # When True, a missing (None) safety signal FAILS instead of warning.
    strict: bool = False


class FilterEngine:
    """Evaluate a :class:`TokenSnapshot` against a filter + universal safety."""

    def __init__(
        self,
        thresholds: dict[FilterName, FilterThresholds] | None = None,
        universal: UniversalSafety | None = None,
    ) -> None:
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.universal = universal or UniversalSafety()

    def evaluate(self, snap: TokenSnapshot, filter_name: FilterName) -> FilterResult:
        thr = self.thresholds[filter_name]
        failures: list[str] = []
        warnings: list[str] = []

        self._check_market(snap, thr, failures)
        self._check_holders(snap, thr, failures, warnings)
        self._check_distribution(snap, thr, failures, warnings)
        self._check_solana_extras(snap, thr, failures, warnings)
        self._check_booleans(snap, thr, failures, warnings)
        if self.universal.enabled:
            self._check_universal_safety(snap, failures, warnings)

        return FilterResult(
            passed=not failures,
            filter_name=filter_name.value,
            token_address=snap.token_address,
            chain=snap.chain,
            failures=failures,
            warnings=warnings,
        )

    # ── Check groups ──────────────────────────────────────────────────

    @staticmethod
    def _check_market(snap: TokenSnapshot, thr: FilterThresholds, fails: list[str]) -> None:
        if thr.min_market_cap_usd is not None and snap.market_cap_usd < thr.min_market_cap_usd:
            fails.append(f"MCap ${snap.market_cap_usd:,.0f} < ${thr.min_market_cap_usd:,.0f}")
        if thr.max_market_cap_usd is not None and snap.market_cap_usd > thr.max_market_cap_usd:
            fails.append(f"MCap ${snap.market_cap_usd:,.0f} > ${thr.max_market_cap_usd:,.0f}")
        if thr.min_liquidity_usd is not None and snap.liquidity_usd < thr.min_liquidity_usd:
            fails.append(f"LP ${snap.liquidity_usd:,.0f} < ${thr.min_liquidity_usd:,.0f}")
        if thr.min_volume_24h_usd is not None and snap.volume_24h_usd < thr.min_volume_24h_usd:
            fails.append(f"Vol24h ${snap.volume_24h_usd:,.0f} < ${thr.min_volume_24h_usd:,.0f}")
        if thr.min_age_minutes is not None and snap.age_minutes < thr.min_age_minutes:
            fails.append(f"Age {snap.age_minutes:.0f}m < {thr.min_age_minutes:.0f}m")
        if thr.max_age_minutes is not None and snap.age_minutes > thr.max_age_minutes:
            fails.append(f"Age {snap.age_minutes:.0f}m > {thr.max_age_minutes:.0f}m")

    @staticmethod
    def _check_holders(
        snap: TokenSnapshot, thr: FilterThresholds, fails: list[str], warns: list[str]
    ) -> None:
        if thr.min_holder_count is not None or thr.max_holder_count is not None:
            if snap.holder_count is None:
                warns.append("holder count unavailable")
            else:
                if thr.min_holder_count is not None and snap.holder_count < thr.min_holder_count:
                    fails.append(f"Holders {snap.holder_count} < {thr.min_holder_count}")
                if thr.max_holder_count is not None and snap.holder_count > thr.max_holder_count:
                    fails.append(f"Holders {snap.holder_count} > {thr.max_holder_count}")
        if thr.min_buys_24h is not None and snap.txns_24h_buys < thr.min_buys_24h:
            fails.append(f"Buys24h {snap.txns_24h_buys} < {thr.min_buys_24h}")

    @staticmethod
    def _check_distribution(
        snap: TokenSnapshot, thr: FilterThresholds, fails: list[str], warns: list[str]
    ) -> None:
        _cap(snap.top_holder_pct, thr.max_top_holder_pct, "Top holder", fails, warns)
        _cap(snap.dev_wallet_pct, thr.max_dev_wallet_pct, "Dev wallet", fails, warns)

    @staticmethod
    def _check_solana_extras(
        snap: TokenSnapshot, thr: FilterThresholds, fails: list[str], warns: list[str]
    ) -> None:
        # Bond progress ceiling (Low Cap: pre-migration only).
        if thr.max_bond_progress_pct is not None:
            if snap.bond_progress_pct is None:
                warns.append("bond progress unavailable")
            elif snap.bond_progress_pct > thr.max_bond_progress_pct:
                fails.append(
                    f"Bond {snap.bond_progress_pct:.0f}% > {thr.max_bond_progress_pct:.0f}%"
                )
        _cap(snap.sniper_pct, thr.max_sniper_pct, "Snipers", fails, warns)
        _cap(snap.bundle_pct, thr.max_bundle_pct, "Bundled", fails, warns)

    @staticmethod
    def _check_booleans(
        snap: TokenSnapshot, thr: FilterThresholds, fails: list[str], warns: list[str]
    ) -> None:
        if thr.require_buys_exceed_sells and not snap.buys_exceed_sells:
            fails.append(f"Buys {snap.txns_24h_buys} <= Sells {snap.txns_24h_sells}")
        if thr.require_migrated_or_bond:
            migrated = bool(snap.migrated)
            bonded = (
                thr.min_bond_progress_pct is not None
                and snap.bond_progress_pct is not None
                and snap.bond_progress_pct >= thr.min_bond_progress_pct
            )
            if snap.migrated is None and snap.bond_progress_pct is None:
                warns.append("migration/bond status unavailable")
            elif not (migrated or bonded):
                fails.append("not migrated and bond below threshold")
        if thr.require_verified:
            if snap.safety.contract_verified is None:
                warns.append("contract verification unknown")
            elif snap.safety.contract_verified is False:
                fails.append("contract not verified")
        if thr.require_lp_locked:
            if snap.safety.lp_locked_or_burned is None:
                warns.append("LP lock status unknown")
            elif snap.safety.lp_locked_or_burned is False:
                fails.append("LP not locked/burned")

    def _check_universal_safety(
        self, snap: TokenSnapshot, fails: list[str], warns: list[str]
    ) -> None:
        u = self.universal
        s = snap.safety
        _gate(s.honeypot is False, s.honeypot, u.require_not_honeypot, "honeypot", fails, warns, u)
        _gate(
            s.mint_disabled is True,
            s.mint_disabled,
            u.require_mint_disabled,
            "mint not disabled",
            fails,
            warns,
            u,
        )
        _gate(
            s.freeze_disabled is True,
            s.freeze_disabled,
            u.require_freeze_disabled,
            "freeze not disabled",
            fails,
            warns,
            u,
        )
        _gate(
            s.lp_locked_or_burned is True,
            s.lp_locked_or_burned,
            u.require_lp_locked,
            "LP not locked",
            fails,
            warns,
            u,
        )
        _gate(
            s.blacklist_present is False,
            s.blacklist_present,
            u.require_no_blacklist,
            "blacklist function present",
            fails,
            warns,
            u,
        )
        for tax, label in ((s.buy_tax_pct, "buy"), (s.sell_tax_pct, "sell")):
            if tax is None:
                if u.strict:
                    fails.append(f"{label} tax unknown")
            elif tax > u.max_transfer_tax_pct:
                fails.append(f"{label} tax {tax:.1f}% > {u.max_transfer_tax_pct:.0f}%")


# ── Small check helpers ───────────────────────────────────────────────


def _cap(
    value: float | None, cap: float | None, label: str, fails: list[str], warns: list[str]
) -> None:
    """Fail when ``value`` exceeds ``cap``; warn (skip) when the value is unknown."""
    if cap is None:
        return
    if value is None:
        warns.append(f"{label} % unavailable")
    elif value > cap:
        fails.append(f"{label} {value:.1f}% > {cap:.0f}%")


def _gate(
    ok: bool,
    signal: bool | None,
    required: bool,
    fail_label: str,
    fails: list[str],
    warns: list[str],
    universal: UniversalSafety,
) -> None:
    """Boolean safety gate: pass when ``ok``; when the signal is None, warn unless strict."""
    if not required:
        return
    if signal is None:
        if universal.strict:
            fails.append(f"{fail_label} (unknown)")
        else:
            warns.append(f"{fail_label} unknown")
    elif not ok:
        fails.append(fail_label)


# Chain hint for callers: which snapshot extras are only meaningful on Solana.
SOLANA_ONLY_FIELDS: frozenset[str] = frozenset(
    {"bond_progress_pct", "migrated", "sniper_pct", "bundle_pct", "insider_pct"}
)


def is_solana_extra_relevant(chain: Chain) -> bool:
    return chain is Chain.SOLANA
