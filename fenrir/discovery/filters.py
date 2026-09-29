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
  - DEGEN_LAUNCH      — risk-on: minutes-old trench launches, tiny caps.
  - VOLATILITY_BREAKOUT — risk-on: the vertical 1h move other filters reject.

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
    DEGEN_LAUNCH = "degen_launch"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    VOLUME_SURGE = "volume_surge"
    GRADUATION_WATCH = "graduation_watch"


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
    # Flow freshness (1h window): buy/sell ratio + share of 24h volume in the last hour.
    # The ratio is only enforced when the 1h txn sample is big enough to mean something.
    min_buy_sell_ratio_1h: float | None = None
    min_volume_1h_share: float | None = None
    # Turnover: 24h volume / market cap. Too high = churn/wash; too low = dead tape.
    min_turnover_24h: float | None = None
    max_turnover_24h: float | None = None
    # Liquidity depth relative to market cap (%). Absolute floors alone let a
    # $35k pool "support" anything from $80k to $900k mcap identically.
    min_liquidity_to_mcap_pct: float | None = None
    # Chase guard: reject entries going vertical (None = no guard on this window).
    # Low caps are exempt by design — early volatility IS the low_cap thesis.
    # min_price_change_1h_pct inverts the guard: REQUIRE a minimum 1h move
    # (used by volatility_breakout — the vertical move is the signal).
    min_price_change_1h_pct: float | None = None
    max_price_change_1h_pct: float | None = None
    max_price_change_24h_pct: float | None = None
    # Distribution caps (%)
    max_top_holder_pct: float | None = None
    max_top10_holder_pct: float | None = None  # concentration / bundle proxy (pool excluded)
    max_dev_wallet_pct: float | None = None
    # Solana launch extras (%)
    max_bond_progress_pct: float | None = None
    min_bond_progress_pct: float | None = None
    min_bond_inflow_sol: float | None = None  # SOL into the curve since prev check
    require_bond_data: bool = False  # fail (not warn) when bond progress unknown
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
    min_liquidity_to_mcap_pct=5.0,
    min_volume_24h_usd=2_000.0,
    min_volume_1h_share=0.04,  # ≥4% of daily volume in the last hour = tape is alive
    min_turnover_24h=0.05,
    max_turnover_24h=10.0,  # >10x daily churn = wash/churn, not accumulation
    min_holder_count=25,
    max_holder_count=250,
    min_buys_24h=15,
    min_buy_sell_ratio_1h=1.15,  # needs a real buying edge, not balanced flow
    max_top_holder_pct=12.0,
    max_top10_holder_pct=70.0,
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
    min_liquidity_to_mcap_pct=8.0,
    min_volume_24h_usd=100_000.0,
    min_volume_1h_share=0.02,
    min_turnover_24h=0.10,
    max_turnover_24h=8.0,
    min_holder_count=400,
    max_holder_count=4_000,
    max_top_holder_pct=8.0,
    max_top10_holder_pct=50.0,
    max_dev_wallet_pct=5.0,
    min_buy_sell_ratio_1h=1.2,  # 1h edge on top of the 24h buys>sells requirement
    max_price_change_1h_pct=40.0,  # don't chase the vertical candle
    max_price_change_24h_pct=150.0,
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
    min_liquidity_to_mcap_pct=10.0,
    min_volume_24h_usd=1_000_000.0,
    max_turnover_24h=5.0,
    min_holder_count=3_000,
    max_top10_holder_pct=40.0,
    min_buy_sell_ratio_1h=1.1,
    max_price_change_1h_pct=30.0,  # established coins shouldn't be vertical
    max_price_change_24h_pct=100.0,
    # Soft-safety flags are intentionally OFF for established large-caps: RugCheck
    # reports low/zero lpLockedPct for migrated Raydium tokens (LP burned into the
    # pool, not held in a locker) and Jupiter's verified list is curated/narrow, so
    # requiring them here vetoes legitimate blue-chips (BONK/JUP-class). Universal
    # safety (mint/freeze/honeypot/blacklist) still gates real risk.
    require_verified=False,
    require_lp_locked=False,
)

# ── Risk-on filters ─────────────────────────────────────────────────
# These deliberately trade safety margin for earliness. Universal safety
# (honeypot / mint / freeze / blacklist / tax ceiling) still applies — risk
# here means volatility and earliness, not scams.

DEGEN_LAUNCH = FilterThresholds(
    min_market_cap_usd=500.0,
    max_market_cap_usd=30_000.0,  # below low_cap_alpha's floor — the true trenches
    max_age_minutes=45.0,  # minutes old, not hours
    min_liquidity_usd=500.0,
    min_liquidity_to_mcap_pct=3.0,
    min_volume_24h_usd=1_000.0,
    min_volume_1h_share=0.08,  # tape must be HOT right now
    min_turnover_24h=0.10,
    max_turnover_24h=30.0,  # degen churn is normal — don't mistake it for wash
    min_holder_count=10,
    max_holder_count=500,
    min_buys_24h=10,
    min_buy_sell_ratio_1h=1.5,  # strong early buy edge or nothing
    max_top_holder_pct=20.0,
    max_top10_holder_pct=85.0,
    max_dev_wallet_pct=15.0,
    max_bond_progress_pct=60.0,
    max_sniper_pct=30.0,
    max_bundle_pct=25.0,
    # No chase guard: vertical IS the degen thesis.
    # require_verified off: trench launches are never on curated lists.
)

VOLATILITY_BREAKOUT = FilterThresholds(
    min_market_cap_usd=30_000.0,
    max_market_cap_usd=2_000_000.0,
    max_age_minutes=48 * 60.0,
    min_liquidity_usd=10_000.0,
    min_liquidity_to_mcap_pct=5.0,
    min_volume_24h_usd=50_000.0,
    min_volume_1h_share=0.10,  # the move is happening NOW
    min_turnover_24h=0.20,
    max_turnover_24h=15.0,
    min_holder_count=100,
    min_buys_24h=50,
    min_buy_sell_ratio_1h=1.3,  # buy-driven move, not a short-squeeze wick
    max_top_holder_pct=15.0,
    max_top10_holder_pct=70.0,
    max_dev_wallet_pct=10.0,
    # The inverted chase guard: REQUIRE the vertical move other filters reject.
    min_price_change_1h_pct=40.0,
    max_price_change_1h_pct=400.0,  # beyond this it's one wick, not a trend
    require_buys_exceed_sells=True,
)

VOLUME_SURGE = FilterThresholds(
    # The higher-cap volume trade: established tokens ($2M–$25M) with violent
    # volume turnover — the gap between volatility_breakout's $2M ceiling and
    # high_cap's refusal to touch anything printing +30%/1h. Thesis: when half
    # the market cap changes hands in a day on buy-leaning flow with deep
    # liquidity, the tape can absorb real size and the move has room to run.
    # Volume is the signal here, not the price move — the token may be breaking
    # out or basing into the flow, so there is no minimum 1h change, only a
    # terminal-wick guard on the top end.
    min_market_cap_usd=2_000_000.0,
    max_market_cap_usd=25_000_000.0,
    min_age_minutes=12 * 60.0,  # established, not a fresh launch
    min_liquidity_usd=150_000.0,
    min_liquidity_to_mcap_pct=5.0,
    min_volume_24h_usd=2_000_000.0,  # real money moving
    min_volume_1h_share=0.05,  # tape alive right now
    min_turnover_24h=0.5,  # the core metric: >=50% of mcap traded today
    max_turnover_24h=20.0,  # volume coins churn; cap only the absurd
    min_holder_count=2_000,
    min_buys_24h=500,
    # Flow must not be sell-dominated; the 24h buy edge below is the real
    # accumulation signal (per-pair 1h flow splits across venues, so this stays
    # at parity rather than demanding a strong edge).
    min_buy_sell_ratio_1h=1.0,
    require_buys_exceed_sells=True,
    # Relaxed vs mid_cap's 8%: at $2M+ the #1 holder is often an exchange
    # omnibus or LP-adjacent wallet, not a single actor. Concentration risk is
    # carried by the top-10 cap instead. (Gate-tracker data will validate.)
    max_top_holder_pct=15.0,
    max_top10_holder_pct=50.0,
    max_dev_wallet_pct=5.0,
    max_price_change_1h_pct=80.0,  # heat is fine; the terminal wick is not
    # Same reasoning as HIGH_CAP: migrated Raydium LPs report ~0 locked and the
    # verified list is curated/narrow — universal safety still gates real risk.
    require_verified=False,
    require_lp_locked=False,
)

GRADUATION_WATCH = FilterThresholds(
    # The pre-graduation window: token sits at 50-85% of the pump.fun bonding
    # curve with fresh SOL flowing in. The graduation pump hasn't happened yet
    # — this is the "catch it lower" filter the DexScreener-momentum filters
    # structurally miss.
    min_market_cap_usd=5_000.0,
    max_market_cap_usd=150_000.0,
    max_age_minutes=240.0,  # graduation plays are fast; older = stalled
    # No Dex liquidity gate by design: DexScreener reports liquidity.usd=None
    # for pre-graduation pump.fun pairs (verified live), so our snapshot reads
    # 0.0 and any min_liquidity_usd would make this filter unpassable. The
    # 50% bond-progress floor below already implies ~42.5 SOL of real curve
    # liquidity, which is what the gate was trying to ensure.
    min_volume_24h_usd=5_000.0,
    min_holder_count=20,
    min_buys_24h=20,
    min_buy_sell_ratio_1h=1.1,  # tape must lean buy; the curve inflow is the real signal
    max_top_holder_pct=30.0,  # looser: pre-graduation distribution is raw
    min_bond_progress_pct=50.0,
    max_bond_progress_pct=85.0,
    min_bond_inflow_sol=0.5,  # stalled curves are the red flag for this thesis
    require_bond_data=True,  # no curve data = not a graduation play, fail
    max_sniper_pct=30.0,
    max_bundle_pct=25.0,
    # No chase guard by design: the vertical move hasn't happened yet.
    # require_verified off: pump.fun launches are never on curated lists.
)


DEFAULT_THRESHOLDS: dict[FilterName, FilterThresholds] = {
    FilterName.LOW_CAP_ALPHA: LOW_CAP_ALPHA,
    FilterName.MID_CAP_MOMENTUM: MID_CAP_MOMENTUM,
    FilterName.HIGH_CAP: HIGH_CAP,
    FilterName.DEGEN_LAUNCH: DEGEN_LAUNCH,
    FilterName.VOLATILITY_BREAKOUT: VOLATILITY_BREAKOUT,
    FilterName.VOLUME_SURGE: VOLUME_SURGE,
    FilterName.GRADUATION_WATCH: GRADUATION_WATCH,
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
        self._check_flow(snap, thr, failures, warnings)
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
        _cap(snap.top10_holder_pct, thr.max_top10_holder_pct, "Top-10 holders", fails, warns)
        _cap(snap.dev_wallet_pct, thr.max_dev_wallet_pct, "Dev wallet", fails, warns)

    # Minimum 1h txn sample before the buy/sell ratio is treated as signal.
    _MIN_1H_TXN_SAMPLE = 10

    @staticmethod
    def _check_flow(
        snap: TokenSnapshot, thr: FilterThresholds, fails: list[str], warns: list[str]
    ) -> None:
        """Fresh-momentum checks: 1h buy/sell edge, live tape, turnover,
        liquidity depth relative to mcap, and the anti-chase guard."""
        # 1h buy/sell edge (stricter and fresher than the 24h buys>sells boolean).
        if thr.min_buy_sell_ratio_1h is not None:
            total_1h = snap.txns_1h_buys + snap.txns_1h_sells
            if total_1h < FilterEngine._MIN_1H_TXN_SAMPLE:
                warns.append(f"1h txn sample too small ({total_1h})")
            else:
                ratio = snap.buy_sell_ratio_1h  # inf when sells==0 and buys>0
                if ratio is None:
                    warns.append("1h buy/sell data unavailable")
                elif ratio < thr.min_buy_sell_ratio_1h:
                    r = f"{ratio:.2f}" if ratio != float("inf") else "all-buys"
                    fails.append(f"1h buy/sell {r} < {thr.min_buy_sell_ratio_1h}")
        # Live tape: share of 24h volume printed in the last hour.
        if thr.min_volume_1h_share is not None:
            share = snap.volume_1h_share
            if share is None:
                warns.append("1h/24h volume share unavailable")
            elif share < thr.min_volume_1h_share:
                fails.append(f"1h vol share {share:.1%} < {thr.min_volume_1h_share:.0%}")
        # Turnover: 24h volume / mcap. Too high = churn/wash, too low = dead tape.
        if thr.min_turnover_24h is not None or thr.max_turnover_24h is not None:
            t = snap.turnover_24h
            if t is None:
                warns.append("turnover unavailable")
            else:
                if thr.min_turnover_24h is not None and t < thr.min_turnover_24h:
                    fails.append(f"Turnover {t:.2f}x < {thr.min_turnover_24h:.2f}x")
                if thr.max_turnover_24h is not None and t > thr.max_turnover_24h:
                    fails.append(f"Turnover {t:.1f}x > {thr.max_turnover_24h:.0f}x (churn)")
        # Liquidity depth relative to market cap.
        if thr.min_liquidity_to_mcap_pct is not None:
            if snap.market_cap_usd <= 0:
                warns.append("liquidity/mcap unavailable")
            else:
                depth_pct = snap.liquidity_to_mcap * 100.0
                if depth_pct < thr.min_liquidity_to_mcap_pct:
                    fails.append(
                        f"Liq/mcap {depth_pct:.1f}% < {thr.min_liquidity_to_mcap_pct:.0f}%"
                    )
        # Chase guard: don't enter vertical candles — unless the filter's thesis
        # IS the vertical move (min_price_change_1h_pct set), in which case the
        # minimum move is required and only absurd wicks are capped.
        if (
            thr.min_price_change_1h_pct is not None
            and snap.price_change_1h_pct < thr.min_price_change_1h_pct
        ):
            fails.append(
                f"1h change {snap.price_change_1h_pct:+.0f}% < "
                f"+{thr.min_price_change_1h_pct:.0f}% (no breakout)"
            )
        if (
            thr.max_price_change_1h_pct is not None
            and snap.price_change_1h_pct > thr.max_price_change_1h_pct
        ):
            fails.append(
                f"1h change +{snap.price_change_1h_pct:.0f}% > "
                f"+{thr.max_price_change_1h_pct:.0f}% (vertical — don't chase)"
            )
        if (
            thr.max_price_change_24h_pct is not None
            and snap.price_change_24h_pct > thr.max_price_change_24h_pct
        ):
            fails.append(
                f"24h change +{snap.price_change_24h_pct:.0f}% > "
                f"+{thr.max_price_change_24h_pct:.0f}% (vertical — don't chase)"
            )

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
        # Bond progress floor (Graduation Watch: the 50-85% window).
        # Unknown data warns rather than fails — fail-open like other extras —
        # unless the filter requires bond data (graduation_watch is meaningless
        # without it: a non-pump.fun token must not pass on market metrics alone).
        if thr.min_bond_progress_pct is not None or thr.require_bond_data:
            if snap.bond_progress_pct is None:
                msg = "bond progress unavailable"
                (fails if thr.require_bond_data else warns).append(msg)
            elif (
                thr.min_bond_progress_pct is not None
                and snap.bond_progress_pct < thr.min_bond_progress_pct
            ):
                fails.append(
                    f"Bond {snap.bond_progress_pct:.0f}% < {thr.min_bond_progress_pct:.0f}%"
                )
        # Curve velocity: fresh SOL must be flowing in (stalled curves are the
        # graduation play's red flag). Unknown on first sighting -> warn only.
        if thr.min_bond_inflow_sol is not None:
            if snap.bond_inflow_sol is None:
                warns.append("bond inflow unavailable")
            elif snap.bond_inflow_sol < thr.min_bond_inflow_sol:
                fails.append(
                    f"Bond inflow {snap.bond_inflow_sol:.1f} SOL "
                    f"< {thr.min_bond_inflow_sol:.1f} SOL"
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
