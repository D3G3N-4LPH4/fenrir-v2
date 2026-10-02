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
  - VOLUME_SURGE      — higher-cap coins whose tape is accelerating into a move.
  - GRADUATION_WATCH  — pump.fun curves at 50-85% with fresh SOL inflow.
  - MOMENTUM_TRANSITION — pre-run acceleration (txn/holder/buy-edge growth).
  - CURVE_IGNITION    — earliest on-chain entry, curve at 10-50% with inflow.
  - FLUSH_RECOVERY    — post -60%..-95% flush, stabilized, buyers returning.
  - SECOND_LIFE       — the SAPLING model: a coin that survived days with its
    floor intact, now re-igniting (volume/edge/price lifting off its own base).

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
    MOMENTUM_TRANSITION = "momentum_transition"
    CURVE_IGNITION = "curve_ignition"
    FLUSH_RECOVERY = "flush_recovery"
    SECOND_LIFE = "second_life"


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
    # Buy-edge ceiling: an extreme ratio on a vertical move is one-sided flow —
    # painters in, no profit-takers yet — which reads as distribution, not
    # accumulation. None = dimension not checked.
    max_buy_sell_ratio_1h: float | None = None
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
    # 24h change floor: REQUIRE the change to be at least this (mirrors the 1h
    # min). Used by flush_recovery — the flush must not be a total death
    # (e.g. -95% floor rejects coins that went to zero and stayed there).
    min_price_change_24h_pct: float | None = None
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
    # Deployer-cluster cap (bundle_check, 2026-09-30): fail when the share of
    # supply that flowed through deployer-linked wallets exceeds this.
    # None = dimension not checked. Set on the risk-on filters only.
    max_deployer_cluster_pct: float | None = None
    # Fail closed when no distribution data exists at all (no top-holder /
    # top-10 / bundle / sniper figures). The 2026-10-01 volatility_breakout
    # blowups (s/acc, SIC, JANE) all cleared while every distribution field
    # was "unavailable", so the concentration caps were dead code. Set on
    # volatility_breakout only: a vertical move with invisible holders is
    # exactly the shape that just rugged four times. Every other filter keeps
    # the historical warn-and-pass behavior.
    require_distribution_known: bool = False
    # Acceleration (poll-over-poll, attached by the scout's AccelTracker).
    # None = dimension not checked. require_accel_history fails the filter when
    # there is no prior poll — the filter only fires from the second sighting,
    # which is exactly the pre-run window it is designed for.
    require_accel_history: bool = False
    min_txn_accel_1h: float | None = None  # 1h txn count growth vs previous poll
    min_holder_growth: float | None = None  # holder count growth vs previous poll
    min_buy_edge_delta: float | None = None  # 1h buy-pressure improvement vs previous poll
    # Boolean requirements
    require_buys_exceed_sells: bool = False
    require_migrated_or_bond: bool = False  # migrated OR bond >= min_bond_progress_pct
    require_verified: bool = False  # soft: warn when unverifiable
    require_lp_locked: bool = False
    # LP-lock hardening (ATM lesson, 2026-09-30): for young launches an UNKNOWN
    # LP lock is where LP-pull rugs hide. When set, a coin younger than this
    # many minutes with unknown LP lock FAILS instead of warning. Pre-migration
    # (liquidity still in the bonding curve — nothing lockable yet) keeps the
    # old warn behavior. None = keep the legacy warn-on-unknown behavior.
    require_lp_lock_known_max_age_m: float | None = None
    # Second-life (SAPLING model, 2026-10-01): re-ignition off a survived base.
    # Baseline fields are attached by fenrir.discovery.second_life from trailing
    # GeckoTerminal candles. require_second_life_baseline fails closed when the
    # baseline is missing — without the coin's own history there is no base to
    # measure the breakout against.
    require_second_life_baseline: bool = False
    min_reignition_volume_x: float | None = None  # 1h volume vs median 1h baseline
    min_price_vs_floor_x: float | None = None  # price vs trailing floor
    min_floor_vs_max_pct: float | None = None  # floor >= x% of trailing max (survival)


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
    max_deployer_cluster_pct=30.0,  # bundle/deployer check (2026-09-30)
    require_verified=True,
    # ATM (2026-09-30): passed with unknown LP lock, rugged -92.5% on an LP
    # pull 30m later. Young + migrated + unknown lock now fails closed.
    require_lp_lock_known_max_age_m=120.0,
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
    max_deployer_cluster_pct=30.0,  # bundle/deployer check (2026-09-30)
    # No chase guard: vertical IS the degen thesis.
    # require_verified off: trench launches are never on curated lists.
    # Young + unknown LP lock fails closed (ATM lesson, 2026-09-30).
    require_lp_lock_known_max_age_m=120.0,
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
    min_buy_sell_ratio_1h=2.0,  # 2026-10-01: 1.3x is tape noise, not a buy-driven
    # move (HIHI 1.54x died). Real vertical demand shows a clear edge.
    max_buy_sell_ratio_1h=8.0,  # 2026-10-01: >8x buys on a vertical move is
    # one-sided flow — painters in, no profit-takers yet (SIC 11.67x, JANE
    # 9.35x both died). The marginal buyer is already in: distribution shape.
    max_top_holder_pct=15.0,
    max_top10_holder_pct=70.0,
    max_dev_wallet_pct=10.0,
    # The inverted chase guard: REQUIRE the vertical move other filters reject.
    min_price_change_1h_pct=40.0,
    max_price_change_1h_pct=120.0,  # 2026-10-01: was 400. Past ~+120%/1h the
    # move is the exit, not the entry (JANE +144%, SuperCali +243% both topped).
    require_buys_exceed_sells=True,
    max_bundle_pct=30.0,  # bundle/deployer check (2026-09-30)
    max_deployer_cluster_pct=30.0,
    # 2026-10-01: the blowups (s/acc, SIC, JANE) all cleared with every
    # distribution field "unavailable". Fail closed — a vertical Solana move
    # with invisible holders is the exact shape that rugged. The Solana
    # forensics leg (solana_forensics.py) supplies top-10 from direct RPC, so
    # this only bites when the chain read itself fails.
    require_distribution_known=True,
    # Young + unknown LP lock fails closed (ATM lesson, 2026-09-30).
    require_lp_lock_known_max_age_m=120.0,
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


MOMENTUM_TRANSITION = FilterThresholds(
    # The pre-run catcher: fills the mcap gap between low_cap_alpha ($75k) and
    # volume_surge ($2M) for coins whose tape is *accelerating into* a move.
    # Static gates see "buy/sell 1.4"; this filter sees "buy/sell went 0.9 →
    # 1.4 while 1h activity grew 50%+ and holders grew 40%+". It only fires from
    # the second scout sighting on (require_accel_history) — the first sighting
    # seeds the baseline, so manual one-shot evaluations always fail it closed.
    min_market_cap_usd=75_000.0,
    max_market_cap_usd=2_000_000.0,
    min_age_minutes=15.0,  # skip the literal first seconds of a launch
    max_age_minutes=7 * 24 * 60.0,
    min_liquidity_usd=30_000.0,
    min_liquidity_to_mcap_pct=5.0,
    min_volume_24h_usd=100_000.0,
    min_volume_1h_share=0.03,  # tape alive right now
    min_turnover_24h=0.10,
    max_turnover_24h=15.0,  # >15x daily churn = wash, not accumulation
    min_buy_sell_ratio_1h=1.2,  # real edge right now, not balanced flow
    min_price_change_1h_pct=-10.0,  # moving up, not dumping into the bid
    max_price_change_1h_pct=100.0,  # pre-vertical: the big move hasn't happened yet
    max_top_holder_pct=20.0,  # concentration still matters at thin liquidity
    max_top10_holder_pct=75.0,
    require_accel_history=True,
    min_txn_accel_1h=1.5,  # 1h activity up 50%+ since the last poll
    min_holder_growth=1.4,  # holders up 40%+ since the last poll
    min_buy_edge_delta=0.02,  # buy pressure improving, not just high
    max_bundle_pct=30.0,  # bundle/deployer check (2026-09-30)
    max_deployer_cluster_pct=30.0,
    # Young + unknown LP lock fails closed (ATM lesson, 2026-09-30).
    require_lp_lock_known_max_age_m=120.0,
)


CURVE_IGNITION = FilterThresholds(
    # The earliest on-chain entry: a pump.fun curve at 10-50% progress with
    # strong SOL inflow velocity. This is pre-DexScreener — the ignition, not
    # the graduation. Most curves die here, so the inflow bar is high and the
    # sniper/bundle caps stay strict; inflow velocity is the signal, everything
    # else is risk control. No chase guard by design: at <50% progress the
    # vertical move hasn't happened yet.
    min_market_cap_usd=3_000.0,
    max_market_cap_usd=150_000.0,
    max_age_minutes=90.0,  # ignition is fast; older + still <50% = stalled
    min_volume_24h_usd=3_000.0,
    min_volume_1h_share=0.05,  # tape must be alive right now
    min_holder_count=10,  # earlier than graduation_watch's 20
    min_buys_24h=10,
    min_buy_sell_ratio_1h=1.1,  # tape must lean buy
    min_bond_progress_pct=10.0,
    max_bond_progress_pct=50.0,
    min_bond_inflow_sol=1.0,  # the core signal: SOL pouring into the curve
    require_bond_data=True,  # no curve data = not an ignition play, fail
    max_top_holder_pct=25.0,  # raw early distribution
    max_sniper_pct=25.0,
    max_bundle_pct=20.0,
    max_deployer_cluster_pct=30.0,  # bundle/deployer check (2026-09-30)
    require_verified=False,  # pump.fun launches are never on curated lists
    # Young + unknown LP lock fails closed (ATM lesson, 2026-09-30).
    # (Mostly moot here: curve tokens are pre-migration → warn path.)
    require_lp_lock_known_max_age_m=120.0,
)


FLUSH_RECOVERY = FilterThresholds(
    # The kioto $100M-runner structure: a coin that flushed -60%..-95% in 24h,
    # stabilized on the short windows, with buyers stepping back in and the
    # holder base intact (supply migrated to strong hands). Entered for the
    # second leg, not the flush itself.
    min_market_cap_usd=500_000.0,  # post-flush coins are bigger; flushes need size
    max_market_cap_usd=100_000_000.0,
    min_age_minutes=360.0,  # the flush leg takes time to play out
    min_liquidity_usd=50_000.0,
    min_liquidity_to_mcap_pct=3.0,  # thin sell side is expected post-flush
    min_volume_24h_usd=200_000.0,  # still a real coin
    min_volume_1h_share=0.03,  # tape alive right now
    min_turnover_24h=0.10,
    max_turnover_24h=15.0,  # >15x daily churn = wash, not accumulation
    min_buy_sell_ratio_1h=1.05,  # buyers stepping back in at the lows
    min_price_change_24h_pct=-95.0,  # flushed, not dead
    max_price_change_24h_pct=-60.0,  # negative cap = REQUIRE the flush
    min_price_change_1h_pct=-10.0,  # the knife has stopped…
    max_price_change_1h_pct=50.0,  # …but the second leg hasn't gone vertical yet
    max_top_holder_pct=25.0,  # supply migrated, not concentrated
    max_top10_holder_pct=70.0,
)


SECOND_LIFE = FilterThresholds(
    # The SAPLING model (2026-10-01): d3g3n's 20x. At his $183k entry it looked
    # identical to every dead launch; the discriminating information was time —
    # it held a floor for 5+ days and kept its holders, then re-ignited. This
    # filter buys the base breakout, not the launch: old enough to have a
    # history, floor intact, and 1h tape violently above its own baseline.
    min_age_minutes=3 * 24 * 60.0,  # survived — the actual discriminator
    min_liquidity_usd=20_000.0,  # still a real pool
    min_holder_count=200,  # community didn't evaporate
    min_buys_24h=50,
    min_buy_sell_ratio_1h=1.5,  # buyers back in control…
    max_buy_sell_ratio_1h=8.0,  # …but not one-sided painter flow (vb lesson)
    min_price_change_1h_pct=15.0,  # the re-ignition is real
    max_price_change_1h_pct=400.0,  # base breakout, not the top
    max_top_holder_pct=30.0,
    max_top10_holder_pct=80.0,
    require_buys_exceed_sells=True,
    # Baseline from trailing candles (second_life.py). Fail closed without it:
    # no history, no base, no breakout to measure.
    require_second_life_baseline=True,
    min_floor_vs_max_pct=5.0,  # never went to zero vs its own range
    min_price_vs_floor_x=1.5,  # lifting off the defended base
    min_reignition_volume_x=3.0,  # 1h volume 3x its own median
)


DEFAULT_THRESHOLDS: dict[FilterName, FilterThresholds] = {
    FilterName.LOW_CAP_ALPHA: LOW_CAP_ALPHA,
    FilterName.MID_CAP_MOMENTUM: MID_CAP_MOMENTUM,
    FilterName.HIGH_CAP: HIGH_CAP,
    FilterName.DEGEN_LAUNCH: DEGEN_LAUNCH,
    FilterName.VOLATILITY_BREAKOUT: VOLATILITY_BREAKOUT,
    FilterName.VOLUME_SURGE: VOLUME_SURGE,
    FilterName.GRADUATION_WATCH: GRADUATION_WATCH,
    FilterName.MOMENTUM_TRANSITION: MOMENTUM_TRANSITION,
    FilterName.CURVE_IGNITION: CURVE_IGNITION,
    FilterName.FLUSH_RECOVERY: FLUSH_RECOVERY,
    FilterName.SECOND_LIFE: SECOND_LIFE,
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
        self._check_acceleration(snap, thr, failures, warnings)
        self._check_second_life(snap, thr, failures, warnings)
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
        if thr.require_distribution_known:
            known = (
                snap.top_holder_pct is not None
                or snap.top10_holder_pct is not None
                or snap.bundle_pct is not None
                or snap.safety.bundled_supply_pct is not None
                or snap.sniper_pct is not None
            )
            if not known:
                fails.append("distribution unknown — no holder/bundle/sniper data (fail closed)")

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
        # Buy-edge ceiling (same sample gating as the floor): an extreme ratio
        # on a vertical move means one-sided flow — the marginal buyer is
        # already in, which is the shape of a painted top, not a breakout.
        if thr.max_buy_sell_ratio_1h is not None:
            total_1h = snap.txns_1h_buys + snap.txns_1h_sells
            if total_1h < FilterEngine._MIN_1H_TXN_SAMPLE:
                warns.append(f"1h txn sample too small ({total_1h})")
            else:
                ratio = snap.buy_sell_ratio_1h  # inf when sells==0 and buys>0
                if ratio is None:
                    warns.append("1h buy/sell data unavailable")
                elif ratio == float("inf"):
                    fails.append("1h buy/sell all-buys (no sells) — one-sided flow")
                elif ratio > thr.max_buy_sell_ratio_1h:
                    fails.append(
                        f"1h buy/sell {ratio:.2f} > {thr.max_buy_sell_ratio_1h} (one-sided)"
                    )
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
            if thr.max_price_change_24h_pct < 0:
                # Negative cap: the filter REQUIRES a deep flush (flush_recovery).
                fails.append(
                    f"24h change {snap.price_change_24h_pct:+.0f}% — not a flush "
                    f"(want ≤ {thr.max_price_change_24h_pct:.0f}%)"
                )
            else:
                fails.append(
                    f"24h change +{snap.price_change_24h_pct:.0f}% > "
                    f"+{thr.max_price_change_24h_pct:.0f}% (vertical — don't chase)"
                )
        if (
            thr.min_price_change_24h_pct is not None
            and snap.price_change_24h_pct < thr.min_price_change_24h_pct
        ):
            fails.append(
                f"24h change {snap.price_change_24h_pct:+.0f}% < "
                f"{thr.min_price_change_24h_pct:.0f}% (dead, not a flush)"
            )

    @staticmethod
    def _check_acceleration(
        snap: TokenSnapshot, thr: FilterThresholds, fails: list[str], warns: list[str]
    ) -> None:
        """Poll-over-poll acceleration: the pre-run signature.

        Metrics are attached to the snapshot by the scout's AccelTracker. The
        txn-growth dimension fails closed (activity data always exists); holder
        growth and edge delta only warn when the underlying data is missing,
        since holder/edge coverage varies by chain and provider.
        """
        if (
            not thr.require_accel_history
            and thr.min_txn_accel_1h is None
            and thr.min_holder_growth is None
            and thr.min_buy_edge_delta is None
        ):
            return
        if thr.require_accel_history and (snap.accel_polls_seen or 0) < 2:
            fails.append(f"no acceleration history (polls seen: {snap.accel_polls_seen or 0})")
            return
        if thr.min_txn_accel_1h is not None:
            g = snap.accel_txn_growth
            if g is None:
                fails.append("txn acceleration unavailable")
            elif g < thr.min_txn_accel_1h:
                fails.append(f"1h txn growth {g:.2f}x < {thr.min_txn_accel_1h:.2f}x")
        if thr.min_holder_growth is not None:
            g = snap.accel_holder_growth
            if g is None:
                warns.append("holder growth unavailable")
            elif g < thr.min_holder_growth:
                fails.append(f"holder growth {g:.2f}x < {thr.min_holder_growth:.2f}x")
        if thr.min_buy_edge_delta is not None:
            d = snap.accel_edge_delta
            if d is None:
                warns.append("buy-edge delta unavailable")
            elif d < thr.min_buy_edge_delta:
                fails.append(f"buy-edge delta {d:+.3f} < {thr.min_buy_edge_delta:+.3f}")

    @staticmethod
    def _check_second_life(
        snap: TokenSnapshot, thr: FilterThresholds, fails: list[str], warns: list[str]
    ) -> None:
        """Second-life re-ignition: the SAPLING model.

        Baseline (floor price, trailing max, median 1h volume) is attached by
        fenrir.discovery.second_life from trailing GeckoTerminal candles. The
        filter fails closed when the baseline is missing — without the coin's
        own history there is no base to measure the breakout against.
        """
        if (
            not thr.require_second_life_baseline
            and thr.min_reignition_volume_x is None
            and thr.min_price_vs_floor_x is None
            and thr.min_floor_vs_max_pct is None
        ):
            return
        floor = snap.base_floor_price_usd
        base_max = snap.base_max_price_usd
        base_vol = snap.base_median_1h_volume_usd
        if thr.require_second_life_baseline and (
            floor is None or base_max is None or base_vol is None
        ):
            fails.append("second-life baseline unknown (no trailing history)")
            return
        if floor is None or base_max is None or base_vol is None:
            return
        if thr.min_floor_vs_max_pct is not None and base_max > 0:
            ratio = floor / base_max * 100.0
            if ratio < thr.min_floor_vs_max_pct:
                fails.append(
                    f"floor broken: floor ${floor:.6f} < {thr.min_floor_vs_max_pct:.0f}% "
                    f"of trailing max ${base_max:.6f}"
                )
        if thr.min_price_vs_floor_x is not None and floor > 0:
            mult = snap.price_usd / floor
            if mult < thr.min_price_vs_floor_x:
                fails.append(
                    f"price ${snap.price_usd:.6f} only {mult:.2f}x floor ${floor:.6f} "
                    f"(need {thr.min_price_vs_floor_x:.1f}x)"
                )
        if thr.min_reignition_volume_x is not None and base_vol > 0:
            mult = snap.volume_1h_usd / base_vol
            if mult < thr.min_reignition_volume_x:
                fails.append(
                    f"1h volume ${snap.volume_1h_usd:,.0f} only {mult:.2f}x "
                    f"baseline ${base_vol:,.0f} (need {thr.min_reignition_volume_x:.1f}x)"
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
        if thr.require_lp_lock_known_max_age_m is not None:
            # ATM lesson (2026-09-30): a young migrated launch with UNKNOWN LP
            # lock is where LP-pull rugs hide — fail closed instead of
            # warn-and-pass. Pre-migration (liquidity still in the bonding
            # curve; nothing lockable yet) keeps the old warn behavior.
            if _is_pre_migration(snap):
                warns.append("pre-migration: no LP to lock yet")
            elif snap.safety.lp_locked_or_burned is None:
                if snap.age_minutes < thr.require_lp_lock_known_max_age_m:
                    fails.append(
                        f"LP lock status unknown on {snap.age_minutes:.0f}m-old coin "
                        "(young-coin LP-pull risk)"
                    )
                else:
                    warns.append("LP lock status unknown")
            elif snap.safety.lp_locked_or_burned is False:
                fails.append("LP not locked/burned")
        # Bundle / deployer-cluster check (2026-09-30, Bubblemaps bands):
        # >30% coordinated supply fails the risk-on filters; 5-15% bundled is
        # a moderate warn; unknown stays fail-open (warn only).
        _cap(
            snap.safety.deployer_cluster_pct,
            thr.max_deployer_cluster_pct,
            "Deployer cluster",
            fails,
            warns,
        )
        bundled = snap.safety.bundled_supply_pct
        if bundled is not None and 5.0 <= bundled < 15.0:
            warns.append(f"Bundled launch buys {bundled:.1f}% of supply (moderate)")
        dcluster = snap.safety.deployer_cluster_pct
        if dcluster is not None and 15.0 <= dcluster < 30.0:
            warns.append(f"Deployer-linked wallets moved {dcluster:.1f}% of supply (elevated)")
        dhold = snap.safety.deployer_holding_pct
        if dhold is not None and 1.0 <= dhold <= 5.0:
            warns.append(f"Deployer still holds {dhold:.1f}% (overhang)")
        if snap.safety.deployer_funder_is_serial_launcher:
            warns.append("Deployer/funder is a known serial launcher (behavior risk)")
        elif thr.require_lp_locked:
            if snap.migrated is False:
                warns.append("pre-migration: no LP to lock yet")
            elif snap.safety.lp_locked_or_burned is None:
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
            s.lp_locked_or_burned is True or snap.migrated is False,
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


def _is_pre_migration(snap: TokenSnapshot) -> bool:
    """True when liquidity still sits in a bonding curve (no lockable LP yet).

    ``migrated=False`` is the explicit signal (DexScreener sets it for pump.fun
    curve venues); a sub-100% bond progress means the same thing when the
    migrated flag is absent.
    """
    if snap.migrated is False:
        return True
    return snap.bond_progress_pct is not None and snap.bond_progress_pct < 100.0


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
