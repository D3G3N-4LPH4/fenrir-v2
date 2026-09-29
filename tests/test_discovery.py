#!/usr/bin/env python3
"""
FENRIR - Discovery foundation tests

Covers the chain-agnostic engine (models, filters, scoring) and the Solana adapter
pure mappers. No network — adapters are exercised only via their pure functions.
"""

from __future__ import annotations

from fenrir.discovery.chains.solana import map_rugcheck_summary, snapshot_from_jupiter
from fenrir.discovery.filters import FilterEngine, FilterName, UniversalSafety
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.discovery.scoring import ScoringEngine, ScoringWeights


# All-good safety so the universal gate + require_verified/lp pass by default.
def _safe() -> SafetySignals:
    return SafetySignals(
        mint_disabled=True,
        freeze_disabled=True,
        lp_locked_or_burned=True,
        contract_verified=True,
        blacklist_present=False,
        honeypot=False,
    )


def _low_cap_pass() -> TokenSnapshot:
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address="LOW",
        market_cap_usd=20_000,
        liquidity_usd=5_000,
        volume_24h_usd=10_000,
        volume_1h_usd=2_000,
        age_minutes=15,
        holder_count=100,
        txns_24h_buys=50,
        txns_24h_sells=10,
        txns_1h_buys=30,
        txns_1h_sells=10,
        price_change_1h_pct=10.0,
        price_change_24h_pct=25.0,
        top_holder_pct=8.0,
        top10_holder_pct=40.0,
        dev_wallet_pct=5.0,
        bond_progress_pct=20.0,
        sniper_pct=10.0,
        bundle_pct=8.0,
        safety=_safe(),
    )


def _mid_cap_pass() -> TokenSnapshot:
    return TokenSnapshot(
        chain=Chain.SOLANA,
        token_address="MID",
        market_cap_usd=200_000,
        liquidity_usd=50_000,
        volume_24h_usd=200_000,
        volume_1h_usd=10_000,
        age_minutes=60 * 24,
        holder_count=1_000,
        txns_24h_buys=100,
        txns_24h_sells=50,
        txns_1h_buys=40,
        txns_1h_sells=20,
        price_change_1h_pct=5.0,
        price_change_24h_pct=30.0,
        top_holder_pct=6.0,
        top10_holder_pct=30.0,
        dev_wallet_pct=4.0,
        migrated=True,
        safety=_safe(),
    )


def _high_cap_pass() -> TokenSnapshot:
    return TokenSnapshot(
        chain=Chain.ETHEREUM,
        token_address="HIGH",
        market_cap_usd=5_000_000,
        liquidity_usd=500_000,
        volume_24h_usd=2_000_000,
        volume_1h_usd=100_000,
        age_minutes=60 * 24 * 10,
        holder_count=5_000,
        txns_24h_buys=400,
        txns_24h_sells=300,
        txns_1h_buys=60,
        txns_1h_sells=40,
        price_change_1h_pct=3.0,
        price_change_24h_pct=15.0,
        top10_holder_pct=25.0,
        safety=_safe(),
    )


# ── Models ────────────────────────────────────────────────────────────


class TestModels:
    def test_buy_pressure_and_ratios(self) -> None:
        s = TokenSnapshot(chain=Chain.SOLANA, token_address="X", txns_24h_buys=3, txns_24h_sells=1)
        assert s.buys_exceed_sells is True
        assert s.buy_pressure_24h == 0.75
        empty = TokenSnapshot(chain=Chain.SOLANA, token_address="Y")
        assert empty.buy_pressure_24h == 0.5  # neutral when no txns

    def test_liquidity_to_mcap(self) -> None:
        s = TokenSnapshot(
            chain=Chain.SOLANA, token_address="X", liquidity_usd=10_000, market_cap_usd=100_000
        )
        assert s.liquidity_to_mcap == 0.1

    def test_chain_from_dexscreener(self) -> None:
        assert Chain.from_dexscreener("bsc") is Chain.BNB
        assert Chain.from_dexscreener("base") is Chain.BASE
        assert Chain.from_dexscreener("solana") is Chain.SOLANA
        assert Chain.from_dexscreener("polygon") is None


# ── Filters ───────────────────────────────────────────────────────────


class TestFilters:
    def setup_method(self) -> None:
        self.engine = FilterEngine()

    def test_each_filter_passes_its_ideal_token(self) -> None:
        assert self.engine.evaluate(_low_cap_pass(), FilterName.LOW_CAP_ALPHA).passed
        assert self.engine.evaluate(_mid_cap_pass(), FilterName.MID_CAP_MOMENTUM).passed
        assert self.engine.evaluate(_high_cap_pass(), FilterName.HIGH_CAP).passed

    def test_low_cap_market_cap_bounds(self) -> None:
        below = _low_cap_pass()
        below.market_cap_usd = 2_999
        assert not self.engine.evaluate(below, FilterName.LOW_CAP_ALPHA).passed
        above = _low_cap_pass()
        above.market_cap_usd = 75_001
        assert not self.engine.evaluate(above, FilterName.LOW_CAP_ALPHA).passed

    def test_low_cap_age_holder_buys_bounds(self) -> None:
        old = _low_cap_pass()
        old.age_minutes = 121
        assert not self.engine.evaluate(old, FilterName.LOW_CAP_ALPHA).passed
        few = _low_cap_pass()
        few.holder_count = 24
        assert not self.engine.evaluate(few, FilterName.LOW_CAP_ALPHA).passed
        many = _low_cap_pass()
        many.holder_count = 251
        assert not self.engine.evaluate(many, FilterName.LOW_CAP_ALPHA).passed
        low_buys = _low_cap_pass()
        low_buys.txns_24h_buys = 14
        assert not self.engine.evaluate(low_buys, FilterName.LOW_CAP_ALPHA).passed

    def test_low_cap_distribution_and_solana_extras(self) -> None:
        for field, bad in (
            ("top_holder_pct", 13.0),
            ("dev_wallet_pct", 11.0),
            ("bond_progress_pct", 41.0),
            ("sniper_pct", 21.0),
            ("bundle_pct", 16.0),
        ):
            snap = _low_cap_pass()
            setattr(snap, field, bad)
            res = self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA)
            assert not res.passed, f"{field}={bad} should fail"

    def test_mid_cap_requires_buys_and_migration(self) -> None:
        no_pressure = _mid_cap_pass()
        no_pressure.txns_24h_buys, no_pressure.txns_24h_sells = 40, 60
        assert not self.engine.evaluate(no_pressure, FilterName.MID_CAP_MOMENTUM).passed
        not_ready = _mid_cap_pass()
        not_ready.migrated = False
        not_ready.bond_progress_pct = 50.0  # below 65 and not migrated
        assert not self.engine.evaluate(not_ready, FilterName.MID_CAP_MOMENTUM).passed
        bonded = _mid_cap_pass()
        bonded.migrated = False
        bonded.bond_progress_pct = 80.0  # bonded → OK
        assert self.engine.evaluate(bonded, FilterName.MID_CAP_MOMENTUM).passed

    def test_high_cap_min_market_cap(self) -> None:
        small = _high_cap_pass()
        small.market_cap_usd = 900_000
        assert not self.engine.evaluate(small, FilterName.HIGH_CAP).passed

    def test_high_cap_soft_safety_flags_not_gated(self) -> None:
        # Established large-caps must NOT be vetoed for a missing verified flag or
        # unreported LP lock (RugCheck under-reports lpLockedPct for migrated Raydium
        # tokens; Jupiter's verified list is narrow). Universal safety still applies.
        snap = _high_cap_pass()
        snap.safety.contract_verified = False
        snap.safety.lp_locked_or_burned = False
        assert self.engine.evaluate(snap, FilterName.HIGH_CAP).passed
        snap.safety.contract_verified = None
        snap.safety.lp_locked_or_burned = None
        assert self.engine.evaluate(snap, FilterName.HIGH_CAP).passed
        # ...but honeypot (universal) still hard-fails.
        snap.safety.honeypot = True
        assert not self.engine.evaluate(snap, FilterName.HIGH_CAP).passed

    def test_missing_optional_field_warns_not_fails(self) -> None:
        # Unknown holder count / bond → warning, not a hard fail (fail-open).
        snap = _low_cap_pass()
        snap.holder_count = None
        snap.bond_progress_pct = None
        res = self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA)
        assert res.passed
        assert any("holder count unavailable" in w for w in res.warnings)

    def test_universal_safety_gate(self) -> None:
        bad = _low_cap_pass()
        bad.safety.mint_disabled = False  # universal + spec both require disabled
        assert not self.engine.evaluate(bad, FilterName.LOW_CAP_ALPHA).passed
        hp = _low_cap_pass()
        hp.safety.honeypot = True
        assert not self.engine.evaluate(hp, FilterName.LOW_CAP_ALPHA).passed

    def test_low_cap_passes_pre_migration_without_lp_lock(self) -> None:
        # Pre-migration launches hold liquidity in the bonding curve — no lockable
        # LP. Universal safety must NOT gate Low Cap Alpha on LP lock.
        snap = _low_cap_pass()
        snap.safety.lp_locked_or_burned = None  # unknown (pre-migration)
        assert self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA).passed
        snap.safety.lp_locked_or_burned = False  # explicitly no LP lock yet
        assert self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA).passed
        # Mid Cap still requires LP lock (per-filter).
        mid = _mid_cap_pass()
        mid.safety.lp_locked_or_burned = False
        assert not self.engine.evaluate(mid, FilterName.MID_CAP_MOMENTUM).passed

    def test_universal_safety_can_be_disabled(self) -> None:
        engine = FilterEngine(universal=UniversalSafety(enabled=False))
        snap = _low_cap_pass()
        snap.safety = SafetySignals(contract_verified=True, lp_locked_or_burned=True)
        # mint/freeze unknown, but universal off → only the filter's require_* apply.
        assert engine.evaluate(snap, FilterName.LOW_CAP_ALPHA).passed


class TestFlowChecks:
    """The 2026-09 filter hardening: 1h edge, turnover, chase guard, depth, concentration."""

    def setup_method(self) -> None:
        self.engine = FilterEngine()

    def test_balanced_1h_flow_fails_low_cap(self) -> None:
        snap = _low_cap_pass()
        snap.txns_1h_buys, snap.txns_1h_sells = 20, 20  # ratio 1.0 < 1.15
        res = self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA)
        assert not res.passed
        assert any("1h buy/sell" in f for f in res.failures)

    def test_tiny_1h_sample_warns_not_fails(self) -> None:
        snap = _low_cap_pass()
        snap.txns_1h_buys, snap.txns_1h_sells = 3, 2  # real but tiny sample
        res = self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA)
        assert res.passed
        assert any("sample too small" in w for w in res.warnings)

    def test_churn_turnover_fails(self) -> None:
        snap = _low_cap_pass()
        snap.volume_24h_usd = 300_000  # 15x turnover on $20k mcap
        snap.volume_1h_usd = 60_000  # keep the live-tape check passing
        res = self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA)
        assert not res.passed
        assert any("churn" in f for f in res.failures)

    def test_dead_tape_turnover_fails(self) -> None:
        snap = _mid_cap_pass()
        snap.volume_24h_usd = 100_000  # floor volume…
        snap.market_cap_usd = 900_000  # …on max mcap → 0.11x turnover, still ≥ 0.10
        snap.volume_1h_usd = 5_000
        snap.liquidity_usd = 80_000  # keep liq/mcap ≥ 8%
        assert self.engine.evaluate(snap, FilterName.MID_CAP_MOMENTUM).passed
        snap.volume_24h_usd = 80_000  # below floor AND turnover 0.089x
        snap.volume_1h_usd = 4_000
        res = self.engine.evaluate(snap, FilterName.MID_CAP_MOMENTUM)
        assert not res.passed
        assert any("Turnover" in f for f in res.failures)

    def test_thin_relative_liquidity_fails(self) -> None:
        snap = _mid_cap_pass()
        snap.liquidity_usd = 35_000  # absolute floor…
        snap.market_cap_usd = 900_000  # …but only 3.9% of mcap
        res = self.engine.evaluate(snap, FilterName.MID_CAP_MOMENTUM)
        assert not res.passed
        assert any("Liq/mcap" in f for f in res.failures)

    def test_chase_guard_rejects_vertical_mid_cap(self) -> None:
        snap = _mid_cap_pass()
        snap.price_change_1h_pct = 45.0
        res = self.engine.evaluate(snap, FilterName.MID_CAP_MOMENTUM)
        assert not res.passed
        assert any("vertical" in f for f in res.failures)
        snap.price_change_1h_pct = 5.0
        snap.price_change_24h_pct = 402.0
        res = self.engine.evaluate(snap, FilterName.MID_CAP_MOMENTUM)
        assert not res.passed

    def test_low_cap_exempt_from_chase_guard(self) -> None:
        # Early volatility IS the low_cap thesis — no 1h/24h vertical guard there.
        snap = _low_cap_pass()
        snap.price_change_1h_pct = 120.0
        assert self.engine.evaluate(snap, FilterName.LOW_CAP_ALPHA).passed

    def test_top10_concentration_fails(self) -> None:
        snap = _mid_cap_pass()
        snap.top10_holder_pct = 55.0  # > 50% cap
        res = self.engine.evaluate(snap, FilterName.MID_CAP_MOMENTUM)
        assert not res.passed
        assert any("Top-10" in f for f in res.failures)

    def test_distribution_metrics_excludes_infrastructure(self) -> None:
        from fenrir.discovery.providers.goplus import distribution_metrics

        holders = [
            ("0xPoolManager", 47.13, 1, 0),  # v4 PoolManager singleton (contract)
            ("0xdead", 11.2, 0, 1),  # burned (locked)
            ("0xA", 4.30, 0, 0),
            ("0xB", 3.69, 0, 0),
            ("0xUnknown", 2.0, None, None),  # unknown flags → kept (conservative)
        ]
        top, top10 = distribution_metrics(holders, {"0xunmatched"})
        assert top == 4.30
        assert abs(top10 - (4.30 + 3.69 + 2.0)) < 1e-9
        # Explicit address exclusion still works when the flag is missing.
        top2, _ = distribution_metrics([("0xPool", 47.0, None, None)], {"0xpool"})
        assert top2 is None  # everything excluded → unknown, not zero


# ── Scoring ───────────────────────────────────────────────────────────


class TestScoring:
    def setup_method(self) -> None:
        self.engine = ScoringEngine()

    def test_strong_token_scores_high(self) -> None:
        s = _mid_cap_pass()
        s.price_change_24h_pct = 40.0
        s.twitter = "t"
        s.telegram = "tg"
        b = self.engine.score(s)
        assert b.overall > 60
        assert b.safety > 80  # all-good safety
        assert 0 <= b.risk <= 100

    def test_honeypot_tanks_safety_and_overall(self) -> None:
        s = _mid_cap_pass()
        s.safety.honeypot = True
        b = self.engine.score(s)
        assert b.safety < 20
        assert b.risk >= 100 or b.risk > 80

    def test_missing_data_is_neutral_not_zero(self) -> None:
        s = TokenSnapshot(chain=Chain.SOLANA, token_address="X", market_cap_usd=100_000)
        b = self.engine.score(s)
        assert 0 < b.overall < 100  # unknowns don't zero it out

    def test_empty_safety_caps_overall(self) -> None:
        # No provider data at all: even perfect momentum can't earn confidence.
        s = _mid_cap_pass()
        s.safety = SafetySignals()  # nothing known
        assert s.safety.is_empty
        b = self.engine.score(s)
        assert b.overall <= 60.0
        # …but partial data (RugCheck-style) is not "empty" and not capped.
        s.safety = SafetySignals(
            mint_disabled=True, freeze_disabled=True, lp_locked_or_burned=True,
            risk_score=10.0, honeypot=False,
        )
        assert not s.safety.is_empty
        b2 = self.engine.score(s)
        assert b2.overall > 60.0

    def test_weights_shift_overall(self) -> None:
        s = _mid_cap_pass()
        momentum_heavy = ScoringEngine(
            ScoringWeights(momentum=1, safety=0, liquidity=0, holder=0, community=0, risk=0)
        )
        safety_heavy = ScoringEngine(
            ScoringWeights(momentum=0, safety=1, liquidity=0, holder=0, community=0, risk=0)
        )
        assert momentum_heavy.score(s).overall == momentum_heavy.score(s).momentum
        assert safety_heavy.score(s).overall == safety_heavy.score(s).safety


# ── Solana adapter pure mappers ───────────────────────────────────────


class TestSolanaMappers:
    def test_snapshot_from_jupiter(self) -> None:
        tok = {
            "id": "MINT",
            "symbol": "DOGE",
            "name": "Doge",
            "usdPrice": 0.5,
            "mcap": 400_000,
            "fdv": 500_000,
            "liquidity": 120_000,
            "holderCount": 3_000,
            "isVerified": True,
            "organicScore": 80,
            "graduatedAt": "2026-01-01",
            "twitter": "https://x.com/doge",
            "audit": {"topHoldersPercentage": 22.0},
            "stats24h": {
                "buyVolume": 100_000,
                "sellVolume": 50_000,
                "numBuys": 800,
                "numSells": 400,
                "priceChange": 12.5,
            },
        }
        s = snapshot_from_jupiter(tok)
        assert s.chain is Chain.SOLANA
        assert s.token_address == "MINT"
        assert s.market_cap_usd == 400_000
        assert s.liquidity_usd == 120_000
        assert s.holder_count == 3_000
        assert s.volume_24h_usd == 150_000
        assert s.txns_24h_buys == 800
        assert s.price_change_24h_pct == 12.5
        assert s.top_holder_pct == 22.0
        assert s.migrated is True
        assert s.safety.contract_verified is True

    def test_map_rugcheck_summary(self) -> None:
        summary = {
            "score_normalised": 8.0,
            "lpLockedPct": 95.0,
            "risks": [{"name": "Mint Authority still enabled", "level": "danger"}],
        }
        sig = map_rugcheck_summary(summary)
        assert sig.mint_disabled is False  # "mint authority" risk present
        assert sig.freeze_disabled is True  # no freeze risk
        assert sig.lp_locked_or_burned is True  # 95 >= 90
        assert sig.lp_locked_pct == 95.0
        assert sig.risk_score == 8.0
        assert "Mint Authority still enabled" in sig.risk_flags
