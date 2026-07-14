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
        age_minutes=15,
        holder_count=100,
        txns_24h_buys=50,
        txns_24h_sells=10,
        top_holder_pct=8.0,
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
        age_minutes=60 * 24,
        holder_count=1_000,
        txns_24h_buys=100,
        txns_24h_sells=50,
        top_holder_pct=6.0,
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
        age_minutes=60 * 24 * 10,
        holder_count=5_000,
        txns_24h_buys=400,
        txns_24h_sells=300,
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
