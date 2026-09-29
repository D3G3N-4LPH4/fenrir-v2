#!/usr/bin/env python3
"""
FENRIR - Graduation Watch Test Suite

Covers the pump.fun bonding-curve provider (decode math, velocity state),
the GRADUATION_WATCH entry filter, and the alert curve line.
Network I/O is fully mocked — no RPC calls are made.

Run with: pytest tests/test_graduation_watch.py -v
"""

from __future__ import annotations

import struct
import time

from fenrir.discovery.alerts import format_scout_alert
from fenrir.discovery.filters import FilterEngine, FilterName
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.discovery.providers.pumpfun import PumpFunProvider
from fenrir.protocol.pumpfun import PumpFunProgram

TOKEN = "Grad111111111111111111111111111111111111111"


def _safe() -> SafetySignals:
    return SafetySignals(
        mint_disabled=True,
        freeze_disabled=True,
        lp_locked_or_burned=True,
        contract_verified=True,
        blacklist_present=False,
        honeypot=False,
    )


def _curve_bytes(real_sol_lamports: int, complete: bool = False) -> bytes:
    """Synthetic bonding-curve account (>=73 bytes, with creator pubkey)."""
    buf = b"\x00" * 8  # discriminator
    buf += struct.pack("<Q", 1_073_000_000_000_000)  # virtual_token_reserves
    buf += struct.pack("<Q", 30_000_000_000)  # virtual_sol_reserves
    buf += struct.pack("<Q", 793_100_000_000_000)  # real_token_reserves
    buf += struct.pack("<Q", real_sol_lamports)  # real_sol_reserves
    buf += struct.pack("<Q", 1_000_000_000_000_000)  # token_total_supply
    buf += b"\x01" if complete else b"\x00"
    buf += b"\x11" * 32  # creator pubkey
    return buf


def _grad_snapshot(**kw) -> TokenSnapshot:
    base = dict(
        chain=Chain.SOLANA,
        token_address=TOKEN,
        market_cap_usd=40_000,
        liquidity_usd=8_000,
        volume_24h_usd=20_000,
        age_minutes=60,
        holder_count=50,
        txns_24h_buys=100,
        txns_24h_sells=40,
        txns_1h_buys=60,
        txns_1h_sells=40,
        price_change_1h_pct=12.0,
        top_holder_pct=15.0,
        bond_progress_pct=60.0,
        bond_inflow_sol=2.0,
        bond_sol_remaining=34.0,
        sniper_pct=10.0,
        bundle_pct=8.0,
        safety=_safe(),
    )
    base.update(kw)
    return TokenSnapshot(**base)


# ---------------------------------------------------------------------------
# Curve decoding
# ---------------------------------------------------------------------------


class TestCurveDecode:
    def test_progress_50_pct(self):
        prog = PumpFunProgram()
        state = prog.decode_bonding_curve(_curve_bytes(42_500_000_000))
        assert state is not None
        assert state.get_migration_progress() == 50.0
        assert not state.complete

    def test_progress_85_pct(self):
        prog = PumpFunProgram()
        state = prog.decode_bonding_curve(_curve_bytes(72_250_000_000))
        assert state is not None
        assert state.get_migration_progress() == 85.0

    def test_complete_flag(self):
        prog = PumpFunProgram()
        state = prog.decode_bonding_curve(_curve_bytes(85_000_000_000, complete=True))
        assert state is not None
        assert state.complete
        assert state.get_migration_progress() == 100.0

    def test_too_short_returns_none(self):
        assert PumpFunProgram().decode_bonding_curve(b"\x00" * 10) is None

    def test_sol_to_graduation(self):
        prog = PumpFunProgram()
        state = prog.decode_bonding_curve(_curve_bytes(42_500_000_000))
        assert PumpFunProvider.sol_to_graduation(state) == 42.5


# ---------------------------------------------------------------------------
# Velocity state file
# ---------------------------------------------------------------------------


class TestVelocityState:
    def _state(self, sol: float):
        prog = PumpFunProgram()
        lamports = int(sol * 1e9)
        return prog.decode_bonding_curve(_curve_bytes(lamports))

    def test_inflow_between_readings(self, tmp_path):
        path = str(tmp_path / "curves.json")
        p = PumpFunProvider()
        t0 = time.time()
        p.record_reading(TOKEN, self._state(50.0), now=t0, path=path)
        entry = p.record_reading(TOKEN, self._state(52.5), now=t0 + 60, path=path)
        assert PumpFunProvider.inflow_sol(entry, now=t0 + 60) == 2.5

    def test_first_reading_has_no_inflow(self, tmp_path):
        path = str(tmp_path / "curves.json")
        p = PumpFunProvider()
        entry = p.record_reading(TOKEN, self._state(50.0), path=path)
        assert PumpFunProvider.inflow_sol(entry) is None

    def test_stale_reading_has_no_inflow(self, tmp_path):
        path = str(tmp_path / "curves.json")
        p = PumpFunProvider()
        t0 = time.time() - 3600
        p.record_reading(TOKEN, self._state(50.0), now=t0, path=path)
        entry = p.record_reading(TOKEN, self._state(52.5), now=t0 + 60, path=path)
        # prev_check is >30 min old -> too stale for velocity
        assert PumpFunProvider.inflow_sol(entry, now=time.time()) is None

    def test_prune(self, tmp_path):
        path = str(tmp_path / "curves.json")
        p = PumpFunProvider()
        p.record_reading(TOKEN, self._state(50.0), now=time.time() - 8 * 3600, path=path)
        assert p.prune_state(max_age_hours=6.0, path=path) == 1
        assert p._load_state(path) == {}


# ---------------------------------------------------------------------------
# GRADUATION_WATCH filter
# ---------------------------------------------------------------------------


class TestGraduationWatch:
    engine = FilterEngine()

    def test_pass(self):
        r = self.engine.evaluate(_grad_snapshot(), FilterName.GRADUATION_WATCH)
        assert r.passed, f"failures={r.failures} warnings={r.warnings}"

    def test_fail_above_85(self):
        r = self.engine.evaluate(
            _grad_snapshot(bond_progress_pct=90.0), FilterName.GRADUATION_WATCH
        )
        assert not r.passed
        assert any("Bond 90%" in f for f in r.failures)

    def test_fail_below_50(self):
        r = self.engine.evaluate(
            _grad_snapshot(bond_progress_pct=30.0), FilterName.GRADUATION_WATCH
        )
        assert not r.passed
        assert any("Bond 30%" in f for f in r.failures)

    def test_fail_stalled_inflow(self):
        r = self.engine.evaluate(
            _grad_snapshot(bond_inflow_sol=0.1), FilterName.GRADUATION_WATCH
        )
        assert not r.passed
        assert any("inflow" in f for f in r.failures)

    def test_first_sighting_warns_not_fails_on_inflow(self):
        # inflow unknown (first sighting) -> warn only, can still pass
        r = self.engine.evaluate(
            _grad_snapshot(bond_inflow_sol=None), FilterName.GRADUATION_WATCH
        )
        assert r.passed
        assert any("inflow" in w for w in r.warnings)

    def test_fail_no_bond_data(self):
        # require_bond_data=True: a non-pump.fun token must not pass
        r = self.engine.evaluate(
            _grad_snapshot(bond_progress_pct=None, bond_inflow_sol=None),
            FilterName.GRADUATION_WATCH,
        )
        assert not r.passed
        assert any("bond progress unavailable" in f for f in r.failures)

    def test_fail_robinhood_token(self):
        # A Robinhood token has no curve data -> require_bond_data fails it
        snap = _grad_snapshot(chain=Chain.ROBINHOOD,
                              bond_progress_pct=None, bond_inflow_sol=None,
                              bond_sol_remaining=None)
        r = self.engine.evaluate(snap, FilterName.GRADUATION_WATCH)
        assert not r.passed

    def test_fail_too_old(self):
        r = self.engine.evaluate(
            _grad_snapshot(age_minutes=300.0), FilterName.GRADUATION_WATCH
        )
        assert not r.passed

    def test_midcap_migrated_regression(self):
        # MID_CAP's min_bond_progress_pct=65 must still pass for migrated tokens
        snap = _grad_snapshot(
            market_cap_usd=200_000,
            liquidity_usd=40_000,
            volume_24h_usd=150_000,
            volume_1h_usd=5_000,
            age_minutes=120,
            holder_count=800,
            txns_1h_buys=200,
            txns_1h_sells=100,
            price_change_1h_pct=10.0,
            price_change_24h_pct=50.0,
            top_holder_pct=5.0,
            top10_holder_pct=30.0,
            dev_wallet_pct=3.0,
            bond_progress_pct=100.0,
            bond_inflow_sol=5.0,
            migrated=True,
        )
        r = self.engine.evaluate(snap, FilterName.MID_CAP_MOMENTUM)
        assert r.passed, f"failures={r.failures} warnings={r.warnings}"

    def test_midcap_low_bond_still_fails(self):
        snap = _grad_snapshot(bond_progress_pct=40.0, migrated=False)
        r = self.engine.evaluate(snap, FilterName.MID_CAP_MOMENTUM)
        assert not r.passed


# ---------------------------------------------------------------------------
# Alert formatting
# ---------------------------------------------------------------------------


class TestAlertCurveLine:
    def _cand(self, **kw):
        base = {
            "symbol": "TEST", "name": "Test Token", "chain": "solana",
            "source": "graduation", "address": TOKEN,
            "score": {"overall": 72.5}, "passed_filters": ["graduation_watch"],
            "market_cap_usd": 40000, "liquidity_usd": 8000,
            "volume_24h_usd": 20000, "buys_1h": 60, "sells_1h": 40,
            "price_change_1h_pct": 12.0, "age_minutes": 60,
            "bond_progress_pct": 67.0, "bond_sol_remaining": 28.0,
            "bond_inflow_sol": 2.5,
            "dexscreener": f"https://dexscreener.com/solana/{TOKEN}",
        }
        base.update(kw)
        return base

    def test_curve_line_renders(self):
        text = format_scout_alert(self._cand())
        assert "Bonding curve 67%" in text
        assert "28 SOL to graduation" in text

    def test_no_curve_line_without_data(self):
        text = format_scout_alert(self._cand(bond_progress_pct=None,
                                            bond_sol_remaining=None,
                                            bond_inflow_sol=None))
        assert "Bonding curve" not in text
