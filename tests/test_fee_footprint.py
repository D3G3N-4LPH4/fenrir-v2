#!/usr/bin/env python3
"""
FENRIR - fee-footprint tests (2026-10-04 fee/mcap study).

Covers:
- TokenSnapshot.fee_mcap_24h: chain fee rates, None handling
- TokenSnapshot.wash_volume: farm-shaped volume detection
- ScoringEngine._risk: wash-volume penalty
- pump_vault PDA derivation (pure, no network) + bad-mint fail-open

No network: RPC reads are exercised live via tools/pump_vault.py only.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from solders.pubkey import Pubkey

from fenrir.discovery.models import Chain, TokenSnapshot
from fenrir.discovery.pump_vault import read_creator_vault
from fenrir.discovery.scoring import ScoringEngine
from fenrir.protocol.pumpfun import PumpFunProgram


def _snap(**kw: Any) -> TokenSnapshot:
    base: dict[str, Any] = dict(
        chain=Chain.SOLANA,
        token_address="T",
        market_cap_usd=100_000,
        volume_24h_usd=100_000,
        price_change_24h_pct=0.0,
    )
    base.update(kw)
    return TokenSnapshot(**base)


class TestFeeMcap:
    def test_solana_rate(self):
        s = _snap()  # 1.0x turnover
        assert s.fee_mcap_24h == pytest.approx(0.0025)

    def test_robinhood_rate(self):
        s = _snap(chain=Chain.ROBINHOOD)
        assert s.fee_mcap_24h == pytest.approx(0.003)

    def test_none_when_no_mcap(self):
        s = _snap(market_cap_usd=0)
        assert s.turnover_24h is None
        assert s.fee_mcap_24h is None

    def test_median_ballpark(self):
        # The study's median successful coin: 0.61% fees/mcap ~= 2.4x turnover
        s = _snap(volume_24h_usd=244_000)
        assert s.fee_mcap_24h == pytest.approx(0.0061, abs=1e-4)


class TestWashVolume:
    def test_joekin_archetype_flags(self):
        # 909x turnover after a -97% collapse
        s = _snap(volume_24h_usd=90_900_000, price_change_24h_pct=-97.0)
        assert s.wash_volume is True

    def test_below_turnover_floor_no_flag(self):
        s = _snap(volume_24h_usd=4_900_000, price_change_24h_pct=-80.0)  # 49x
        assert s.wash_volume is False

    def test_healthy_runner_no_flag(self):
        # High turnover *with* a positive move is a live runner, not wash
        s = _snap(volume_24h_usd=6_000_000, price_change_24h_pct=+300.0)  # 60x
        assert s.wash_volume is False

    def test_drawdown_boundary(self):
        s = _snap(volume_24h_usd=5_000_000, price_change_24h_pct=-50.0)  # exactly 50x/-50
        assert s.wash_volume is True
        s2 = _snap(volume_24h_usd=5_000_000, price_change_24h_pct=-49.9)
        assert s2.wash_volume is False


class TestWashRiskPenalty:
    def test_penalty_applied(self):
        engine = ScoringEngine()
        clean = _snap(volume_24h_usd=100_000, price_change_24h_pct=10.0)
        wash = _snap(volume_24h_usd=90_900_000, price_change_24h_pct=-97.0)
        assert engine._risk(wash) == pytest.approx(engine._risk(clean) + 20.0)

    def test_no_penalty_for_runner(self):
        engine = ScoringEngine()
        base = _snap(volume_24h_usd=100_000, price_change_24h_pct=10.0)
        runner = _snap(volume_24h_usd=6_000_000, price_change_24h_pct=300.0)
        assert engine._risk(runner) == pytest.approx(engine._risk(base))


class TestVaultDerivation:
    def test_known_vault_pda(self):
        # Observed live 2026-10-04: SI's creator vault
        prog = PumpFunProgram()
        creator = Pubkey.from_string("Dp3dBtKAA5m9YmbVVztHP3FsygcUdoaQtZ4NSZyzQ9ha")
        vault = prog.derive_creator_vault(creator)
        assert str(vault) == "8BPhzYEbFXLQbJeKCSTm3PpBGwJUYAZGP86ReZU7ScKh"

    def test_bad_mint_returns_none_without_network(self):
        assert asyncio.run(read_creator_vault("not-a-mint")) is None
