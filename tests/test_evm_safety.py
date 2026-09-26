#!/usr/bin/env python3
"""
FENRIR - EVM safety hard-gate tests (Phase 7.3, read-only)

The gate rejects honeypot / high-tax / blacklist (and optionally LP-unlocked) EVM tokens
before any strategy sees them, with a fail-open/closed policy for unknown signals. Also
its integration into the evaluator (rejected tokens produce no signals) and config wiring.
No network.
"""

from __future__ import annotations

from typing import Any

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.evm import EvmSafetyConfig, EvmSafetyGate, EvmTokenEvaluator
from fenrir.strategies.momentum import MomentumStrategy

TOKEN = "0x1234567890abcdef1234567890abcdef12345678"


def _clean() -> SafetySignals:
    return SafetySignals(
        honeypot=False,
        blacklist_present=False,
        buy_tax_pct=1.0,
        sell_tax_pct=1.0,
        lp_locked_or_burned=True,
    )


class TestGate:
    def test_clean_passes(self) -> None:
        assert EvmSafetyGate().check(_clean()).passed is True

    def test_honeypot_rejected(self) -> None:
        v = EvmSafetyGate().check(SafetySignals(honeypot=True))
        assert v.passed is False
        assert "honeypot" in v.reason

    def test_blacklist_rejected(self) -> None:
        v = EvmSafetyGate().check(SafetySignals(honeypot=False, blacklist_present=True))
        assert v.passed is False
        assert "blacklist" in v.reason

    def test_high_buy_tax_rejected(self) -> None:
        v = EvmSafetyGate().check(
            SafetySignals(honeypot=False, blacklist_present=False, buy_tax_pct=25.0)
        )
        assert v.passed is False
        assert "buy tax" in v.reason

    def test_high_sell_tax_rejected(self) -> None:
        s = SafetySignals(
            honeypot=False, blacklist_present=False, buy_tax_pct=1.0, sell_tax_pct=40.0
        )
        v = EvmSafetyGate().check(s)
        assert v.passed is False
        assert "sell tax" in v.reason

    def test_require_lp_locked(self) -> None:
        gate = EvmSafetyGate(EvmSafetyConfig(require_lp_locked=True))
        s = SafetySignals(
            honeypot=False,
            blacklist_present=False,
            buy_tax_pct=1.0,
            sell_tax_pct=1.0,
            lp_locked_or_burned=False,
        )
        assert gate.check(s).passed is False

    def test_fail_open_on_unknown(self) -> None:
        # All-None safety + fail_open (default) → passes (only explicit bad signals reject).
        assert EvmSafetyGate().check(SafetySignals()).passed is True

    def test_fail_closed_on_unknown(self) -> None:
        gate = EvmSafetyGate(EvmSafetyConfig(fail_open=False))
        v = gate.check(SafetySignals())  # honeypot unknown → reject
        assert v.passed is False
        assert "unknown" in v.reason

    def test_disabled_passes_everything(self) -> None:
        gate = EvmSafetyGate(EvmSafetyConfig(enabled=False))
        assert gate.check(SafetySignals(honeypot=True)).passed is True

    def test_none_safety_fail_open(self) -> None:
        assert EvmSafetyGate().check(None).passed is True


def _snapshot(safety: SafetySignals) -> TokenSnapshot:
    return TokenSnapshot(
        chain=Chain.ETHEREUM,
        token_address=TOKEN,
        symbol="PEPE",
        dex_id="uniswap",
        price_usd=0.001,
        market_cap_usd=5_000_000.0,
        liquidity_usd=500_000.0,
        volume_5m_usd=60_000.0,
        volume_1h_usd=400_000.0,
        txns_5m_buys=70,
        txns_5m_sells=30,
        price_change_5m_pct=2.0,
        price_change_1h_pct=25.0,
        price_change_24h_pct=150.0,
        age_minutes=120.0,
        safety=safety,
    )


def _fetch(snapshot: TokenSnapshot) -> Any:
    async def _get(addr: str, chain: Any = None) -> TokenSnapshot:
        return snapshot

    return _get


class TestEvaluatorIntegration:
    def _ev(self, snapshot: TokenSnapshot, gate: EvmSafetyGate) -> EvmTokenEvaluator:
        return EvmTokenEvaluator(
            [MomentumStrategy(BotConfig())], _fetch(snapshot), safety_gate=gate
        )

    async def test_honeypot_blocks_signals(self) -> None:
        # Would fire momentum, but the honeypot gate rejects it first → no signals.
        snap = _snapshot(SafetySignals(honeypot=True))
        result = await self._ev(snap, EvmSafetyGate()).evaluate(TOKEN)
        assert result is not None
        assert result.rejected is True
        assert result.rejected_reason == "honeypot"
        assert result.signals == []

    async def test_clean_token_still_evaluates(self) -> None:
        result = await self._ev(_snapshot(_clean()), EvmSafetyGate()).evaluate(TOKEN)
        assert result is not None
        assert result.rejected is False
        assert result.strategy_ids == ["momentum"]

    async def test_no_gate_no_rejection(self) -> None:
        # Without a gate, even a honeypot evaluates (gate is opt-in on the evaluator).
        ev = EvmTokenEvaluator(
            [MomentumStrategy(BotConfig())], _fetch(_snapshot(SafetySignals(honeypot=True)))
        )
        result = await ev.evaluate(TOKEN)
        assert result is not None
        assert result.rejected is False
        assert result.strategy_ids == ["momentum"]


class TestConfig:
    def test_defaults(self) -> None:
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.evm_safety_enabled is True
        assert cfg.evm_safety_fail_open is True

    def test_build_attaches_gate(self) -> None:
        async def fetch(t: str, c: Any = None) -> Any:
            return None

        cfg = BotConfig(mode=TradingMode.SIMULATION)
        ev = cfg.build_evm_evaluator(strategies=[], fetch_snapshot=fetch)
        assert ev._safety_gate is not None
        assert ev._safety_gate.config.enabled is True

    def test_env_toggles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVM_SAFETY_ENABLED", "false")
        monkeypatch.setenv("EVM_SAFETY_MAX_BUY_TAX_PCT", "5")
        cfg = BotConfig(mode=TradingMode.SIMULATION)
        assert cfg.evm_safety_enabled is False
        assert cfg.evm_safety_max_buy_tax_pct == 5.0
