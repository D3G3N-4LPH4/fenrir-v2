#!/usr/bin/env python3
"""
FENRIR - Transaction Config Test Suite

Covers per-strategy execution profiles: slippage/fee/Jito value objects,
the prebuilt profiles, the strategy→profile mapping, env-var overrides,
per-manager isolation (deep copy), and dynamic priority-fee percentile logic.

All RPC I/O is mocked — no network calls.

Run with: pytest tests/test_tx_config.py -v
"""

from __future__ import annotations

from typing import Any

import pytest

from fenrir.trading.tx_config import (
    STRATEGY_TX_PROFILES,
    DynamicFeePreset,
    FeeMode,
    JitoConfig,
    PriorityFeeConfig,
    SlippageConfig,
    TxConfigManager,
    fast_momentum_profile,
    swing_trade_profile,
    ultra_early_snipe_profile,
)

LAMPORTS = 1_000_000_000


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class TestValueObjects:
    def test_slippage_bps(self) -> None:
        s = SlippageConfig(pct=20.0, max_pct=40.0)
        assert s.bps == 2000
        assert s.max_bps == 4000

    def test_priority_fee_lamports(self) -> None:
        p = PriorityFeeConfig(fixed_fee_sol=0.02, max_fee_sol=0.05, min_fee_sol=0.01)
        assert p.fixed_fee_lamports == int(0.02 * LAMPORTS)
        assert p.max_fee_lamports == int(0.05 * LAMPORTS)
        assert p.min_fee_lamports == int(0.01 * LAMPORTS)

    def test_jito_tip_lamports(self) -> None:
        assert JitoConfig(tip_sol=0.001).tip_lamports == int(0.001 * LAMPORTS)


# ---------------------------------------------------------------------------
# Prebuilt profiles
# ---------------------------------------------------------------------------


class TestPrebuiltProfiles:
    def test_snipe_uses_fixed_fee_and_jito(self) -> None:
        p = ultra_early_snipe_profile()
        assert p.priority_fee.mode == FeeMode.FIXED
        assert p.priority_fee.fixed_fee_sol == 0.02
        assert p.jito.enabled is True
        assert "Fixed" in p.summary()

    def test_momentum_dynamic_turbo(self) -> None:
        p = fast_momentum_profile()
        assert p.priority_fee.mode == FeeMode.DYNAMIC
        assert p.priority_fee.dynamic_preset == DynamicFeePreset.TURBO
        assert "Dynamic/turbo" in p.summary()

    def test_swing_jito_off(self) -> None:
        p = swing_trade_profile()
        assert p.jito.enabled is False
        assert p.slippage.pct == 2.0
        assert "Jito OFF" in p.summary()


# ---------------------------------------------------------------------------
# Strategy → profile mapping
# ---------------------------------------------------------------------------


class TestMapping:
    @pytest.mark.parametrize(
        "sid",
        [
            "migration_snipe",
            "sniper",
            "reversal",
            "graduation",
            "volume_anomaly",
            "narrative_tracker",
            "default",
        ],
    )
    def test_all_strategies_mapped(self, sid: str) -> None:
        assert sid in STRATEGY_TX_PROFILES

    def test_unknown_strategy_falls_back_to_default(self) -> None:
        mgr = TxConfigManager()
        assert mgr.get_profile("does_not_exist").name == mgr.get_profile("default").name

    def test_signal_strategy_profiles(self) -> None:
        mgr = TxConfigManager()
        assert mgr.get_profile("migration_snipe").name == "UltraEarlySnipe"
        assert mgr.get_profile("volume_anomaly").name == "VolumeScalp"
        assert mgr.get_profile("narrative_tracker").name == "SwingTrade"

    def test_accessors(self) -> None:
        mgr = TxConfigManager()
        assert mgr.get_slippage_bps("migration_snipe") == 2500
        assert mgr.jito_enabled("migration_snipe") is True
        assert mgr.jito_enabled("narrative_tracker") is False
        assert mgr.jito_tip_lamports("migration_snipe") == int(0.002 * LAMPORTS)

    def test_log_all_profiles_smoke(self) -> None:
        TxConfigManager().log_all_profiles()  # must not raise


# ---------------------------------------------------------------------------
# Environment overrides + isolation
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    def test_snipe_slippage_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SNIPE_SLIPPAGE_PCT", "30")
        mgr = TxConfigManager()
        assert mgr.get_slippage_bps("migration_snipe") == 3000
        assert mgr.get_slippage_bps("sniper") == 3000

    def test_snipe_fee_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SNIPE_PRIORITY_FEE_SOL", "0.03")
        mgr = TxConfigManager()
        assert mgr.get_profile("migration_snipe").priority_fee.fixed_fee_lamports == int(
            0.03 * LAMPORTS
        )

    def test_momentum_slippage_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOMENTUM_SLIPPAGE_PCT", "15")
        mgr = TxConfigManager()
        assert mgr.get_slippage_bps("reversal") == 1500
        assert mgr.get_slippage_bps("graduation") == 1500

    def test_jito_disabled_globally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JITO_ENABLED", "false")
        mgr = TxConfigManager()
        assert mgr.jito_enabled("migration_snipe") is False
        assert mgr.jito_enabled("reversal") is False

    def test_jito_tip_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JITO_TIP_SOL", "0.005")
        mgr = TxConfigManager()
        # Applied only to profiles where Jito is enabled.
        assert mgr.jito_tip_lamports("migration_snipe") == int(0.005 * LAMPORTS)

    def test_override_does_not_mutate_shared_singletons(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SNIPE_SLIPPAGE_PCT", "35")
        overridden = TxConfigManager()
        assert overridden.get_slippage_bps("migration_snipe") == 3500
        # The module-level template is untouched (deep copy isolation)...
        assert STRATEGY_TX_PROFILES["migration_snipe"].slippage.pct == 25.0
        # ...and a fresh manager without the env var uses the default.
        monkeypatch.delenv("SNIPE_SLIPPAGE_PCT")
        fresh = TxConfigManager()
        assert fresh.get_slippage_bps("migration_snipe") == 2500


# ---------------------------------------------------------------------------
# Priority fee calculation
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeResp:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def post(self, url: str, **kwargs: Any) -> _FakeResp:
        return _FakeResp(self._payload)


class _RaisingSession:
    def post(self, url: str, **kwargs: Any) -> _FakeResp:
        raise ConnectionError("rpc down")


class TestPriorityFee:
    async def test_fixed_mode_returns_fixed(self) -> None:
        mgr = TxConfigManager()
        fee = await mgr.get_priority_fee_lamports("migration_snipe")
        assert fee == int(0.02 * LAMPORTS)

    async def test_dynamic_no_session_uses_fallback_clamped(self) -> None:
        mgr = TxConfigManager()
        # reversal = TURBO; fallback 1_000_000 is below the reversal min floor
        # (0.003 SOL = 3_000_000), so it clamps up to the floor.
        fee = await mgr.get_priority_fee_lamports("reversal")
        assert fee == int(0.003 * LAMPORTS)

    async def test_dynamic_exception_returns_min_floor(self) -> None:
        mgr = TxConfigManager(rpc_url="https://rpc.example.com")
        fee = await mgr.get_priority_fee_lamports("reversal", session=_RaisingSession())
        assert fee == int(0.003 * LAMPORTS)

    async def test_fetch_dynamic_fee_percentile(self) -> None:
        mgr = TxConfigManager(rpc_url="https://rpc.example.com")
        fees = {"result": [{"prioritizationFee": n} for n in range(100_000, 1_100_000, 100_000)]}
        session = _FakeSession(fees)
        # 10 sorted values; TURBO -> 90th pct -> index 9 -> 1_000_000
        turbo = await mgr._fetch_dynamic_fee(DynamicFeePreset.TURBO, session)
        assert turbo == 1_000_000
        # MEDIUM -> 50th pct -> index 5 -> 600_000
        medium = await mgr._fetch_dynamic_fee(DynamicFeePreset.MEDIUM, session)
        assert medium == 600_000

    async def test_fetch_dynamic_fee_empty_result(self) -> None:
        mgr = TxConfigManager(rpc_url="https://rpc.example.com")
        fee = await mgr._fetch_dynamic_fee(
            DynamicFeePreset.AGGRESSIVE, _FakeSession({"result": []})
        )
        assert fee == 500_000

    async def test_fetch_dynamic_fee_no_session_fallback(self) -> None:
        mgr = TxConfigManager()
        assert await mgr._fetch_dynamic_fee(DynamicFeePreset.TURBO, None) == 1_000_000
        assert await mgr._fetch_dynamic_fee(DynamicFeePreset.MEDIUM, None) == 100_000
