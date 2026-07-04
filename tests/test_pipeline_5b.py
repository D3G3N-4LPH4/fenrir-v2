#!/usr/bin/env python3
"""
FENRIR - Pipeline Wiring 5b Test Suite

Covers per-strategy transaction execution: the TradingEngine resolves
slippage / priority-fee / jito from a TxConfigManager when tx_profiles_enabled,
and falls back to the flat BotConfig values otherwise.

The engine is built with mocked collaborators; no chain I/O. Only the
resolution helpers are exercised (the live tx path needs Solana).

Run with: pytest tests/test_pipeline_5b.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.trading.engine import TradingEngine

LAMPORTS = 1_000_000_000


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TX_PROFILES_ENABLED", raising=False)


def _engine(jito: Any = None, **cfg_over: Any) -> TradingEngine:
    cfg = BotConfig(mode=TradingMode.SIMULATION, ai_analysis_enabled=False, **cfg_over)
    return TradingEngine(cfg, Mock(), Mock(), Mock(), Mock(), Mock(), jito=jito)


# ---------------------------------------------------------------------------
# Flat config (tx profiles OFF) — default behavior
# ---------------------------------------------------------------------------


class TestFlatConfig:
    def test_no_tx_config_built(self) -> None:
        assert _engine().tx_config is None

    async def test_priority_fee_is_flat(self) -> None:
        eng = _engine(priority_fee_lamports=750_000)
        assert await eng._resolve_priority_fee("migration_snipe") == 750_000
        assert await eng._resolve_priority_fee("anything") == 750_000

    def test_slippage_is_flat(self) -> None:
        eng = _engine(max_slippage_bps=900)
        assert eng._resolve_slippage_bps("reversal") == 900

    def test_jito_none_is_false(self) -> None:
        assert _engine(jito=None, use_jito=True)._resolve_use_jito("sniper") is False

    def test_jito_present_follows_flat_flag(self) -> None:
        assert _engine(jito=Mock(), use_jito=True)._resolve_use_jito("sniper") is True
        assert _engine(jito=Mock(), use_jito=False)._resolve_use_jito("sniper") is False


# ---------------------------------------------------------------------------
# tx profiles ON — per-strategy
# ---------------------------------------------------------------------------


class TestTxProfiles:
    def test_tx_config_built(self) -> None:
        assert _engine(tx_profiles_enabled=True).tx_config is not None

    async def test_fixed_fee_profile(self) -> None:
        eng = _engine(tx_profiles_enabled=True)
        # migration_snipe = UltraEarlySnipe: fixed 0.02 SOL.
        assert await eng._resolve_priority_fee("migration_snipe") == int(0.02 * LAMPORTS)

    async def test_dynamic_fee_clamped_to_floor(self) -> None:
        eng = _engine(tx_profiles_enabled=True)
        # reversal = FastMomentum (TURBO, no session): fallback 1_000_000 clamps
        # up to the profile's 0.003 SOL floor.
        assert await eng._resolve_priority_fee("reversal") == int(0.003 * LAMPORTS)

    def test_slippage_per_profile(self) -> None:
        eng = _engine(tx_profiles_enabled=True)
        assert eng._resolve_slippage_bps("migration_snipe") == 2500  # 25%
        assert eng._resolve_slippage_bps("reversal") == 800  # 8%
        assert eng._resolve_slippage_bps("narrative_tracker") == 200  # 2%

    def test_unknown_strategy_uses_default_profile(self) -> None:
        eng = _engine(tx_profiles_enabled=True)
        # default profile = FastMomentum → 8% slippage.
        assert eng._resolve_slippage_bps("does_not_exist") == 800

    def test_jito_gated_by_profile(self) -> None:
        eng = _engine(jito=Mock(), tx_profiles_enabled=True)
        assert eng._resolve_use_jito("migration_snipe") is True  # Jito ON
        assert eng._resolve_use_jito("narrative_tracker") is False  # SwingTrade Jito OFF

    def test_jito_none_overrides_profile(self) -> None:
        eng = _engine(jito=None, tx_profiles_enabled=True)
        assert eng._resolve_use_jito("migration_snipe") is False
