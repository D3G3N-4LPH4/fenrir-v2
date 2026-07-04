#!/usr/bin/env python3
"""
FENRIR - Config Surface Test Suite

Covers the strategy-selection / pre-trade-filter / tx-profile fields added to
BotConfig: defaults, env-var parsing, the env helpers, the component-config
builders, and validation. These fields are inert (not yet consumed by the bot
loop) — this suite pins the surface itself.

Run with: pytest tests/test_config_surface.py -v
"""

from __future__ import annotations

import pytest

from fenrir.config import BotConfig, TradingMode, _env_bool, _env_float
from fenrir.filters import MarketFilterConfig, SecurityFilterConfig
from fenrir.trading.tx_config import TxConfigManager

# Env vars this surface reads — cleared before each test for determinism
# regardless of the developer's local .env.
_SURFACE_ENV = [
    "ENABLED_STRATEGIES",
    "SECURITY_FILTER_ENABLED",
    "SECURITY_REQUIRE_MINT_REVOKED",
    "SECURITY_REQUIRE_FREEZE_REVOKED",
    "SECURITY_MIN_LP_BURNED_PCT",
    "SECURITY_MAX_TOP10_HOLDER_PCT",
    "SECURITY_FAIL_OPEN_ON_HOLDER_FETCH_ERROR",
    "HELIUS_API_KEY",
    "MARKET_FILTER_ENABLED",
    "MARKET_FAIL_OPEN_ON_FETCH_ERROR",
    "TX_PROFILES_ENABLED",
]


@pytest.fixture(autouse=True)
def _clear_surface_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _SURFACE_ENV:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


class TestEnvHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("1", True), ("YES", True), ("on", True), ("false", False), ("0", False)],
    )
    def test_env_bool_parsing(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        monkeypatch.setenv("X_FLAG", raw)
        assert _env_bool("X_FLAG", not expected) is expected

    def test_env_bool_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("X_FLAG", raising=False)
        assert _env_bool("X_FLAG", True) is True
        assert _env_bool("X_FLAG", False) is False

    def test_env_float_parsing_and_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X_NUM", "12.5")
        assert _env_float("X_NUM", 1.0) == 12.5
        monkeypatch.setenv("X_NUM", "not-a-number")
        assert _env_float("X_NUM", 1.0) == 1.0
        monkeypatch.delenv("X_NUM")
        assert _env_float("X_NUM", 3.0) == 3.0


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_strategy_default(self) -> None:
        assert BotConfig().enabled_strategies == ["sniper"]

    def test_filters_off_by_default(self) -> None:
        cfg = BotConfig()
        assert cfg.security_filter_enabled is False
        assert cfg.market_filter_enabled is False
        assert cfg.tx_profiles_enabled is False

    def test_security_threshold_defaults(self) -> None:
        cfg = BotConfig()
        assert cfg.security_min_lp_burned_pct == 90.0
        assert cfg.security_max_top10_holder_pct == 30.0
        assert cfg.security_require_mint_revoked is True

    def test_enabled_strategies_is_per_instance(self) -> None:
        a = BotConfig()
        a.enabled_strategies.append("reversal")
        assert BotConfig().enabled_strategies == ["sniper"]  # not shared

    def test_from_mode_keeps_surface_defaults(self) -> None:
        cfg = BotConfig.from_mode(TradingMode.DEGEN)
        assert cfg.enabled_strategies == ["sniper"]
        assert cfg.security_filter_enabled is False


# ---------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------


class TestEnvParsing:
    def test_enabled_strategies_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENABLED_STRATEGIES", "sniper, reversal ,volume_anomaly")
        assert BotConfig().enabled_strategies == ["sniper", "reversal", "volume_anomaly"]

    def test_filter_flags_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECURITY_FILTER_ENABLED", "true")
        monkeypatch.setenv("MARKET_FILTER_ENABLED", "true")
        monkeypatch.setenv("TX_PROFILES_ENABLED", "true")
        cfg = BotConfig()
        assert cfg.security_filter_enabled is True
        assert cfg.market_filter_enabled is True
        assert cfg.tx_profiles_enabled is True

    def test_security_thresholds_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECURITY_MIN_LP_BURNED_PCT", "80")
        monkeypatch.setenv("SECURITY_MAX_TOP10_HOLDER_PCT", "20")
        monkeypatch.setenv("SECURITY_REQUIRE_MINT_REVOKED", "false")
        cfg = BotConfig()
        assert cfg.security_min_lp_burned_pct == 80.0
        assert cfg.security_max_top10_holder_pct == 20.0
        assert cfg.security_require_mint_revoked is False

    def test_helius_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HELIUS_API_KEY", "hx-test")
        assert BotConfig().helius_api_key == "hx-test"

    def test_explicit_kwarg_preserved_when_env_absent(self) -> None:
        cfg = BotConfig(security_filter_enabled=True, security_min_lp_burned_pct=75.0)
        assert cfg.security_filter_enabled is True
        assert cfg.security_min_lp_burned_pct == 75.0


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------


class TestBuilders:
    def test_build_security_filter_config(self) -> None:
        cfg = BotConfig(
            security_require_mint_revoked=False,
            security_require_freeze_revoked=True,
            security_min_lp_burned_pct=85.0,
            security_max_top10_holder_pct=25.0,
            security_fail_open_on_holder_fetch_error=True,
        )
        sec = cfg.build_security_filter_config()
        assert isinstance(sec, SecurityFilterConfig)
        assert sec.require_mint_revoked is False
        assert sec.require_freeze_revoked is True
        assert sec.min_lp_burned_pct == 85.0
        assert sec.max_top10_holder_pct == 25.0
        assert sec.fail_open_on_holder_fetch_error is True

    def test_build_market_filter_config(self) -> None:
        cfg = BotConfig(market_filter_enabled=True, market_fail_open_on_fetch_error=False)
        mkt = cfg.build_market_filter_config()
        assert isinstance(mkt, MarketFilterConfig)
        assert mkt.enabled is True
        assert mkt.fail_open_on_fetch_error is False

    def test_build_tx_config_manager(self) -> None:
        cfg = BotConfig(rpc_url="https://rpc.example.com")
        mgr = cfg.build_tx_config_manager()
        assert isinstance(mgr, TxConfigManager)
        assert mgr.rpc_url == "https://rpc.example.com"
        # Sanity: it resolves a profile for a signal strategy.
        assert mgr.get_profile("migration_snipe").name == "UltraEarlySnipe"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_known_strategies_pass(self) -> None:
        cfg = BotConfig(
            enabled_strategies=[
                "sniper",
                "migration_snipe",
                "reversal",
                "volume_anomaly",
                "narrative_tracker",
            ]
        )
        errors = cfg.validate()
        assert not any("Unknown strategy" in e for e in errors)

    def test_unknown_strategy_flagged(self) -> None:
        errors = BotConfig(enabled_strategies=["sniper", "nope"]).validate()
        assert any("Unknown strategy" in e and "nope" in e for e in errors)

    def test_empty_strategies_flagged(self) -> None:
        errors = BotConfig(enabled_strategies=[]).validate()
        assert any("At least one strategy" in e for e in errors)

    def test_lp_burned_out_of_range_flagged(self) -> None:
        errors = BotConfig(security_min_lp_burned_pct=150.0).validate()
        assert any("security_min_lp_burned_pct" in e for e in errors)

    def test_top10_out_of_range_flagged(self) -> None:
        errors = BotConfig(security_max_top10_holder_pct=-5.0).validate()
        assert any("security_max_top10_holder_pct" in e for e in errors)

    def test_valid_thresholds_pass(self) -> None:
        errors = BotConfig(
            security_min_lp_burned_pct=90.0, security_max_top10_holder_pct=30.0
        ).validate()
        assert not any("security_min_lp_burned_pct" in e for e in errors)
        assert not any("security_max_top10_holder_pct" in e for e in errors)
