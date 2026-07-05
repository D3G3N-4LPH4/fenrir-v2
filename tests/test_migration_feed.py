#!/usr/bin/env python3
"""
FENRIR - Migration Feed (5c) Test Suite

Covers the offline-verifiable parts of the experimental pump→Raydium migration
feed: the MigrationDetector (hint matcher + token_data builder), the
migration_feed_enabled config flag, and the monitor's token-mint extraction
heuristic (driven with fabricated transaction objects).

The live logsSubscribe loop itself is network I/O and is not unit-tested — it
is gated behind config.migration_feed_enabled (off by default) and flagged
experimental in the source.

Run with: pytest tests/test_migration_feed.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.trading.migration import MigrationDetector
from fenrir.trading.monitor import WSOL_MINT, PumpFunMonitor

TOKEN = "So11111111111111111111111111111111111111112ABC"
OTHER = "Mint2222222222222222222222222222222222222222"


# ---------------------------------------------------------------------------
# MigrationDetector (pure)
# ---------------------------------------------------------------------------


class TestMigrationDetector:
    @pytest.mark.parametrize(
        "log",
        [
            "Program log: Instruction: Migrate",
            "Program log: Migrate",
            "Program log: Instruction: Withdraw",
            "Program ... invoke [1]: initialize2",
        ],
    )
    def test_hint_matches(self, log: str) -> None:
        assert MigrationDetector().has_migration_hint(["noise", log, "more"]) is True

    def test_hint_no_match(self) -> None:
        assert MigrationDetector().has_migration_hint(["Program log: Instruction: Buy"]) is False

    def test_hint_empty(self) -> None:
        assert MigrationDetector().has_migration_hint([]) is False

    def test_build_token_data_defaults(self) -> None:
        td = MigrationDetector().build_token_data(TOKEN)
        assert td["token_address"] == TOKEN
        assert td["dex_id"] == "raydium"
        assert td["migrated"] is True
        assert td["pair_address"] is None
        assert "bonding_curve_state" not in td  # curve complete post-migration

    def test_build_token_data_custom(self) -> None:
        td = MigrationDetector().build_token_data(
            TOKEN, symbol="DOGE", name="Doge", pair_address="PAIR"
        )
        assert td["symbol"] == "DOGE"
        assert td["name"] == "Doge"
        assert td["pair_address"] == "PAIR"


# ---------------------------------------------------------------------------
# Config flag
# ---------------------------------------------------------------------------


class TestConfigFlag:
    @pytest.fixture(autouse=True)
    def _clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MIGRATION_FEED_ENABLED", raising=False)

    def test_default_off(self) -> None:
        assert BotConfig().migration_feed_enabled is False

    def test_env_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATION_FEED_ENABLED", "true")
        assert BotConfig().migration_feed_enabled is True

    def test_kwarg_preserved(self) -> None:
        assert BotConfig(migration_feed_enabled=True).migration_feed_enabled is True


# ---------------------------------------------------------------------------
# Monitor extractor heuristic
# ---------------------------------------------------------------------------


def _monitor() -> PumpFunMonitor:
    cfg = BotConfig(mode=TradingMode.SIMULATION, ai_analysis_enabled=False)
    return PumpFunMonitor(cfg, Mock(), Mock())


def _tx(post: list[str], pre: list[str] | None = None) -> Any:
    def _bals(mints: list[str]) -> list[Any]:
        return [SimpleNamespace(mint=m) for m in mints]

    return SimpleNamespace(
        transaction=SimpleNamespace(
            meta=SimpleNamespace(
                post_token_balances=_bals(post),
                pre_token_balances=_bals(pre or []),
            )
        )
    )


class TestMigrationExtractor:
    def test_single_non_wsol_mint(self) -> None:
        td = _monitor()._extract_migration_token_data(_tx(post=[TOKEN, WSOL_MINT]))
        assert td is not None
        assert td["token_address"] == TOKEN
        assert td["dex_id"] == "raydium"

    def test_wsol_only_returns_none(self) -> None:
        assert _monitor()._extract_migration_token_data(_tx(post=[WSOL_MINT])) is None

    def test_ambiguous_multiple_mints_returns_none(self) -> None:
        # Two distinct non-WSOL mints → skip rather than guess.
        assert _monitor()._extract_migration_token_data(_tx(post=[TOKEN, OTHER])) is None

    def test_no_meta_returns_none(self) -> None:
        tx = SimpleNamespace(transaction=SimpleNamespace(meta=None))
        assert _monitor()._extract_migration_token_data(tx) is None

    def test_mint_from_pre_balances(self) -> None:
        td = _monitor()._extract_migration_token_data(_tx(post=[WSOL_MINT], pre=[TOKEN]))
        assert td is not None
        assert td["token_address"] == TOKEN
