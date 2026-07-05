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
# Real pump.fun mints observed in live MigrateV2 txs (vanity "pump" suffix).
PUMP_TOKEN = "5e8zEuwJu7mYshNqsPhvKw4YGkaCTx77zysxCmYQpump"
PUMP_TOKEN2 = "97pz15v1VYv5MfQw2Sgf8T4THa814QmoGn5KiWcPpump"


# ---------------------------------------------------------------------------
# MigrationDetector (pure)
# ---------------------------------------------------------------------------


class TestMigrationDetector:
    @pytest.mark.parametrize(
        "log",
        [
            "Program log: Instruction: Migrate",
            "Program log: Instruction: MigrateV2",  # real graduations use V2
            "Program log: Migrate",
        ],
    )
    def test_hint_matches(self, log: str) -> None:
        assert MigrationDetector().has_migration_hint(["noise", log, "more"]) is True

    @pytest.mark.parametrize(
        "log",
        [
            "Program log: Instruction: Buy",
            "Program log: Instruction: Withdraw",  # narrowed out (FP-prone)
            "Program ... invoke [1]: initialize2",  # narrowed out (FP-prone)
            "Program log: Instruction: InitializeMint2",  # generic SPL, not a migration
        ],
    )
    def test_hint_no_match(self, log: str) -> None:
        assert MigrationDetector().has_migration_hint([log]) is False

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
        # Two distinct non-WSOL mints, neither pump-suffixed → skip.
        assert _monitor()._extract_migration_token_data(_tx(post=[TOKEN, OTHER])) is None

    def test_ambiguous_prefers_pump_suffix(self) -> None:
        # Token mint + an LP-like mint → pick the pump-suffixed token.
        td = _monitor()._extract_migration_token_data(_tx(post=[OTHER, PUMP_TOKEN, WSOL_MINT]))
        assert td is not None
        assert td["token_address"] == PUMP_TOKEN

    def test_no_meta_returns_none(self) -> None:
        tx = SimpleNamespace(transaction=SimpleNamespace(meta=None))
        assert _monitor()._extract_migration_token_data(tx) is None

    def test_mint_from_pre_balances(self) -> None:
        td = _monitor()._extract_migration_token_data(_tx(post=[WSOL_MINT], pre=[TOKEN]))
        assert td is not None
        assert td["token_address"] == TOKEN


class TestPickTokenMint:
    def _pick(self, mints: list[str]) -> str | None:
        return MigrationDetector().pick_token_mint(mints)

    def test_single(self) -> None:
        assert self._pick([TOKEN]) == TOKEN

    def test_dedup_single(self) -> None:
        assert self._pick([TOKEN, TOKEN]) == TOKEN

    def test_wsol_filtered(self) -> None:
        assert self._pick([WSOL_MINT]) is None
        assert self._pick([PUMP_TOKEN, WSOL_MINT]) == PUMP_TOKEN

    def test_empty(self) -> None:
        assert self._pick([]) is None
        assert self._pick(["", ""]) is None

    def test_ambiguous_non_pump(self) -> None:
        assert self._pick([TOKEN, OTHER]) is None

    def test_prefers_unique_pump(self) -> None:
        assert self._pick([OTHER, PUMP_TOKEN]) == PUMP_TOKEN

    def test_two_pump_is_ambiguous(self) -> None:
        assert self._pick([PUMP_TOKEN, PUMP_TOKEN2]) is None
