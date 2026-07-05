#!/usr/bin/env python3
"""
FENRIR - SolanaClient Test Suite

Pins the live-path fix that simulate_transaction runs at Confirmed commitment
(matching the fetched blockhash + send preflight) — without it the client
defaults to the finalized bank and rejects fresh blockhashes with
BlockhashNotFound.

Run with: pytest tests/test_client.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from solana.rpc.commitment import Confirmed

from fenrir.config import BotConfig, TradingMode
from fenrir.core.client import SolanaClient
from fenrir.logger import FenrirLogger


def _client() -> SolanaClient:
    cfg = BotConfig(mode=TradingMode.SIMULATION, ai_analysis_enabled=False)
    return SolanaClient(cfg, FenrirLogger(cfg))


class TestSimulateCommitment:
    async def test_simulate_passes_confirmed_commitment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sc = _client()
        mock = AsyncMock(return_value=SimpleNamespace(value=SimpleNamespace(err=None)))
        monkeypatch.setattr(sc.client, "simulate_transaction", mock)
        assert await sc.simulate_transaction(MagicMock()) is True
        assert mock.call_args.kwargs.get("commitment") == Confirmed

    async def test_simulate_false_on_err(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sc = _client()
        mock = AsyncMock(
            return_value=SimpleNamespace(value=SimpleNamespace(err="InstructionError"))
        )
        monkeypatch.setattr(sc.client, "simulate_transaction", mock)
        assert await sc.simulate_transaction(MagicMock()) is False

    async def test_simulate_false_on_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sc = _client()
        monkeypatch.setattr(sc.client, "simulate_transaction", AsyncMock(return_value=None))
        assert await sc.simulate_transaction(MagicMock()) is False
