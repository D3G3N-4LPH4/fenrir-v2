#!/usr/bin/env python3
"""
FENRIR - Geyser (Yellowstone gRPC) monitor transport tests

Covers transport selection + WebSocket fallback (Phase 1.1 acceptance criteria)
and the update handler that turns a Geyser transaction into a launch, reusing the
same full-tx parse as the WebSocket path. Network is fully mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.trading.monitor import PumpFunMonitor

PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
CREATE_LOG = "Program log: Instruction: CreateV2"


def _monitor(**cfg_over: Any) -> PumpFunMonitor:
    cfg = BotConfig(mode=TradingMode.SIMULATION, ai_analysis_enabled=False, **cfg_over)
    m = PumpFunMonitor(cfg, Mock(), Mock())
    m.client.pumpfun_program = PUMP
    return m


def _tx_update(sig: bytes, logs: list[str], is_vote: bool = False, has_err: bool = False) -> Any:
    """Mimic a geyser SubscribeUpdate.transaction shape closely enough for the handler."""
    meta = SimpleNamespace(
        log_messages=logs,
        HasField=lambda f: (f == "err" and has_err),
    )
    info = SimpleNamespace(signature=sig, is_vote=is_vote, meta=meta)
    return SimpleNamespace(transaction=info)


# ---------------------------------------------------------------------------
# Transport selection + fallback (Phase 1.1 acceptance criteria)
# ---------------------------------------------------------------------------


class TestTransportSelection:
    @pytest.mark.asyncio
    async def test_falls_back_to_websocket_when_no_provider(self, monkeypatch):
        """No GEYSER endpoint -> the WebSocket path runs, gRPC path is never touched."""
        m = _monitor(geyser_grpc_endpoint="")
        ws = AsyncMock()
        geyser = AsyncMock()
        monkeypatch.setattr(m, "_monitor_websocket", ws)
        monkeypatch.setattr(m, "_monitor_geyser", geyser)
        await m.start_monitoring(AsyncMock())
        ws.assert_awaited_once()
        geyser.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_geyser_when_provider_configured(self, monkeypatch):
        m = _monitor(geyser_grpc_endpoint="https://solana-mainnet.g.alchemy.com/v2/key")
        ws = AsyncMock()
        geyser = AsyncMock()
        monkeypatch.setattr(m, "_monitor_websocket", ws)
        monkeypatch.setattr(m, "_monitor_geyser", geyser)
        await m.start_monitoring(AsyncMock())
        geyser.assert_awaited_once()
        ws.assert_not_awaited()


# ---------------------------------------------------------------------------
# Subscribe request + channel target
# ---------------------------------------------------------------------------


class TestSubscribeRequest:
    def test_filters_pump_transactions_only(self):
        req = _monitor()._geyser_subscribe_request()
        f = req.transactions["pump"]
        assert list(f.account_include) == [PUMP]
        assert f.vote is False
        assert f.failed is False

    def test_channel_target_is_host_443_from_rpc_url(self, monkeypatch):
        import grpc

        captured = {}

        def fake_secure(target, creds):
            captured["target"] = target
            return Mock()

        monkeypatch.setattr(grpc.aio, "secure_channel", fake_secure)
        monkeypatch.setattr(grpc, "ssl_channel_credentials", lambda: Mock())
        _monitor(
            geyser_grpc_endpoint="https://solana-mainnet.g.alchemy.com/v2/alch_key"
        )._geyser_channel()
        assert captured["target"] == "solana-mainnet.g.alchemy.com:443"


# ---------------------------------------------------------------------------
# Update handler — reuses the shared parse, mirrors the WebSocket gating
# ---------------------------------------------------------------------------


class TestHandleGeyserTx:
    async def _run(self, monitor, update):
        launches = []
        monitor.client.get_transaction = AsyncMock(return_value=SimpleNamespace())
        monitor._is_token_launch = Mock(return_value=True)
        monitor._extract_token_data = AsyncMock(
            return_value={"token_address": "MINT", "symbol": "X"}
        )
        monitor._meets_criteria = Mock(return_value=True)

        async def _on_launch(td):
            launches.append(td)

        # _handle_geyser_tx receives update.transaction (a SubscribeUpdateTransaction),
        # which _tx_update models directly.
        await monitor._handle_geyser_tx(update, _on_launch)
        return launches

    @pytest.mark.asyncio
    async def test_createv2_hint_triggers_launch(self):
        m = _monitor()
        launches = await self._run(m, _tx_update(b"\x01" * 64, ["x", CREATE_LOG, "y"]))
        assert len(launches) == 1
        assert launches[0]["symbol"] == "X"

    @pytest.mark.asyncio
    async def test_no_createv2_hint_is_skipped(self):
        m = _monitor()
        m.client.get_transaction = AsyncMock()
        await m._handle_geyser_tx(_tx_update(b"\x02" * 64, ["unrelated log"]), AsyncMock())
        m.client.get_transaction.assert_not_awaited()  # cheap hint gate, no RPC

    @pytest.mark.asyncio
    async def test_duplicate_signature_skipped(self):
        m = _monitor()
        upd = _tx_update(b"\x03" * 64, [CREATE_LOG])
        first = await self._run(m, upd)
        second = await self._run(m, upd)  # same signature
        assert len(first) == 1
        assert len(second) == 0

    @pytest.mark.asyncio
    async def test_vote_and_failed_skipped(self):
        m = _monitor()
        m.client.get_transaction = AsyncMock()
        await m._handle_geyser_tx(_tx_update(b"\x04" * 64, [CREATE_LOG], is_vote=True), AsyncMock())
        await m._handle_geyser_tx(_tx_update(b"\x05" * 64, [CREATE_LOG], has_err=True), AsyncMock())
        m.client.get_transaction.assert_not_awaited()


# ---------------------------------------------------------------------------
# Config (additive, non-breaking)
# ---------------------------------------------------------------------------


class TestGeyserConfig:
    def test_defaults_empty_so_websocket_is_used(self, monkeypatch):
        for v in ("GEYSER_GRPC_ENDPOINT", "GEYSER_X_TOKEN"):
            monkeypatch.delenv(v, raising=False)
        cfg = BotConfig()
        assert cfg.geyser_grpc_endpoint == ""
        assert cfg.geyser_commitment == "processed"

    def test_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("GEYSER_GRPC_ENDPOINT", "https://x.example/v2/k")
        monkeypatch.setenv("GEYSER_X_TOKEN", "tok")
        cfg = BotConfig()
        assert cfg.geyser_grpc_endpoint == "https://x.example/v2/k"
        assert cfg.geyser_x_token == "tok"
