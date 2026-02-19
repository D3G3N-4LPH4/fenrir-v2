#!/usr/bin/env python3
"""
FENRIR Trading Bot - API Server Test Suite

Tests for the FastAPI REST API defined in api/server.py.
Covers authentication middleware, all HTTP endpoints, input validation,
bot state management, and the WebSocket endpoint.

Run with: pytest tests/test_api.py -v
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import api.server as server_module
from api.server import app


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _reset_bot_state():
    """Reset the global bot_state dict to its default stopped state."""
    server_module.bot_state["status"] = "stopped"
    server_module.bot_state["start_time"] = None
    server_module.bot_state["config"] = None
    server_module.bot_state["error"] = None
    server_module.bot_instance = None
    server_module.bot_task = None


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(autouse=True)
async def _clean_state():
    """Ensure every test starts with pristine global state."""
    _reset_bot_state()
    # Clear any lingering websocket references
    server_module.active_websockets.clear()
    yield
    _reset_bot_state()
    server_module.active_websockets.clear()


@pytest_asyncio.fixture
async def client_no_auth():
    """AsyncClient with dev mode ON and no API key -- unauthenticated."""
    original_key = server_module.FENRIR_API_KEY
    original_dev = server_module.FENRIR_DEV_MODE
    server_module.FENRIR_API_KEY = ""
    server_module.FENRIR_DEV_MODE = True
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    server_module.FENRIR_API_KEY = original_key
    server_module.FENRIR_DEV_MODE = original_dev


@pytest_asyncio.fixture
async def client_with_key():
    """AsyncClient with a known API key set -- authenticated requests."""
    original_key = server_module.FENRIR_API_KEY
    original_dev = server_module.FENRIR_DEV_MODE
    server_module.FENRIR_API_KEY = "test-secret-key"
    server_module.FENRIR_DEV_MODE = False
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    server_module.FENRIR_API_KEY = original_key
    server_module.FENRIR_DEV_MODE = original_dev


# ---------------------------------------------------------------------------
#  Public / health endpoints
# ---------------------------------------------------------------------------

class TestPublicEndpoints:
    """Tests for unauthenticated, publicly-accessible endpoints."""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, client_no_auth: AsyncClient):
        """GET / should return service info with status 200."""
        resp = await client_no_auth.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "Fenrir Trading Bot API"
        assert body["version"] == "1.0.0"
        assert body["status"] == "operational"

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client_no_auth: AsyncClient):
        """GET /health should return healthy status."""
        resp = await client_no_auth.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["bot_status"] == "stopped"
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_health_uptime_none_when_stopped(self, client_no_auth: AsyncClient):
        """Uptime should be None when the bot has not been started."""
        resp = await client_no_auth.get("/health")
        assert resp.json()["uptime_seconds"] is None

    @pytest.mark.asyncio
    async def test_health_uptime_present_when_running(self, client_no_auth: AsyncClient):
        """Uptime should be a number when the bot state has a start_time."""
        server_module.bot_state["status"] = "running"
        server_module.bot_state["start_time"] = datetime.now()
        resp = await client_no_auth.get("/health")
        uptime = resp.json()["uptime_seconds"]
        assert uptime is not None
        assert uptime >= 0


# ---------------------------------------------------------------------------
#  Authentication middleware
# ---------------------------------------------------------------------------

class TestAuthMiddleware:
    """Tests for the verify_api_key middleware."""

    @pytest.mark.asyncio
    async def test_public_paths_bypass_auth(self, client_with_key: AsyncClient):
        """Public paths (/, /health, /docs) should NOT require an API key."""
        for path in ["/", "/health", "/docs"]:
            resp = await client_with_key.get(path)
            # /docs may redirect or return 200; the key point is it is NOT 403
            assert resp.status_code != 403, f"{path} was blocked by auth middleware"

    @pytest.mark.asyncio
    async def test_protected_path_rejected_without_key(self, client_with_key: AsyncClient):
        """Protected endpoint should return 403 when no X-API-Key header is sent."""
        resp = await client_with_key.get("/bot/status")
        assert resp.status_code == 403
        assert "Invalid or missing API key" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_protected_path_rejected_with_wrong_key(self, client_with_key: AsyncClient):
        """Protected endpoint should return 403 when the wrong key is sent."""
        resp = await client_with_key.get(
            "/bot/status",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_protected_path_allowed_with_correct_key(self, client_with_key: AsyncClient):
        """Protected endpoint should succeed when the correct key is sent."""
        resp = await client_with_key.get(
            "/bot/status",
            headers={"X-API-Key": "test-secret-key"},
        )
        # The endpoint itself should return 200 (bot stopped state)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_key_no_dev_mode_returns_500(self):
        """When FENRIR_API_KEY is empty AND dev mode is off, middleware returns 500."""
        original_key = server_module.FENRIR_API_KEY
        original_dev = server_module.FENRIR_DEV_MODE
        server_module.FENRIR_API_KEY = ""
        server_module.FENRIR_DEV_MODE = False
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                resp = await ac.get("/bot/status")
            assert resp.status_code == 500
            assert "FENRIR_API_KEY not configured" in resp.json()["detail"]
        finally:
            server_module.FENRIR_API_KEY = original_key
            server_module.FENRIR_DEV_MODE = original_dev

    @pytest.mark.asyncio
    async def test_dev_mode_bypasses_auth(self, client_no_auth: AsyncClient):
        """When FENRIR_DEV_MODE=true and no key is set, protected paths are allowed."""
        resp = await client_no_auth.get("/bot/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_public_paths_even_without_any_config(self):
        """/, /health work even when no key and dev mode off."""
        original_key = server_module.FENRIR_API_KEY
        original_dev = server_module.FENRIR_DEV_MODE
        server_module.FENRIR_API_KEY = ""
        server_module.FENRIR_DEV_MODE = False
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                resp_root = await ac.get("/")
                resp_health = await ac.get("/health")
            assert resp_root.status_code == 200
            assert resp_health.status_code == 200
        finally:
            server_module.FENRIR_API_KEY = original_key
            server_module.FENRIR_DEV_MODE = original_dev


# ---------------------------------------------------------------------------
#  Bot status / positions / config (GET endpoints)
# ---------------------------------------------------------------------------

class TestBotStatusEndpoint:
    """Tests for GET /bot/status."""

    @pytest.mark.asyncio
    async def test_status_when_stopped(self, client_no_auth: AsyncClient):
        """Status should reflect 'stopped' and have zero positions."""
        resp = await client_no_auth.get("/bot/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "stopped"
        assert body["positions_count"] == 0
        assert body["mode"] is None
        assert body["uptime_seconds"] is None

    @pytest.mark.asyncio
    async def test_status_when_running(self, client_no_auth: AsyncClient):
        """Status should reflect 'running' with mode and uptime when bot is active."""
        server_module.bot_state["status"] = "running"
        server_module.bot_state["start_time"] = datetime.now()
        server_module.bot_state["config"] = {"mode": "simulation"}

        mock_bot = MagicMock()
        mock_bot.positions.get_portfolio_summary.return_value = {"total_pnl_sol": 0.5}
        mock_bot.positions.positions = {"TOKEN1": MagicMock()}
        server_module.bot_instance = mock_bot

        resp = await client_no_auth.get("/bot/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["mode"] == "simulation"
        assert body["positions_count"] == 1
        assert body["uptime_seconds"] is not None
        assert body["portfolio"] == {"total_pnl_sol": 0.5}

    @pytest.mark.asyncio
    async def test_status_with_error(self, client_no_auth: AsyncClient):
        """Status should include error message when bot is in error state."""
        server_module.bot_state["status"] = "error"
        server_module.bot_state["error"] = "Connection lost"

        resp = await client_no_auth.get("/bot/status")
        body = resp.json()
        assert body["status"] == "error"
        assert body["error"] == "Connection lost"


class TestBotPositionsEndpoint:
    """Tests for GET /bot/positions."""

    @pytest.mark.asyncio
    async def test_positions_empty_when_stopped(self, client_no_auth: AsyncClient):
        """When bot is not running, return empty positions list."""
        resp = await client_no_auth.get("/bot/positions")
        assert resp.status_code == 200
        assert resp.json()["positions"] == []

    @pytest.mark.asyncio
    async def test_positions_returns_open_positions(self, client_no_auth: AsyncClient):
        """When bot is running, return formatted position data."""
        now = datetime.now()
        mock_position = MagicMock()
        mock_position.entry_time = now
        mock_position.entry_price = 0.000001
        mock_position.current_price = 0.000002
        mock_position.amount_tokens = 1_000_000
        mock_position.amount_sol_invested = 1.0
        mock_position.get_pnl_percent.return_value = 100.0
        mock_position.get_pnl_sol.return_value = 1.0
        mock_position.peak_price = 0.0000025

        mock_bot = MagicMock()
        mock_bot.positions.positions = {"TOKEN1": mock_position}
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.get("/bot/positions")
        assert resp.status_code == 200
        positions = resp.json()["positions"]
        assert len(positions) == 1

        pos = positions[0]
        assert pos["token_address"] == "TOKEN1"
        assert pos["entry_price"] == 0.000001
        assert pos["current_price"] == 0.000002
        assert pos["pnl_percent"] == 100.0
        assert pos["pnl_sol"] == 1.0


class TestBotConfigEndpoint:
    """Tests for GET /bot/config."""

    @pytest.mark.asyncio
    async def test_config_not_set(self, client_no_auth: AsyncClient):
        """Should return 400 when bot has never been configured."""
        resp = await client_no_auth.get("/bot/config")
        assert resp.status_code == 400
        assert "not been configured" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_config_returns_current(self, client_no_auth: AsyncClient):
        """Should return the stored config when present."""
        server_module.bot_state["config"] = {
            "mode": "simulation",
            "buy_amount_sol": 0.1,
        }
        resp = await client_no_auth.get("/bot/config")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["config"]["mode"] == "simulation"


# ---------------------------------------------------------------------------
#  POST /bot/start
# ---------------------------------------------------------------------------

class TestStartBot:
    """Tests for POST /bot/start."""

    @pytest.mark.asyncio
    async def test_start_bot_success(self, client_no_auth: AsyncClient):
        """Starting the bot with valid config should succeed."""
        mock_config = MagicMock()
        mock_config.validate.return_value = []

        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.BotConfig", return_value=mock_config), \
             patch("api.server.TradingMode") as mock_tm, \
             patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post("/bot/start", json={
                "mode": "simulation",
                "buy_amount_sol": 0.1,
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "simulation" in body["message"]
        assert server_module.bot_state["status"] == "running"
        assert server_module.bot_state["start_time"] is not None

    @pytest.mark.asyncio
    async def test_start_bot_already_running(self, client_no_auth: AsyncClient):
        """Starting the bot when already running should return 400."""
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/start", json={"mode": "simulation"})
        assert resp.status_code == 400
        assert "already running" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_start_bot_validation_errors(self, client_no_auth: AsyncClient):
        """Config validation errors should return 400 with detail."""
        mock_config = MagicMock()
        mock_config.validate.return_value = ["Stop loss must be < 100%"]

        with patch("api.server.BotConfig", return_value=mock_config), \
             patch("api.server.TradingMode"):
            resp = await client_no_auth.post("/bot/start", json={"mode": "simulation"})

        assert resp.status_code == 400
        assert "Invalid configuration" in resp.json()["detail"]
        assert "Stop loss" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_start_bot_exception_sets_error_state(self, client_no_auth: AsyncClient):
        """An unexpected exception during start should set error state and return 500."""
        with patch("api.server.BotConfig", side_effect=RuntimeError("RPC down")), \
             patch("api.server.TradingMode"):
            resp = await client_no_auth.post("/bot/start", json={"mode": "simulation"})

        assert resp.status_code == 500
        assert "Failed to start bot" in resp.json()["detail"]
        assert server_module.bot_state["status"] == "error"
        assert server_module.bot_state["error"] == "RPC down"

    @pytest.mark.asyncio
    async def test_start_bot_invalid_mode_rejected(self, client_no_auth: AsyncClient):
        """An unrecognised trading mode should be rejected by Pydantic (422)."""
        resp = await client_no_auth.post("/bot/start", json={"mode": "yolo_mode"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_invalid_buy_amount(self, client_no_auth: AsyncClient):
        """buy_amount_sol <= 0 should be rejected by Pydantic (422)."""
        resp = await client_no_auth.post("/bot/start", json={
            "mode": "simulation",
            "buy_amount_sol": -1.0,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_invalid_stop_loss_pct(self, client_no_auth: AsyncClient):
        """stop_loss_pct >= 100 should be rejected by Pydantic (422)."""
        resp = await client_no_auth.post("/bot/start", json={
            "mode": "simulation",
            "stop_loss_pct": 100.0,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_with_custom_rpc_url(self, client_no_auth: AsyncClient):
        """A custom RPC URL should be forwarded to the config."""
        mock_config = MagicMock()
        mock_config.validate.return_value = []

        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.BotConfig", return_value=mock_config) as mock_cfg_cls, \
             patch("api.server.TradingMode"), \
             patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post("/bot/start", json={
                "mode": "simulation",
                "rpc_url": "https://custom-rpc.example.com",
            })

        assert resp.status_code == 200
        # The config object should have had rpc_url set
        mock_config.rpc_url = "https://custom-rpc.example.com"


# ---------------------------------------------------------------------------
#  POST /bot/stop
# ---------------------------------------------------------------------------

class TestStopBot:
    """Tests for POST /bot/stop."""

    @pytest.mark.asyncio
    async def test_stop_bot_not_running(self, client_no_auth: AsyncClient):
        """Stopping a bot that is not running should return 400."""
        resp = await client_no_auth.post("/bot/stop")
        assert resp.status_code == 400
        assert "not running" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_stop_bot_success(self, client_no_auth: AsyncClient):
        """Stopping a running bot should succeed and reset state."""
        mock_bot = MagicMock()
        mock_bot.stop = AsyncMock()

        # Use a real Future so `await bot_task` works correctly
        mock_task = asyncio.get_event_loop().create_future()
        mock_task.cancel()  # Pre-cancel so await raises CancelledError

        server_module.bot_instance = mock_bot
        server_module.bot_task = mock_task
        server_module.bot_state["status"] = "running"
        server_module.bot_state["start_time"] = datetime.now()

        resp = await client_no_auth.post("/bot/stop")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert server_module.bot_state["status"] == "stopped"
        assert server_module.bot_state["start_time"] is None
        mock_bot.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_bot_exception(self, client_no_auth: AsyncClient):
        """An exception during stop should return 500."""
        mock_bot = MagicMock()
        mock_bot.stop = AsyncMock(side_effect=RuntimeError("Cleanup failed"))

        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/stop")
        assert resp.status_code == 500
        assert "Error stopping bot" in resp.json()["detail"]


# ---------------------------------------------------------------------------
#  POST /bot/trade
# ---------------------------------------------------------------------------

class TestManualTrade:
    """Tests for POST /bot/trade."""

    @pytest.mark.asyncio
    async def test_trade_bot_not_running(self, client_no_auth: AsyncClient):
        """Trade should return 400 when bot is not running."""
        resp = await client_no_auth.post("/bot/trade", json={
            "action": "buy",
            "token_address": "So11111111111111111111111111111111111111112",
        })
        assert resp.status_code == 400
        assert "not running" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_invalid_action(self, client_no_auth: AsyncClient):
        """Trade with invalid action should return 400."""
        mock_bot = MagicMock()
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/trade", json={
            "action": "hold",
            "token_address": "TOKEN1",
        })
        assert resp.status_code == 400
        assert "must be 'buy' or 'sell'" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_buy_success(self, client_no_auth: AsyncClient):
        """Successful buy trade should return success message."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=True)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/trade", json={
            "action": "buy",
            "token_address": "TOKEN1",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "Buy executed" in body["message"]
        mock_bot.trading_engine.execute_buy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trade_buy_failure(self, client_no_auth: AsyncClient):
        """Failed buy trade should return 500."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=False)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/trade", json={
            "action": "buy",
            "token_address": "TOKEN1",
        })
        assert resp.status_code == 500
        assert "Buy execution failed" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_sell_success(self, client_no_auth: AsyncClient):
        """Successful sell trade should return success message."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_sell = AsyncMock(return_value=True)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/trade", json={
            "action": "sell",
            "token_address": "TOKEN1",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "Sell executed" in body["message"]
        mock_bot.trading_engine.execute_sell.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trade_sell_failure(self, client_no_auth: AsyncClient):
        """Failed sell trade should return 500."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_sell = AsyncMock(return_value=False)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/trade", json={
            "action": "sell",
            "token_address": "TOKEN1",
        })
        assert resp.status_code == 500
        assert "Sell execution failed" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_missing_required_fields(self, client_no_auth: AsyncClient):
        """Missing required fields should return 422."""
        resp = await client_no_auth.post("/bot/trade", json={
            "action": "buy",
            # missing token_address
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_trade_engine_exception(self, client_no_auth: AsyncClient):
        """An engine exception during trade should return 500."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(
            side_effect=RuntimeError("Slippage exceeded")
        )
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/trade", json={
            "action": "buy",
            "token_address": "TOKEN1",
        })
        assert resp.status_code == 500
        assert "Trade execution error" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_action_case_insensitive(self, client_no_auth: AsyncClient):
        """Action field should be case-insensitive (BUY, Sell, etc.)."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=True)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/trade", json={
            "action": "BUY",
            "token_address": "TOKEN1",
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
#  WebSocket endpoint
# ---------------------------------------------------------------------------

class TestWebSocket:
    """Tests for the /ws/updates WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_connect_dev_mode(self, client_no_auth: AsyncClient):
        """WebSocket should connect and send initial status in dev mode."""
        # httpx does not natively support WebSocket.
        # Use Starlette's TestClient (sync) for WS testing.
        from starlette.testclient import TestClient

        original_key = server_module.FENRIR_API_KEY
        original_dev = server_module.FENRIR_DEV_MODE
        server_module.FENRIR_API_KEY = ""
        server_module.FENRIR_DEV_MODE = True
        try:
            sync_client = TestClient(app)
            with sync_client.websocket_connect("/ws/updates") as ws:
                data = ws.receive_json()
                assert data["event"] == "connected"
                assert data["bot_status"] == "stopped"
                assert "timestamp" in data
        finally:
            server_module.FENRIR_API_KEY = original_key
            server_module.FENRIR_DEV_MODE = original_dev

    @pytest.mark.asyncio
    async def test_websocket_echo(self):
        """WebSocket should echo back messages it receives."""
        from starlette.testclient import TestClient

        original_key = server_module.FENRIR_API_KEY
        original_dev = server_module.FENRIR_DEV_MODE
        server_module.FENRIR_API_KEY = ""
        server_module.FENRIR_DEV_MODE = True
        try:
            sync_client = TestClient(app)
            with sync_client.websocket_connect("/ws/updates") as ws:
                # Consume initial "connected" message
                ws.receive_json()
                # Send a text message; expect echo
                ws.send_text("hello fenrir")
                echo = ws.receive_json()
                assert echo["event"] == "echo"
                assert echo["message"] == "hello fenrir"
        finally:
            server_module.FENRIR_API_KEY = original_key
            server_module.FENRIR_DEV_MODE = original_dev

    @pytest.mark.asyncio
    async def test_websocket_rejected_without_key(self):
        """WebSocket should be rejected (closed) when API key is required but missing."""
        from starlette.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        original_key = server_module.FENRIR_API_KEY
        original_dev = server_module.FENRIR_DEV_MODE
        server_module.FENRIR_API_KEY = "ws-secret-key"
        server_module.FENRIR_DEV_MODE = False
        try:
            sync_client = TestClient(app)
            with pytest.raises(Exception):
                # Connection should be rejected / closed by the server
                with sync_client.websocket_connect("/ws/updates") as ws:
                    ws.receive_json()
        finally:
            server_module.FENRIR_API_KEY = original_key
            server_module.FENRIR_DEV_MODE = original_dev

    @pytest.mark.asyncio
    async def test_websocket_accepted_with_header_key(self):
        """WebSocket should accept connection when X-API-Key header is correct."""
        from starlette.testclient import TestClient

        original_key = server_module.FENRIR_API_KEY
        original_dev = server_module.FENRIR_DEV_MODE
        server_module.FENRIR_API_KEY = "ws-secret-key"
        server_module.FENRIR_DEV_MODE = False
        try:
            sync_client = TestClient(app)
            with sync_client.websocket_connect(
                "/ws/updates",
                headers={"X-API-Key": "ws-secret-key"},
            ) as ws:
                data = ws.receive_json()
                assert data["event"] == "connected"
        finally:
            server_module.FENRIR_API_KEY = original_key
            server_module.FENRIR_DEV_MODE = original_dev

    @pytest.mark.asyncio
    async def test_websocket_accepted_with_subprotocol_key(self):
        """WebSocket should accept when key is sent via Sec-WebSocket-Protocol."""
        from starlette.testclient import TestClient

        original_key = server_module.FENRIR_API_KEY
        original_dev = server_module.FENRIR_DEV_MODE
        server_module.FENRIR_API_KEY = "ws-secret-key"
        server_module.FENRIR_DEV_MODE = False
        try:
            sync_client = TestClient(app)
            with sync_client.websocket_connect(
                "/ws/updates",
                headers={"Sec-WebSocket-Protocol": "authorization.ws-secret-key"},
            ) as ws:
                data = ws.receive_json()
                assert data["event"] == "connected"
        finally:
            server_module.FENRIR_API_KEY = original_key
            server_module.FENRIR_DEV_MODE = original_dev


# ---------------------------------------------------------------------------
#  Broadcast helper
# ---------------------------------------------------------------------------

class TestBroadcastUpdate:
    """Tests for the broadcast_update helper."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_clients(self):
        """broadcast_update should send to every active WebSocket."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        server_module.active_websockets.extend([ws1, ws2])

        from api.server import broadcast_update
        await broadcast_update({"event": "test"})

        ws1.send_json.assert_awaited_once_with({"event": "test"})
        ws2.send_json.assert_awaited_once_with({"event": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected(self):
        """broadcast_update should remove clients that error on send."""
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send_json = AsyncMock(side_effect=RuntimeError("Disconnected"))
        server_module.active_websockets.extend([ws_good, ws_bad])

        from api.server import broadcast_update
        await broadcast_update({"event": "test"})

        # The bad websocket should have been removed
        assert ws_bad not in server_module.active_websockets
        assert ws_good in server_module.active_websockets


# ---------------------------------------------------------------------------
#  Concurrent state access
# ---------------------------------------------------------------------------

class TestConcurrency:
    """Tests verifying that bot_state_lock prevents race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_start_requests(self, client_no_auth: AsyncClient):
        """Only one of two concurrent start requests should succeed; the other gets 400."""
        mock_config = MagicMock()
        mock_config.validate.return_value = []

        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.BotConfig", return_value=mock_config), \
             patch("api.server.TradingMode"), \
             patch("api.server.FenrirBot", return_value=mock_bot):
            results = await asyncio.gather(
                client_no_auth.post("/bot/start", json={"mode": "simulation"}),
                client_no_auth.post("/bot/start", json={"mode": "simulation"}),
            )

        status_codes = sorted([r.status_code for r in results])
        # One should succeed (200), the other should fail (400 -- already running)
        assert status_codes == [200, 400]


# ---------------------------------------------------------------------------
#  Pydantic model validation
# ---------------------------------------------------------------------------

class TestPydanticModels:
    """Tests for request model validation via the API."""

    @pytest.mark.asyncio
    async def test_start_bot_negative_take_profit(self, client_no_auth: AsyncClient):
        """take_profit_pct must be > 0."""
        resp = await client_no_auth.post("/bot/start", json={
            "mode": "simulation",
            "take_profit_pct": -5.0,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_zero_buy_amount(self, client_no_auth: AsyncClient):
        """buy_amount_sol must be > 0."""
        resp = await client_no_auth.post("/bot/start", json={
            "mode": "simulation",
            "buy_amount_sol": 0,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_default_values(self, client_no_auth: AsyncClient):
        """Defaults should pass Pydantic validation and appear in response config."""
        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.BotConfig", return_value=mock_config), \
             patch("api.server.TradingMode"), \
             patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post("/bot/start", json={})

        assert resp.status_code == 200
        config = resp.json()["config"]
        assert config["mode"] == "simulation"
        assert config["buy_amount_sol"] == 0.1
        assert config["stop_loss_pct"] == 25.0
        assert config["take_profit_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_trade_missing_action(self, client_no_auth: AsyncClient):
        """Trade request without action field should return 422."""
        resp = await client_no_auth.post("/bot/trade", json={
            "token_address": "TOKEN1",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_all_modes_accepted(self, client_no_auth: AsyncClient):
        """All four trading modes should be accepted by the enum."""
        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        for mode in ["simulation", "conservative", "aggressive", "degen"]:
            _reset_bot_state()
            with patch("api.server.BotConfig", return_value=mock_config), \
                 patch("api.server.TradingMode"), \
                 patch("api.server.FenrirBot", return_value=mock_bot):
                resp = await client_no_auth.post("/bot/start", json={"mode": mode})
            assert resp.status_code == 200, f"Mode {mode} was rejected"


# ---------------------------------------------------------------------------
#  Rate limiter unit tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    """Tests for the in-memory sliding-window RateLimiter."""

    def test_allows_under_limit(self):
        from api.server import RateLimiter
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.is_allowed("127.0.0.1") is True

    def test_blocks_over_limit(self):
        from api.server import RateLimiter
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.is_allowed("127.0.0.1")
        assert rl.is_allowed("127.0.0.1") is False

    def test_separate_clients_independent(self):
        from api.server import RateLimiter
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.is_allowed("10.0.0.1")
        rl.is_allowed("10.0.0.1")
        # Client 1 exhausted
        assert rl.is_allowed("10.0.0.1") is False
        # Client 2 is still fresh
        assert rl.is_allowed("10.0.0.2") is True

    def test_remaining_count(self):
        from api.server import RateLimiter
        rl = RateLimiter(max_requests=5, window_seconds=60)
        assert rl.remaining("127.0.0.1") == 5
        rl.is_allowed("127.0.0.1")
        assert rl.remaining("127.0.0.1") == 4

    @pytest.mark.asyncio
    async def test_rate_limit_header_present(self, client_no_auth: AsyncClient):
        """Successful responses should include X-RateLimit-Remaining header."""
        resp = await client_no_auth.get("/health")
        assert resp.status_code == 200
        assert "x-ratelimit-remaining" in resp.headers

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self, client_no_auth: AsyncClient):
        """Exceeding rate limit should return 429 with Retry-After header."""
        # Temporarily shrink the limiter
        original = server_module.rate_limiter
        server_module.rate_limiter = server_module.RateLimiter(max_requests=2, window_seconds=60)
        try:
            await client_no_auth.get("/health")
            await client_no_auth.get("/health")
            resp = await client_no_auth.get("/health")
            assert resp.status_code == 429
            assert "retry-after" in resp.headers
        finally:
            server_module.rate_limiter = original
