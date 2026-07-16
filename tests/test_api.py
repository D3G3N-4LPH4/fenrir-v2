#!/usr/bin/env python3
"""
FENRIR Trading Bot - API Server Test Suite

Tests for the FastAPI REST API defined in api/server.py.
Covers authentication middleware, all HTTP endpoints, input validation,
bot state management, and the WebSocket endpoint.

Run with: pytest tests/test_api.py -v
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
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
    # Reset the sliding-window rate limiter so cross-test request accumulation
    # doesn't trip 429s in the full suite.
    server_module.rate_limiter.reset()
    yield
    _reset_bot_state()
    server_module.active_websockets.clear()
    server_module.rate_limiter.reset()


@pytest.fixture(autouse=True)
def _valid_config_env(monkeypatch):
    """Dummy (non-secret) keys so a real BotConfig passes validate() for every mode.

    tests/conftest.py clears all config env for determinism; BotConfig.validate then
    requires an AI key (ai_analysis_enabled defaults True) and, for any non-simulation
    mode, a wallet key. These let the start tests exercise the real config path.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("WALLET_PRIVATE_KEY", "test-wallet-key-not-a-real-key")


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
    async def test_cors_preflight_not_blocked_by_auth(self, client_with_key: AsyncClient):
        """OPTIONS preflight must pass the auth middleware so CORS can answer it.

        Browsers send preflight requests without the X-API-Key header; if the
        auth middleware rejects them with 403, every cross-origin request from
        the dashboard fails. The preflight should reach CORSMiddleware (200).
        """
        resp = await client_with_key.options(
            "/bot/status",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert resp.status_code != 403, "preflight was blocked by the auth middleware"
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    @pytest.mark.asyncio
    async def test_rate_limited_response_has_cors_header(self, client_with_key: AsyncClient):
        """A 429 must still carry Access-Control-Allow-Origin.

        CORSMiddleware is the outermost middleware, so even the rate-limiter's
        early 429 return gets CORS headers. Otherwise the browser reports a
        misleading "No 'Access-Control-Allow-Origin'" error instead of the 429.
        """
        original_max = server_module.rate_limiter.max_requests
        server_module.rate_limiter.max_requests = 0  # force an immediate 429
        try:
            resp = await client_with_key.get(
                "/bot/status",
                headers={"X-API-Key": "test-secret-key", "Origin": "http://localhost:5173"},
            )
            assert resp.status_code == 429
            assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
        finally:
            server_module.rate_limiter.max_requests = original_max

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

    @pytest.mark.asyncio
    async def test_status_ok_with_staged_config_without_mode(self, client_no_auth: AsyncClient):
        """A config staged via POST /bot/config while stopped has no 'mode' key.

        Regression: /bot/status did bot_state["config"]["mode"], which raised
        KeyError('mode') → 500 on every poll once the dashboard saved a setting
        while the bot was stopped.
        """
        server_module.bot_state["status"] = "stopped"
        server_module.bot_state["config"] = {"market_scanner_enabled": True, "buy_amount_sol": 0.01}

        resp = await client_no_auth.get("/bot/status")
        assert resp.status_code == 200
        assert resp.json()["mode"] is None


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
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post(
                "/bot/start",
                json={
                    "mode": "simulation",
                    "buy_amount_sol": 0.1,
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "simulation" in body["message"]
        assert server_module.bot_state["status"] == "running"
        assert server_module.bot_state["start_time"] is not None

    @pytest.mark.asyncio
    async def test_start_applies_mode_preset_when_unspecified(self, client_no_auth: AsyncClient):
        """A mode must trade its own preset, not StartBotRequest's old hardcoded defaults."""
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post("/bot/start", json={"mode": "conservative"})

        assert resp.status_code == 200
        cfg = resp.json()["config"]
        # CONSERVATIVE preset — previously these came back as 0.1 / 25.0 / 100.0.
        assert cfg["mode"] == "conservative"
        assert cfg["buy_amount_sol"] == 0.05
        assert cfg["stop_loss_pct"] == 15.0
        assert cfg["take_profit_pct"] == 75.0
        assert cfg["max_slippage_bps"] == 300  # never settable via the request at all
        assert cfg["ai_min_confidence_to_buy"] == 0.75

    @pytest.mark.asyncio
    async def test_start_explicit_value_overrides_preset(self, client_no_auth: AsyncClient):
        """An explicitly sent field still wins over the mode preset."""
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post(
                "/bot/start", json={"mode": "conservative", "stop_loss_pct": 5.0}
            )

        assert resp.status_code == 200
        cfg = resp.json()["config"]
        assert cfg["stop_loss_pct"] == 5.0  # explicit override
        assert cfg["take_profit_pct"] == 75.0  # untouched preset

    @pytest.mark.asyncio
    async def test_start_can_override_ai_confidence(self, client_no_auth: AsyncClient):
        """ai_min_confidence_to_buy is settable at start (degen presets a low 0.4 bar)."""
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot) as mock_fenrir:
            resp = await client_no_auth.post(
                "/bot/start", json={"mode": "degen", "ai_min_confidence_to_buy": 0.6}
            )

        assert resp.status_code == 200
        cfg = resp.json()["config"]
        assert cfg["ai_min_confidence_to_buy"] == 0.6  # override, not degen's 0.4
        assert cfg["stop_loss_pct"] == 50.0  # other degen preset values untouched
        # ...and it reached the config the bot was actually built with.
        assert mock_fenrir.call_args[0][0].ai_min_confidence_to_buy == 0.6

    @pytest.mark.asyncio
    async def test_start_uses_mode_ai_confidence_when_unset(self, client_no_auth: AsyncClient):
        """Unset → the mode's preset confidence applies."""
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post("/bot/start", json={"mode": "degen"})

        assert resp.status_code == 200
        assert resp.json()["config"]["ai_min_confidence_to_buy"] == 0.4

    @pytest.mark.asyncio
    async def test_start_rejects_out_of_range_ai_confidence(self, client_no_auth: AsyncClient):
        """Confidence outside 0-1 is rejected by Pydantic (422), not silently clamped."""
        resp = await client_no_auth.post(
            "/bot/start", json={"mode": "degen", "ai_min_confidence_to_buy": 1.5}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_response_reports_resolved_config_without_rpc_url(
        self, client_no_auth: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ):
        """Report what's actually running — and never leak the RPC URL (embeds API keys)."""
        monkeypatch.setenv("BUY_AMOUNT_SOL", "0.01")  # operator pin
        monkeypatch.setenv("SOLANA_RPC_URL", "https://rpc.example.com/?api-key=SECRET")
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post(
                "/bot/start", json={"mode": "simulation", "buy_amount_sol": 0.2}
            )

        assert resp.status_code == 200
        cfg = resp.json()["config"]
        # The env pin wins over the request, and the response says so rather than
        # echoing the request's 0.2 back at the operator.
        assert cfg["buy_amount_sol"] == 0.01
        assert "rpc_url" not in cfg
        assert "SECRET" not in resp.text

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

        with patch("api.server.BotConfig.from_mode", return_value=mock_config):
            resp = await client_no_auth.post("/bot/start", json={"mode": "simulation"})

        assert resp.status_code == 400
        assert "Invalid configuration" in resp.json()["detail"]
        assert "Stop loss" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_start_bot_exception_sets_error_state(self, client_no_auth: AsyncClient):
        """An unexpected exception during start should set error state and return 500."""
        with patch("api.server.BotConfig.from_mode", side_effect=RuntimeError("RPC down")):
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
        resp = await client_no_auth.post(
            "/bot/start",
            json={
                "mode": "simulation",
                "buy_amount_sol": -1.0,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_invalid_stop_loss_pct(self, client_no_auth: AsyncClient):
        """stop_loss_pct >= 100 should be rejected by Pydantic (422)."""
        resp = await client_no_auth.post(
            "/bot/start",
            json={
                "mode": "simulation",
                "stop_loss_pct": 100.0,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_with_custom_rpc_url(self, client_no_auth: AsyncClient):
        """A custom RPC URL should be forwarded to the config."""
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot) as mock_fenrir:
            resp = await client_no_auth.post(
                "/bot/start",
                json={
                    "mode": "simulation",
                    "rpc_url": "https://custom-rpc.example.com",
                },
            )

        assert resp.status_code == 200
        assert server_module.bot_instance is mock_bot
        # Assert the URL actually reached the config the bot was built with. (The
        # previous line here was `mock_config.rpc_url = ...` — an assignment, not an
        # assertion, so this never verified anything.)
        config_arg = mock_fenrir.call_args[0][0]
        assert config_arg.rpc_url == "https://custom-rpc.example.com"


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
        # A Future is awaitable/cancelable like a Task; cast to satisfy the
        # bot_task type annotation without changing runtime behavior.
        server_module.bot_task = cast("asyncio.Task[Any]", mock_task)
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
        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "buy",
                "token_address": "So11111111111111111111111111111111111111112",
            },
        )
        assert resp.status_code == 400
        assert "not running" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_invalid_action(self, client_no_auth: AsyncClient):
        """Trade with invalid action should return 400."""
        mock_bot = MagicMock()
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "hold",
                "token_address": "TOKEN1",
            },
        )
        assert resp.status_code == 400
        assert "must be 'buy' or 'sell'" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_buy_success(self, client_no_auth: AsyncClient):
        """Successful buy trade should return success message."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=True)
        mock_bot.jupiter.search_token = AsyncMock(return_value=None)  # unknown → curve path
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "buy",
                "token_address": "TOKEN1",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "Buy executed" in body["message"]
        mock_bot.trading_engine.execute_buy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_manual_buy_routes_migrated_token_off_curve(self, client_no_auth: AsyncClient):
        """A migrated/AMM token must carry tier+migrated so it routes via Jupiter.

        Regression: the handler fabricated token_data with no tier/migrated, so
        _is_non_curve_token was always False and EVERY manual buy was sent down the
        pump.fun bonding-curve path — which fails outright for an AMM token.
        """
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=True)
        mock_bot.jupiter.search_token = AsyncMock(
            return_value={
                "id": "MIGRATED_MINT",
                "symbol": "EST",
                "name": "Established",
                "mcap": 727_639.0,
                "liquidity": 69_700.0,
                "holderCount": 2375,
                "graduatedAt": "2026-07-14T05:11:34Z",
                "audit": {"topHoldersPercentage": 19.3, "devMints": 2543},
                "stats24h": {"priceChange": 572.0},
            }
        )
        mock_bot.scanner._tier = MagicMock(return_value="mid")
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post(
            "/bot/trade", json={"action": "buy", "token_address": "MIGRATED_MINT"}
        )

        assert resp.status_code == 200
        td = mock_bot.trading_engine.execute_buy.call_args[0][0]
        assert td["tier"] == "mid"
        assert td["migrated"] is True  # -> _is_non_curve_token -> Jupiter route
        assert td["market_cap_usd"] == 727_639.0  # real, not the old fabricated 50 SOL
        assert td["liquidity_usd"] == 69_700.0
        assert td["dev_mints"] == 2543
        assert "launch_time" not in td  # no longer claims an old coin just launched

    @pytest.mark.asyncio
    async def test_manual_buy_honors_amount(self, client_no_auth: AsyncClient):
        """The amount field was silently ignored — buys always used config size."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=True)
        mock_bot.jupiter.search_token = AsyncMock(return_value=None)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post(
            "/bot/trade",
            json={"action": "buy", "token_address": "TOKEN1", "amount": 0.02},
        )

        assert resp.status_code == 200
        assert mock_bot.trading_engine.execute_buy.call_args.kwargs["amount_sol"] == 0.02

    @pytest.mark.asyncio
    async def test_manual_buy_unknown_token_uses_curve_path(self, client_no_auth: AsyncClient):
        """A token Jupiter doesn't know is a live pump.fun curve token."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=True)
        mock_bot.jupiter.search_token = AsyncMock(return_value=None)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post(
            "/bot/trade", json={"action": "buy", "token_address": "FRESH_MINT"}
        )

        assert resp.status_code == 200
        td = mock_bot.trading_engine.execute_buy.call_args[0][0]
        assert td == {"token_address": "FRESH_MINT"}  # engine prices it off the curve

    @pytest.mark.asyncio
    async def test_trade_buy_failure(self, client_no_auth: AsyncClient):
        """Failed buy trade should return 500."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=False)
        mock_bot.jupiter.search_token = AsyncMock(return_value=None)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "buy",
                "token_address": "TOKEN1",
            },
        )
        assert resp.status_code == 500
        assert "Buy execution failed" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_sell_success(self, client_no_auth: AsyncClient):
        """Successful sell trade should return success message."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_sell = AsyncMock(return_value=True)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "sell",
                "token_address": "TOKEN1",
            },
        )
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

        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "sell",
                "token_address": "TOKEN1",
            },
        )
        assert resp.status_code == 500
        assert "Sell execution failed" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_missing_required_fields(self, client_no_auth: AsyncClient):
        """Missing required fields should return 422."""
        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "buy",
                # missing token_address
            },
        )
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

        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "buy",
                "token_address": "TOKEN1",
            },
        )
        assert resp.status_code == 500
        assert "Trade execution error" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_trade_action_case_insensitive(self, client_no_auth: AsyncClient):
        """Action field should be case-insensitive (BUY, Sell, etc.)."""
        mock_bot = MagicMock()
        mock_bot.trading_engine.execute_buy = AsyncMock(return_value=True)
        mock_bot.jupiter.search_token = AsyncMock(return_value=None)
        server_module.bot_instance = mock_bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "action": "BUY",
                "token_address": "TOKEN1",
            },
        )
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

        original_key = server_module.FENRIR_API_KEY
        original_dev = server_module.FENRIR_DEV_MODE
        server_module.FENRIR_API_KEY = "ws-secret-key"
        server_module.FENRIR_DEV_MODE = False
        try:
            sync_client = TestClient(app)
            with pytest.raises(Exception):  # noqa: B017
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
                subprotocols=["authorization.ws-secret-key"],
            ) as ws:
                # The server MUST echo the offered subprotocol back, or real
                # browsers reject the handshake (TestClient is more lenient).
                assert ws.accepted_subprotocol == "authorization.ws-secret-key"
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
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot):
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
        resp = await client_no_auth.post(
            "/bot/start",
            json={
                "mode": "simulation",
                "take_profit_pct": -5.0,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_zero_buy_amount(self, client_no_auth: AsyncClient):
        """buy_amount_sol must be > 0."""
        resp = await client_no_auth.post(
            "/bot/start",
            json={
                "mode": "simulation",
                "buy_amount_sol": 0,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_default_values(self, client_no_auth: AsyncClient):
        """An empty body should start in simulation mode on that mode's preset."""
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        with patch("api.server.FenrirBot", return_value=mock_bot):
            resp = await client_no_auth.post("/bot/start", json={})

        assert resp.status_code == 200
        config = resp.json()["config"]
        # These are the SIMULATION preset values (which the old hardcoded request
        # defaults happened to mirror) — now sourced from TRADING_PRESETS.
        assert config["mode"] == "simulation"
        assert config["buy_amount_sol"] == 0.1
        assert config["stop_loss_pct"] == 25.0
        assert config["take_profit_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_trade_missing_action(self, client_no_auth: AsyncClient):
        """Trade request without action field should return 422."""
        resp = await client_no_auth.post(
            "/bot/trade",
            json={
                "token_address": "TOKEN1",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_start_bot_all_modes_accepted(self, client_no_auth: AsyncClient):
        """All four trading modes should be accepted by the enum."""
        mock_bot = MagicMock()
        mock_bot.start = AsyncMock()

        for mode in ["simulation", "conservative", "aggressive", "degen"]:
            _reset_bot_state()
            with patch("api.server.FenrirBot", return_value=mock_bot):
                resp = await client_no_auth.post("/bot/start", json={"mode": mode})
            assert resp.status_code == 200, f"Mode {mode} was rejected"
            assert resp.json()["config"]["mode"] == mode


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


# ===================================================================
#  /bot/config — GET surface + POST live/staged updates
# ===================================================================


def _cfg_surface():
    """A JSON-serializable stand-in for the live BotConfig surface."""
    return SimpleNamespace(
        market_scanner_enabled=True,
        global_daily_sol_limit=0.5,
        buy_amount_sol=0.01,
        mid_cap_min_usd=200_000.0,
        large_cap_min_usd=1_000_000.0,
        scanner_max_positions=3,
        ai_min_confidence_to_buy=0.75,
    )


class TestConfigEndpoints:
    @pytest.mark.asyncio
    async def test_get_config_running_reflects_live(self, client_no_auth: AsyncClient):
        bot = MagicMock()
        bot.config = _cfg_surface()
        server_module.bot_instance = bot
        resp = await client_no_auth.get("/bot/config")
        assert resp.status_code == 200
        cfg = resp.json()["config"]
        assert cfg["market_scanner_enabled"] is True
        assert cfg["ai_min_confidence_to_buy"] == 0.75
        assert set(cfg) == set(server_module._CONFIG_SURFACE_FIELDS)

    @pytest.mark.asyncio
    async def test_get_config_stopped_no_config_400(self, client_no_auth: AsyncClient):
        resp = await client_no_auth.get("/bot/config")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_get_config_stopped_returns_staged(self, client_no_auth: AsyncClient):
        server_module.bot_state["config"] = {"buy_amount_sol": 0.02}
        resp = await client_no_auth.get("/bot/config")
        assert resp.status_code == 200
        assert resp.json()["config"]["buy_amount_sol"] == 0.02

    @pytest.mark.asyncio
    async def test_post_config_running_applies_and_returns_surface(
        self, client_no_auth: AsyncClient
    ):
        bot = MagicMock()
        bot.config = _cfg_surface()
        bot.apply_config_update = AsyncMock(return_value=[])
        server_module.bot_instance = bot
        resp = await client_no_auth.post("/bot/config", json={"market_scanner_enabled": False})
        assert resp.status_code == 200
        bot.apply_config_update.assert_awaited_once_with({"market_scanner_enabled": False})
        assert "market_scanner_enabled" in resp.json()["config"]

    @pytest.mark.asyncio
    async def test_post_config_running_validation_error_400(self, client_no_auth: AsyncClient):
        bot = MagicMock()
        bot.config = _cfg_surface()
        bot.apply_config_update = AsyncMock(return_value=["buy_amount_sol must be > 0"])
        server_module.bot_instance = bot
        resp = await client_no_auth.post("/bot/config", json={"buy_amount_sol": 0.01})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_post_config_stopped_stages(self, client_no_auth: AsyncClient):
        resp = await client_no_auth.post("/bot/config", json={"buy_amount_sol": 0.02})
        assert resp.status_code == 200
        staged = server_module.bot_state["config"]
        assert staged is not None
        assert staged["buy_amount_sol"] == 0.02
        assert "staged" in resp.json().get("note", "")

    @pytest.mark.asyncio
    async def test_post_config_empty_400(self, client_no_auth: AsyncClient):
        resp = await client_no_auth.post("/bot/config", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_post_config_rejects_out_of_range_confidence(self, client_no_auth: AsyncClient):
        # Pydantic bounds: ai_min_confidence_to_buy in [0, 1]
        resp = await client_no_auth.post("/bot/config", json={"ai_min_confidence_to_buy": 1.5})
        assert resp.status_code == 422


# ===================================================================
#  /bot/strategies/available + enable/disable (Strategies tab)
# ===================================================================


class TestStrategySwitching:
    @pytest.mark.asyncio
    async def test_available_when_stopped_all_unloaded(self, client_no_auth: AsyncClient):
        resp = await client_no_auth.get("/bot/strategies/available")
        assert resp.status_code == 200
        strategies = resp.json()["strategies"]
        assert len(strategies) == len(server_module.STRATEGY_REGISTRY)
        assert all(not s["loaded"] and not s["active"] for s in strategies)

    @pytest.mark.asyncio
    async def test_available_when_running_reflects_state(self, client_no_auth: AsyncClient):
        loaded = SimpleNamespace(
            strategy_id="sniper", state=SimpleNamespace(active=True, paused=False)
        )
        bot = MagicMock()
        bot.strategies = [loaded]
        server_module.bot_instance = bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.get("/bot/strategies/available")
        strategies = {s["strategy_id"]: s for s in resp.json()["strategies"]}
        assert strategies["sniper"]["loaded"] and strategies["sniper"]["active"]
        assert not strategies["reversal"]["loaded"]
        assert not strategies["reversal"]["active"]

    @pytest.mark.asyncio
    async def test_enable_running_calls_bot(self, client_no_auth: AsyncClient):
        bot = MagicMock()
        bot.set_strategy_enabled = AsyncMock(return_value=(True, "loaded"))
        server_module.bot_instance = bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/strategies/reversal/enable")
        assert resp.status_code == 200
        bot.set_strategy_enabled.assert_awaited_once_with("reversal", True)
        assert resp.json()["message"] == "loaded"

    @pytest.mark.asyncio
    async def test_disable_running_calls_bot(self, client_no_auth: AsyncClient):
        bot = MagicMock()
        bot.set_strategy_enabled = AsyncMock(return_value=(True, "paused"))
        server_module.bot_instance = bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/strategies/sniper/disable")
        assert resp.status_code == 200
        bot.set_strategy_enabled.assert_awaited_once_with("sniper", False)
        assert resp.json()["enabled"] is False

    @pytest.mark.asyncio
    async def test_enable_unknown_strategy_404(self, client_no_auth: AsyncClient):
        bot = MagicMock()
        bot.set_strategy_enabled = AsyncMock(return_value=(False, "Unknown strategy 'nope'"))
        server_module.bot_instance = bot
        server_module.bot_state["status"] = "running"

        resp = await client_no_auth.post("/bot/strategies/nope/enable")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_enable_when_stopped_400(self, client_no_auth: AsyncClient):
        resp = await client_no_auth.post("/bot/strategies/sniper/enable")
        assert resp.status_code == 400


# ===================================================================
#  CORS origins (dashboard access, incl. remote/Replit)
# ===================================================================


class TestCorsOrigins:
    def test_defaults_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("FENRIR_CORS_ORIGINS", raising=False)
        origins = server_module._parse_cors_origins()
        assert "http://localhost:5173" in origins
        assert all(o.startswith("http://localhost") for o in origins)

    def test_extra_origins_from_env_appended_and_deduped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "FENRIR_CORS_ORIGINS",
            "https://foo.repl.co, https://foo.replit.app , http://localhost:5173",
        )
        origins = server_module._parse_cors_origins()
        assert "https://foo.repl.co" in origins
        assert "https://foo.replit.app" in origins
        # localhost:5173 is a default — must not be duplicated
        assert origins.count("http://localhost:5173") == 1

    def test_no_wildcard_accepted(self, monkeypatch: pytest.MonkeyPatch):
        # '*' would be a literal origin string, never a wildcard — credentials
        # mode means it can't match all origins anyway. Just assert it's opt-in.
        monkeypatch.delenv("FENRIR_CORS_ORIGINS", raising=False)
        assert "*" not in server_module._parse_cors_origins()

    @pytest.mark.asyncio
    async def test_middleware_echoes_allowed_origin(self, client_no_auth: AsyncClient):
        resp = await client_no_auth.get("/health", headers={"Origin": "http://localhost:5173"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
