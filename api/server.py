#!/usr/bin/env python3
"""
Fenrir Trading Bot - FastAPI Backend
RESTful API for controlling the Fenrir pump.fun trading bot from the AI Terminal Agent
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Any, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Load .env so the server's own settings (FENRIR_API_KEY, FENRIR_DEV_MODE,
# FENRIR_CORS_ORIGINS, rate limits) can be set there — they're read via
# os.getenv at import time, before any BotConfig loads .env itself.
load_dotenv()

# Import Fenrir bot components. The real symbols are imported unconditionally
# for type checking; at runtime they may be absent, in which case fallbacks are
# used so the module still imports.
if TYPE_CHECKING:
    from fenrir import (
        BotConfig,
        FenrirBot,
        TradingMode,
    )
    from fenrir.events.bus import EventListener
    from fenrir.events.types import TradeEvent
    from fenrir.strategies import STRATEGY_REGISTRY
    from fenrir.trading.scanner import MarketScanner
else:
    try:
        from fenrir import (  # noqa: F401
            BotConfig,
            FenrirBot,
            FenrirLogger,
            JupiterSwapEngine,
            PositionManager,
            PumpFunMonitor,
            SolanaClient,
            TradingEngine,
            TradingMode,
            WalletManager,
        )
        from fenrir.events.bus import EventListener
        from fenrir.events.types import TradeEvent
        from fenrir.strategies import STRATEGY_REGISTRY
        from fenrir.trading.scanner import MarketScanner
    except ImportError:
        print("Warning: fenrir package not found.")
        print("   Make sure the fenrir/ package is in your Python path.")
        BotConfig = None
        FenrirBot = None
        MarketScanner = None
        STRATEGY_REGISTRY = {}
        TradingMode = None
        EventListener = object
        TradeEvent = object

# Configure logging with file handler for post-mortem analysis
LOG_DIR = os.getenv("FENRIR_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FenrirAPI")

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    _file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "fenrir_api.log"),
        maxBytes=10_000_000,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)8s | %(name)s | %(message)s")
    )
    logger.addHandler(_file_handler)


# ===================================================================
#                          RATE LIMITER
# ===================================================================


class RateLimiter:
    """
    In-memory sliding-window rate limiter.
    Tracks requests per client IP within a rolling time window.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # IP -> list of request timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Return True if the request is within the rate limit."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Prune expired timestamps
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > cutoff]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True

    def remaining(self, client_ip: str) -> int:
        """Requests remaining in the current window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        active = [t for t in self._requests.get(client_ip, []) if t > cutoff]
        return max(0, self.max_requests - len(active))

    def reset(self) -> None:
        """Clear all tracked request timestamps (used for test isolation)."""
        self._requests.clear()


# ===================================================================
#                     WEBSOCKET EVENT ADAPTER
# ===================================================================


class WebSocketEventAdapter(EventListener):
    """
    EventBus adapter that forwards every bot event to all connected
    WebSocket clients. Register on the bot's event_bus after start.
    """

    async def on_event(self, event: "TradeEvent") -> None:  # type: ignore[override]
        await broadcast_update(event.to_dict())


# Rate limit settings (configurable via env)
_rate_limit_max = int(os.getenv("FENRIR_RATE_LIMIT_MAX", "60"))
_rate_limit_window = int(os.getenv("FENRIR_RATE_LIMIT_WINDOW", "60"))
rate_limiter = RateLimiter(max_requests=_rate_limit_max, window_seconds=_rate_limit_window)

# API key for authentication (loaded from environment)
FENRIR_API_KEY = os.getenv("FENRIR_API_KEY", "")
FENRIR_DEV_MODE = os.getenv("FENRIR_DEV_MODE", "false").lower() == "true"


def _parse_cors_origins() -> list[str]:
    """Allowed browser origins for the dashboard.

    Local dev ports by default; add remote origins (e.g. a Replit-hosted
    dashboard) via FENRIR_CORS_ORIGINS as a comma-separated list. A wildcard
    ('*') is intentionally NOT supported because allow_credentials=True — the
    browser rejects the wildcard+credentials combination, so origins must be
    listed explicitly.
    """
    defaults = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3001"]
    extra = [o.strip() for o in os.getenv("FENRIR_CORS_ORIGINS", "").split(",") if o.strip()]
    # De-dupe while preserving order.
    return list(dict.fromkeys(defaults + extra))


CORS_ORIGINS = _parse_cors_origins()

# Global bot instance and state
bot_instance: FenrirBot | None = None
bot_task: asyncio.Task | None = None


class BotState(TypedDict):
    """Mutable global bot state tracked by the API."""

    status: str
    start_time: datetime | None
    config: dict[str, Any] | None
    error: str | None


bot_state: BotState = {"status": "stopped", "start_time": None, "config": None, "error": None}
# Lock protecting bot_state and bot_instance mutations from concurrent requests
bot_state_lock = asyncio.Lock()

# WebSocket connections for real-time updates
active_websockets: list[WebSocket] = []


# ===================================================================
#                          LIFESPAN (replaces deprecated on_event)
# ===================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("Fenrir API server starting...")
    logger.info("Available at http://localhost:8000")
    logger.info("API docs at http://localhost:8000/docs")
    if not FENRIR_API_KEY:
        if FENRIR_DEV_MODE:
            logger.warning(
                "FENRIR_API_KEY not set and FENRIR_DEV_MODE=true. "
                "API is running WITHOUT authentication."
            )
        else:
            logger.warning(
                "FENRIR_API_KEY not set. Protected endpoints will return 500. "
                "Set FENRIR_API_KEY or FENRIR_DEV_MODE=true."
            )
    yield
    # Shutdown
    logger.info("Fenrir API server shutting down...")
    if bot_instance and bot_state["status"] == "running":
        try:
            await bot_instance.stop()
        except Exception as e:
            logger.error(f"Error stopping bot during shutdown: {e}")
    for websocket in active_websockets:
        try:
            await websocket.close()
        except Exception as e:
            logger.debug(f"WebSocket close error during shutdown: {e}")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Fenrir Trading Bot API",
    description="Control and monitor Fenrir pump.fun trading bot",
    version="1.0.0",
    lifespan=lifespan,
)

# NOTE: CORSMiddleware is registered LAST (see below) so it wraps the auth and
# rate-limit middlewares as the outermost layer. Otherwise their early returns
# (403/429) skip CORS and arrive at the browser with no Access-Control-Allow-Origin
# header, which surfaces as a misleading "CORS policy" error instead of the real
# status code.


# ===================================================================
#                        AUTHENTICATION MIDDLEWARE
# ===================================================================


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Require X-API-Key header on all endpoints except health checks."""
    # Let CORS preflights through: browsers send OPTIONS with no credentials or
    # custom headers by design, so rejecting them here (before CORSMiddleware can
    # answer) breaks every cross-origin request from the dashboard. Preflights
    # carry no side effects, so this is safe.
    if request.method == "OPTIONS":
        return await call_next(request)

    # Skip auth for public endpoints and OpenAPI docs
    public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    if request.url.path in public_paths:
        return await call_next(request)

    # Require API key unless explicitly running in dev mode
    if not FENRIR_API_KEY:
        if FENRIR_DEV_MODE:
            return await call_next(request)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "FENRIR_API_KEY not configured. Set it or enable FENRIR_DEV_MODE=true."
            },
        )

    # Skip auth for WebSocket upgrade (handled in WS endpoint)
    if request.headers.get("upgrade", "").lower() == "websocket":
        return await call_next(request)

    provided_key = request.headers.get("X-API-Key", "")
    if provided_key != FENRIR_API_KEY:
        return JSONResponse(status_code=403, content={"detail": "Invalid or missing API key"})

    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-IP rate limiting on all requests."""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": str(rate_limiter.window_seconds)},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(rate_limiter.remaining(client_ip))
    return response


# CORS — registered last so it is the OUTERMOST middleware and every response
# (including 403/429 early-returns from the middlewares above) carries the
# Access-Control-Allow-Origin header. Restricted to specific origins/methods/
# headers; add remote dashboard origins (e.g. Replit) via FENRIR_CORS_ORIGINS.
# Wildcard is unsupported because allow_credentials=True (browsers reject '*' + creds).
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ===================================================================
#                            PYDANTIC MODELS
# ===================================================================


class TradingModeEnum(str, Enum):
    """Trading mode options"""

    SIMULATION = "simulation"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    DEGEN = "degen"


class StartBotRequest(BaseModel):
    """Request to start the trading bot.

    Every trading field defaults to ``None`` meaning "use the mode's preset"
    (``TRADING_PRESETS``). Previously these carried hardcoded defaults that were
    always passed to ``BotConfig``, so the preset never applied and every mode
    traded identically — "degen" ran with 0.1 SOL / 25% stop instead of its own
    0.5 SOL / 50% stop. Send a value only to override that mode's preset.

    NOTE: Private keys must NEVER be sent via API. They are loaded
    exclusively from the WALLET_PRIVATE_KEY environment variable.
    """

    mode: TradingModeEnum = Field(default=TradingModeEnum.SIMULATION)
    buy_amount_sol: float | None = Field(default=None, gt=0, description="SOL per trade")
    stop_loss_pct: float | None = Field(default=None, gt=0, lt=100, description="Stop loss %")
    take_profit_pct: float | None = Field(default=None, gt=0, description="Take profit %")
    trailing_stop_pct: float | None = Field(default=None, gt=0, description="Trailing stop %")
    max_position_age_minutes: int | None = Field(default=None, gt=0, description="Max hold time")
    min_initial_liquidity_sol: float | None = Field(default=None, gt=0, description="Min liquidity")
    max_initial_market_cap_sol: float | None = Field(
        default=None, gt=0, description="Max market cap"
    )
    ai_min_confidence_to_buy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum AI confidence to buy (0-1)"
    )
    rpc_url: str | None = Field(default=None, description="Custom RPC URL")

    def preset_overrides(self) -> dict[str, Any]:
        """Only the trading fields the caller explicitly set (None = use preset)."""
        return {
            name: value
            for name, value in self.model_dump(
                exclude={"mode", "rpc_url"}, exclude_none=True
            ).items()
        }


class UpdateConfigRequest(BaseModel):
    """Partial live-config patch from the dashboard settings panel.

    All fields optional — only the ones the user changed are sent. Mirrors the
    live-tunable subset of BotConfig; applied via FenrirBot.apply_config_update
    (which performs the side-effects, e.g. starting/stopping the scanner) when
    the bot is running, or staged into bot_state for the next start otherwise.
    """

    market_scanner_enabled: bool | None = None
    global_daily_sol_limit: float | None = Field(default=None, ge=0)
    buy_amount_sol: float | None = Field(default=None, gt=0)
    mid_cap_min_usd: float | None = Field(default=None, ge=0)
    large_cap_min_usd: float | None = Field(default=None, ge=0)
    scanner_max_positions: int | None = Field(default=None, ge=0)
    ai_min_confidence_to_buy: float | None = Field(default=None, ge=0.0, le=1.0)


# Fields surfaced by GET /bot/config and echoed back after a POST.
_CONFIG_SURFACE_FIELDS = (
    "market_scanner_enabled",
    "global_daily_sol_limit",
    "buy_amount_sol",
    "mid_cap_min_usd",
    "large_cap_min_usd",
    "scanner_max_positions",
    "ai_min_confidence_to_buy",
)


def _config_surface(cfg) -> dict[str, Any]:
    """Extract the dashboard-tunable subset from a live BotConfig."""
    return {name: getattr(cfg, name) for name in _CONFIG_SURFACE_FIELDS}


# Effective trading params reported after a start. Deliberately excludes rpc_url —
# SOLANA_RPC_URL commonly embeds a provider API key and must not leave the process.
_RESOLVED_START_FIELDS = (
    "buy_amount_sol",
    "stop_loss_pct",
    "take_profit_pct",
    "trailing_stop_pct",
    "max_position_age_minutes",
    "min_initial_liquidity_sol",
    "max_initial_market_cap_sol",
    "max_slippage_bps",
    "priority_fee_lamports",
    "ai_min_confidence_to_buy",
)


def _resolved_start_config(cfg) -> dict[str, Any]:
    """What is ACTUALLY running: mode preset + explicit overrides + env pins.

    Reported instead of echoing the request, which hid both the preset values and
    env pins (e.g. BUY_AMOUNT_SOL) from the operator.
    """
    resolved: dict[str, Any] = {"mode": cfg.mode.value}
    resolved.update({name: getattr(cfg, name) for name in _RESOLVED_START_FIELDS})
    return resolved


class ManualTradeRequest(BaseModel):
    """Manual trade execution"""

    action: str = Field(..., description="'buy' or 'sell'")
    token_address: str = Field(..., description="Token mint address")
    amount: float | None = Field(default=None, description="Amount (SOL for buy, tokens for sell)")


async def _manual_buy_token_data(address: str) -> dict[str, Any]:
    """Resolve REAL market data for a manual buy — never fabricate it.

    This used to hand execute_buy a hardcoded dict (10 SOL liquidity, 50 SOL mcap,
    launch_time=now). That was wrong three ways: the numbers were fiction, it told
    the exit logic an established coin had just launched, and — because it carried
    no ``tier``/``migrated`` — ``_is_non_curve_token`` sent EVERY address down the
    pump.fun bonding-curve path, which simply fails for AMM/migrated tokens.

    Jupiter knows migrated/AMM tokens (``graduatedAt``, mcap, liquidity, audit), so
    reuse the market scanner's own candidate builder for them. Anything Jupiter
    doesn't know is treated as a live pump.fun curve token and priced from the curve
    by the engine.
    """
    assert bot_instance is not None
    tok = await bot_instance.jupiter.search_token(address)
    if tok:
        tier = bot_instance.scanner._tier(tok.get("mcap") or 0.0)
        if tier or tok.get("graduatedAt"):
            # A graduated token trades on an AMM even if it's below the mid-cap
            # threshold — route it off the curve regardless of tier.
            return MarketScanner._build_candidate(tok, tier or "mid")
    return {"token_address": address}


class BotStatusResponse(BaseModel):
    """Bot status response"""

    status: str
    mode: str | None = None
    uptime_seconds: float | None = None
    positions_count: int = 0
    portfolio: dict[str, Any] | None = None
    error: str | None = None


# ===================================================================
#                            HELPER FUNCTIONS
# ===================================================================


async def broadcast_update(message: dict):
    """Broadcast update to all connected WebSocket clients"""
    disconnected = []
    for websocket in active_websockets:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            disconnected.append(websocket)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)


def get_uptime() -> float | None:
    """Calculate bot uptime in seconds"""
    start_time = bot_state["start_time"]
    if start_time is not None:
        return (datetime.now() - start_time).total_seconds()
    return None


# ===================================================================
#                            API ENDPOINTS
# ===================================================================


@app.get("/")
async def root():
    """API health check"""
    return {"service": "Fenrir Trading Bot API", "version": "1.0.0", "status": "operational"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "bot_status": bot_state["status"],
        "uptime_seconds": get_uptime(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/bot/start")
async def start_bot(request: StartBotRequest):
    """Start the Fenrir trading bot.

    Private keys are loaded from the WALLET_PRIVATE_KEY env var, never from API.
    """
    global bot_instance, bot_task, bot_state

    async with bot_state_lock:
        if bot_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Bot is already running")

        try:
            # from_mode applies the mode's TRADING_PRESETS, then the caller's explicit
            # overrides; BotConfig.__post_init__ env pins (e.g. BUY_AMOUNT_SOL) win last.
            config = BotConfig.from_mode(
                TradingMode(request.mode.value), **request.preset_overrides()
            )

            if request.rpc_url:
                config.rpc_url = request.rpc_url

            # Private key is loaded from environment by BotConfig.__post_init__

            errors = config.validate()
            if errors:
                raise HTTPException(
                    status_code=400, detail=f"Invalid configuration: {', '.join(errors)}"
                )

            bot_instance = FenrirBot(config)
            # Bridge all EventBus events to WebSocket clients
            bot_instance.event_bus.register(WebSocketEventAdapter())
            bot_task = asyncio.create_task(bot_instance.start())

            bot_state["status"] = "running"
            bot_state["start_time"] = datetime.now()
            bot_state["config"] = _resolved_start_config(config)
            bot_state["error"] = None

        except HTTPException:
            raise
        except Exception as e:
            bot_state["status"] = "error"
            bot_state["error"] = str(e)
            logger.error(f"Failed to start bot: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}") from e

    await broadcast_update(
        {
            "event": "bot_started",
            "mode": request.mode.value,
            "timestamp": datetime.now().isoformat(),
        }
    )

    logger.info(f"Fenrir bot started in {request.mode.value} mode")

    return {
        "status": "success",
        "message": f"Fenrir bot started in {request.mode.value} mode",
        "config": _resolved_start_config(config),
    }


@app.post("/bot/stop")
async def stop_bot():
    """Stop the Fenrir trading bot"""
    global bot_instance, bot_task, bot_state

    async with bot_state_lock:
        if bot_state["status"] != "running":
            raise HTTPException(status_code=400, detail="Bot is not running")

        try:
            if bot_instance:
                await bot_instance.stop()
            if bot_task:
                bot_task.cancel()
                try:
                    await bot_task
                except asyncio.CancelledError:
                    pass

            bot_state["status"] = "stopped"
            bot_state["start_time"] = None
            bot_state["error"] = None

        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            raise HTTPException(status_code=500, detail=f"Error stopping bot: {str(e)}") from e

    await broadcast_update({"event": "bot_stopped", "timestamp": datetime.now().isoformat()})

    logger.info("Fenrir bot stopped")
    return {"status": "success", "message": "Fenrir bot stopped successfully"}


@app.get("/bot/status", response_model=BotStatusResponse)
async def get_bot_status():
    """Get current bot status and portfolio summary"""
    try:
        portfolio = None
        positions_count = 0

        if bot_instance and bot_state["status"] == "running":
            portfolio = bot_instance.positions.get_portfolio_summary()
            positions_count = len(bot_instance.positions.positions)

        return BotStatusResponse(
            status=bot_state["status"],
            # .get() not [...]: a config staged via POST /bot/config while the bot
            # is stopped is a partial surface dict with no "mode" key. Using [...]
            # there raised KeyError('mode') → every /bot/status poll 500'd.
            mode=bot_state["config"].get("mode") if bot_state["config"] else None,
            uptime_seconds=get_uptime(),
            positions_count=positions_count,
            portfolio=portfolio,
            error=bot_state["error"],
        )

    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}") from e


@app.get("/bot/positions")
async def get_positions():
    """Get all open positions"""
    if not bot_instance or bot_state["status"] != "running":
        return {"positions": []}

    try:
        positions = []
        for token_address, position in bot_instance.positions.positions.items():
            positions.append(
                {
                    "token_address": token_address,
                    "token_symbol": getattr(position, "token_symbol", "???"),
                    "strategy_id": getattr(position, "strategy_id", "default"),
                    "entry_time": position.entry_time.isoformat(),
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "amount_tokens": position.amount_tokens,
                    "amount_sol_invested": position.amount_sol_invested,
                    "pnl_percent": position.get_pnl_percent(),
                    "pnl_sol": position.get_pnl_sol(),
                    "peak_price": position.peak_price,
                    "trailing_stop_override_pct": getattr(
                        position, "trailing_stop_override_pct", None
                    ),
                    "ouroboros_triggered": getattr(position, "trailing_stop_override_pct", None)
                    is not None,
                }
            )

        return {"positions": positions}

    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting positions: {str(e)}") from e


@app.get("/discover")
async def get_discoveries(
    chain: str | None = None,
    filter_name: str | None = Query(default=None, alias="filter"),
    min_score: float = 0.0,
    limit: int = 100,
):
    """Ranked multi-chain discovery results (discovery-only; not traded).

    Query params: ``chain`` (solana|ethereum|bnb|base), ``filter`` (filter name),
    ``min_score`` (0–100), ``limit``. Empty when discovery is disabled/not running.
    """
    scanner = getattr(bot_instance, "discovery_scanner", None) if bot_instance else None
    if scanner is None:
        return {"results": [], "count": 0, "enabled": False}

    from fenrir.discovery.models import Chain

    ch: Chain | None = None
    if chain:
        try:
            ch = Chain(chain.strip().lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"unknown chain '{chain}'") from e

    results = scanner.get_results(
        chain=ch, filter_name=filter_name, min_score=min_score, limit=limit
    )
    return {"results": [r.to_dict() for r in results], "count": len(results), "enabled": True}


@app.get("/discover/filters")
async def get_discovery_filters():
    """The 3 filter definitions (thresholds) — the discovery filter menu."""
    from dataclasses import asdict

    from fenrir.discovery.filters import DEFAULT_THRESHOLDS

    return {name.value: asdict(thr) for name, thr in DEFAULT_THRESHOLDS.items()}


@app.get("/discover/config")
async def get_discovery_config():
    """Current discovery config surface (chains/filters/weights/thresholds)."""
    from dataclasses import asdict

    if bot_instance is None:
        return {"enabled": False}
    scanner = getattr(bot_instance, "discovery_scanner", None)
    cfg = scanner.config if scanner is not None else bot_instance.config.build_discovery_config()
    return {
        "enabled": cfg.enabled,
        "running": scanner is not None,
        "chains": [c.value for c in cfg.chains],
        "filters": [f.value for f in cfg.filters],
        "interval_seconds": cfg.interval_seconds,
        "min_alert_score": cfg.min_alert_score,
        "weights": asdict(cfg.weights),
    }


@app.post("/bot/trade")
async def execute_manual_trade(request: ManualTradeRequest):
    """Execute manual buy or sell trade"""
    if not bot_instance or bot_state["status"] != "running":
        raise HTTPException(status_code=400, detail="Bot is not running")

    try:
        if request.action.lower() == "buy":
            token_data = await _manual_buy_token_data(request.token_address)
            success = await bot_instance.trading_engine.execute_buy(
                token_data, amount_sol=request.amount
            )

            if success:
                await broadcast_update(
                    {
                        "event": "manual_buy",
                        "token": request.token_address,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return {"status": "success", "message": f"Buy executed for {request.token_address}"}
            else:
                raise HTTPException(status_code=500, detail="Buy execution failed")

        elif request.action.lower() == "sell":
            success = await bot_instance.trading_engine.execute_sell(
                request.token_address, "Manual sell via API"
            )

            if success:
                await broadcast_update(
                    {
                        "event": "manual_sell",
                        "token": request.token_address,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return {
                    "status": "success",
                    "message": f"Sell executed for {request.token_address}",
                }
            else:
                raise HTTPException(status_code=500, detail="Sell execution failed")

        else:
            raise HTTPException(status_code=400, detail="Action must be 'buy' or 'sell'")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=f"Trade execution error: {str(e)}") from e


@app.get("/bot/full-status")
async def get_full_status():
    """Full bot status including AI brain, strategies, budget, audit, and Ouroboros stats."""
    if not bot_instance or bot_state["status"] != "running":
        return {"running": False, "status": bot_state["status"]}
    try:
        status = bot_instance.get_full_status()
        # Append Ouroboros detector stats if available
        if hasattr(bot_instance, "dump_detector"):
            status["ouroboros"] = bot_instance.dump_detector.get_stats()
        if hasattr(bot_instance, "market_geometry"):
            status["market_geometry"] = bot_instance.market_geometry.get_stats()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/bot/strategies")
async def get_strategies():
    """List all active strategies and their runtime state."""
    if not bot_instance or bot_state["status"] != "running":
        return {"strategies": []}
    return {"strategies": [s.get_status() for s in bot_instance.strategies]}


@app.post("/bot/strategies/{strategy_id}/pause")
async def pause_strategy(strategy_id: str):
    """Pause a strategy (stops new entries, keeps existing positions)."""
    if not bot_instance or bot_state["status"] != "running":
        raise HTTPException(status_code=400, detail="Bot is not running")
    for strategy in bot_instance.strategies:
        if strategy.strategy_id == strategy_id:
            strategy.pause()
            await broadcast_update(
                {
                    "event": "strategy_paused",
                    "strategy_id": strategy_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return {"status": "success", "strategy_id": strategy_id, "paused": True}
    raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")


@app.post("/bot/strategies/{strategy_id}/resume")
async def resume_strategy(strategy_id: str):
    """Resume a paused strategy."""
    if not bot_instance or bot_state["status"] != "running":
        raise HTTPException(status_code=400, detail="Bot is not running")
    for strategy in bot_instance.strategies:
        if strategy.strategy_id == strategy_id:
            strategy.resume()
            await broadcast_update(
                {
                    "event": "strategy_resumed",
                    "strategy_id": strategy_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return {"status": "success", "strategy_id": strategy_id, "paused": False}
    raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")


@app.get("/bot/strategies/available")
async def get_available_strategies():
    """All registered strategies with their live state (for the Strategies tab).

    active = loaded and not paused; loaded strategies also report paused. When
    the bot isn't running, everything reports not-loaded/inactive.
    """
    loaded = {}
    if bot_instance is not None and bot_state["status"] == "running":
        loaded = {s.strategy_id: s for s in bot_instance.strategies}
    strategies = []
    for sid, cls in STRATEGY_REGISTRY.items():
        s = loaded.get(sid)
        strategies.append(
            {
                "strategy_id": sid,
                "display_name": getattr(cls, "display_name", sid),
                "description": getattr(cls, "description", ""),
                "uses_market_data": getattr(cls, "uses_market_data", False),
                "loaded": s is not None,
                "active": bool(s is not None and s.state.active and not s.state.paused),
                "paused": bool(s is not None and s.state.paused),
            }
        )
    return {"strategies": strategies}


@app.post("/bot/strategies/{strategy_id}/enable")
async def enable_strategy(strategy_id: str):
    """Activate a strategy live — loads it if not already running, else resumes it."""
    if not bot_instance or bot_state["status"] != "running":
        raise HTTPException(status_code=400, detail="Bot is not running")
    ok, message = await bot_instance.set_strategy_enabled(strategy_id, True)
    if not ok:
        raise HTTPException(status_code=404, detail=message)
    await broadcast_update(
        {
            "event": "strategy_enabled",
            "strategy_id": strategy_id,
            "timestamp": datetime.now().isoformat(),
        }
    )
    return {"status": "success", "strategy_id": strategy_id, "enabled": True, "message": message}


@app.post("/bot/strategies/{strategy_id}/disable")
async def disable_strategy(strategy_id: str):
    """Deactivate a strategy live (pauses it — existing positions are kept)."""
    if not bot_instance or bot_state["status"] != "running":
        raise HTTPException(status_code=400, detail="Bot is not running")
    ok, message = await bot_instance.set_strategy_enabled(strategy_id, False)
    if not ok:
        raise HTTPException(status_code=404, detail=message)
    await broadcast_update(
        {
            "event": "strategy_disabled",
            "strategy_id": strategy_id,
            "timestamp": datetime.now().isoformat(),
        }
    )
    return {"status": "success", "strategy_id": strategy_id, "enabled": False, "message": message}


@app.get("/bot/config")
async def get_bot_config():
    """Get the dashboard-tunable config surface.

    Reflects the live BotConfig when the bot is running (not the stale request
    payload that started it); falls back to the staged bot_state config otherwise.
    """
    if bot_instance is not None:
        return {"status": "success", "config": _config_surface(bot_instance.config)}
    if not bot_state["config"]:
        raise HTTPException(status_code=400, detail="Bot has not been configured yet")
    return {"status": "success", "config": bot_state["config"]}


@app.post("/bot/config")
async def update_bot_config(patch: UpdateConfigRequest):
    """Apply a partial config change from the dashboard.

    Running bot: delegates to FenrirBot.apply_config_update, which mutates the
    live config AND performs the side-effects (start/stop the scanner task,
    re-apply the global SOL cap, refresh strategy buy amounts) so changes take
    effect on the next cycle with no restart. Stopped bot: staged into
    bot_state["config"] for the next /bot/start.
    """
    updates = patch.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided")

    if bot_instance is not None:
        errors = await bot_instance.apply_config_update(updates)
        if errors:
            raise HTTPException(
                status_code=400, detail=f"Invalid configuration: {', '.join(errors)}"
            )
        await broadcast_update(
            {
                "event": "config_updated",
                "fields": list(updates.keys()),
                "timestamp": datetime.now().isoformat(),
            }
        )
        logger.info(f"Live config updated: {updates}")
        return {"status": "success", "config": _config_surface(bot_instance.config)}

    # Bot not running yet — stage the change for the next /bot/start.
    bot_state["config"] = {**(bot_state["config"] or {}), **updates}
    return {"status": "success", "config": bot_state["config"], "note": "staged for next start"}


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time bot updates.

    Authentication: pass the API key via the Sec-WebSocket-Protocol header
    using the format 'authorization.<key>'. The server selects that subprotocol
    to complete the handshake. Alternatively, the X-API-Key header is accepted
    for clients that support custom headers on upgrade requests.
    """
    # If the client offered an 'authorization.<key>' subprotocol, we MUST echo it
    # back in the handshake (websocket.accept(subprotocol=...)) — browsers reject
    # the connection when the server doesn't select one of the offered protocols.
    offered = [
        p.strip()
        for p in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if p.strip()
    ]
    auth_proto = next((p for p in offered if p.startswith("authorization.")), None)

    if FENRIR_API_KEY:
        # Prefer the subprotocol (works in browsers); fall back to X-API-Key header.
        api_key = auth_proto[len("authorization.") :] if auth_proto else ""
        if not api_key:
            api_key = websocket.headers.get("x-api-key", "")
        if api_key != FENRIR_API_KEY:
            await websocket.close(code=4003, reason="Invalid API key")
            return

    await websocket.accept(subprotocol=auth_proto)
    active_websockets.append(websocket)

    try:
        await websocket.send_json(
            {
                "event": "connected",
                "bot_status": bot_state["status"],
                "timestamp": datetime.now().isoformat(),
            }
        )

        while True:
            try:
                data = await websocket.receive_text()
                await websocket.send_json({"event": "echo", "message": data})
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104
        port=8000,
        log_level="info",
    )
