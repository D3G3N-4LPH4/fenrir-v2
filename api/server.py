#!/usr/bin/env python3
"""
Fenrir Trading Bot - FastAPI Backend
RESTful API for controlling the Fenrir pump.fun trading bot from the AI Terminal Agent
"""

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import asyncio
import json
import logging
from datetime import datetime
from enum import Enum

# Import Fenrir bot components
try:
    from fenrir import (
        FenrirBot, BotConfig, TradingMode, FenrirLogger,
        WalletManager, SolanaClient, JupiterSwapEngine,
        PositionManager, TradingEngine, PumpFunMonitor
    )
except ImportError:
    print("Warning: fenrir package not found.")
    print("   Make sure the fenrir/ package is in your Python path.")

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
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
    ))
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
        self._requests: Dict[str, List[float]] = defaultdict(list)

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


# Rate limit settings (configurable via env)
_rate_limit_max = int(os.getenv("FENRIR_RATE_LIMIT_MAX", "60"))
_rate_limit_window = int(os.getenv("FENRIR_RATE_LIMIT_WINDOW", "60"))
rate_limiter = RateLimiter(max_requests=_rate_limit_max, window_seconds=_rate_limit_window)

# API key for authentication (loaded from environment)
FENRIR_API_KEY = os.getenv("FENRIR_API_KEY", "")
FENRIR_DEV_MODE = os.getenv("FENRIR_DEV_MODE", "false").lower() == "true"

# Global bot instance and state
bot_instance: Optional[FenrirBot] = None
bot_task: Optional[asyncio.Task] = None
bot_state = {
    "status": "stopped",
    "start_time": None,
    "config": None,
    "error": None
}
# Lock protecting bot_state and bot_instance mutations from concurrent requests
bot_state_lock = asyncio.Lock()

# WebSocket connections for real-time updates
active_websockets: List[WebSocket] = []


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

# CORS configuration - restrict to specific methods and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ===================================================================
#                        AUTHENTICATION MIDDLEWARE
# ===================================================================

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Require X-API-Key header on all endpoints except health checks."""
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
            content={"detail": "FENRIR_API_KEY not configured. Set it or enable FENRIR_DEV_MODE=true."}
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

    NOTE: Private keys must NEVER be sent via API. They are loaded
    exclusively from the WALLET_PRIVATE_KEY environment variable.
    """
    mode: TradingModeEnum = Field(default=TradingModeEnum.SIMULATION)
    buy_amount_sol: float = Field(default=0.1, gt=0, description="SOL per trade")
    stop_loss_pct: float = Field(default=25.0, gt=0, lt=100, description="Stop loss %")
    take_profit_pct: float = Field(default=100.0, gt=0, description="Take profit %")
    trailing_stop_pct: float = Field(default=15.0, gt=0, description="Trailing stop %")
    max_position_age_minutes: int = Field(default=60, gt=0, description="Max hold time")
    min_initial_liquidity_sol: float = Field(default=5.0, gt=0, description="Min liquidity")
    max_initial_market_cap_sol: float = Field(default=100.0, gt=0, description="Max market cap")
    rpc_url: Optional[str] = Field(default=None, description="Custom RPC URL")


class ManualTradeRequest(BaseModel):
    """Manual trade execution"""
    action: str = Field(..., description="'buy' or 'sell'")
    token_address: str = Field(..., description="Token mint address")
    amount: Optional[float] = Field(default=None, description="Amount (SOL for buy, tokens for sell)")


class BotStatusResponse(BaseModel):
    """Bot status response"""
    status: str
    mode: Optional[str] = None
    uptime_seconds: Optional[float] = None
    positions_count: int = 0
    portfolio: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ===================================================================
#                            HELPER FUNCTIONS
# ===================================================================

async def broadcast_update(message: Dict):
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


def get_uptime() -> Optional[float]:
    """Calculate bot uptime in seconds"""
    if bot_state["start_time"]:
        return (datetime.now() - bot_state["start_time"]).total_seconds()
    return None


# ===================================================================
#                            API ENDPOINTS
# ===================================================================

@app.get("/")
async def root():
    """API health check"""
    return {
        "service": "Fenrir Trading Bot API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "bot_status": bot_state["status"],
        "uptime_seconds": get_uptime(),
        "timestamp": datetime.now().isoformat()
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
            config = BotConfig(
                mode=TradingMode(request.mode.value),
                buy_amount_sol=request.buy_amount_sol,
                stop_loss_pct=request.stop_loss_pct,
                take_profit_pct=request.take_profit_pct,
                trailing_stop_pct=request.trailing_stop_pct,
                max_position_age_minutes=request.max_position_age_minutes,
                min_initial_liquidity_sol=request.min_initial_liquidity_sol,
                max_initial_market_cap_sol=request.max_initial_market_cap_sol,
            )

            if request.rpc_url:
                config.rpc_url = request.rpc_url

            # Private key is loaded from environment by BotConfig.__post_init__

            errors = config.validate()
            if errors:
                raise HTTPException(status_code=400, detail=f"Invalid configuration: {', '.join(errors)}")

            bot_instance = FenrirBot(config)
            bot_task = asyncio.create_task(bot_instance.start())

            bot_state["status"] = "running"
            bot_state["start_time"] = datetime.now()
            bot_state["config"] = request.model_dump()
            bot_state["error"] = None

        except HTTPException:
            raise
        except Exception as e:
            bot_state["status"] = "error"
            bot_state["error"] = str(e)
            logger.error(f"Failed to start bot: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")

    await broadcast_update({
        "event": "bot_started",
        "mode": request.mode.value,
        "timestamp": datetime.now().isoformat()
    })

    logger.info(f"Fenrir bot started in {request.mode.value} mode")

    return {
        "status": "success",
        "message": f"Fenrir bot started in {request.mode.value} mode",
        "config": request.model_dump()
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
            raise HTTPException(status_code=500, detail=f"Error stopping bot: {str(e)}")

    await broadcast_update({
        "event": "bot_stopped",
        "timestamp": datetime.now().isoformat()
    })

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
            mode=bot_state["config"]["mode"] if bot_state["config"] else None,
            uptime_seconds=get_uptime(),
            positions_count=positions_count,
            portfolio=portfolio,
            error=bot_state["error"]
        )

    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@app.get("/bot/positions")
async def get_positions():
    """Get all open positions"""
    if not bot_instance or bot_state["status"] != "running":
        return {"positions": []}

    try:
        positions = []
        for token_address, position in bot_instance.positions.positions.items():
            positions.append({
                "token_address": token_address,
                "entry_time": position.entry_time.isoformat(),
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "amount_tokens": position.amount_tokens,
                "amount_sol_invested": position.amount_sol_invested,
                "pnl_percent": position.get_pnl_percent(),
                "pnl_sol": position.get_pnl_sol(),
                "peak_price": position.peak_price,
            })

        return {"positions": positions}

    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting positions: {str(e)}")


@app.post("/bot/trade")
async def execute_manual_trade(request: ManualTradeRequest):
    """Execute manual buy or sell trade"""
    if not bot_instance or bot_state["status"] != "running":
        raise HTTPException(status_code=400, detail="Bot is not running")

    try:
        if request.action.lower() == "buy":
            token_data = {
                "token_address": request.token_address,
                "initial_liquidity_sol": 10.0,
                "market_cap_sol": 50.0,
                "launch_time": datetime.now()
            }
            success = await bot_instance.trading_engine.execute_buy(token_data)

            if success:
                await broadcast_update({
                    "event": "manual_buy",
                    "token": request.token_address,
                    "timestamp": datetime.now().isoformat()
                })
                return {"status": "success", "message": f"Buy executed for {request.token_address}"}
            else:
                raise HTTPException(status_code=500, detail="Buy execution failed")

        elif request.action.lower() == "sell":
            success = await bot_instance.trading_engine.execute_sell(
                request.token_address,
                "Manual sell via API"
            )

            if success:
                await broadcast_update({
                    "event": "manual_sell",
                    "token": request.token_address,
                    "timestamp": datetime.now().isoformat()
                })
                return {"status": "success", "message": f"Sell executed for {request.token_address}"}
            else:
                raise HTTPException(status_code=500, detail="Sell execution failed")

        else:
            raise HTTPException(status_code=400, detail="Action must be 'buy' or 'sell'")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=f"Trade execution error: {str(e)}")


@app.get("/bot/config")
async def get_bot_config():
    """Get current bot configuration"""
    if not bot_state["config"]:
        raise HTTPException(status_code=400, detail="Bot has not been configured yet")

    return {"status": "success", "config": bot_state["config"]}


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time bot updates.

    Authentication: pass the API key via the Sec-WebSocket-Protocol header
    using the format 'authorization.<key>'. The server selects that subprotocol
    to complete the handshake. Alternatively, the X-API-Key header is accepted
    for clients that support custom headers on upgrade requests.
    """
    if FENRIR_API_KEY:
        # Check Sec-WebSocket-Protocol subprotocol (works in browsers)
        protocols = websocket.headers.get("sec-websocket-protocol", "")
        api_key = ""
        for proto in protocols.split(","):
            proto = proto.strip()
            if proto.startswith("authorization."):
                api_key = proto[len("authorization."):]
                break
        # Fallback: check X-API-Key header (works in non-browser clients)
        if not api_key:
            api_key = websocket.headers.get("x-api-key", "")
        if api_key != FENRIR_API_KEY:
            await websocket.close(code=4003, reason="Invalid API key")
            return

    await websocket.accept()
    active_websockets.append(websocket)

    try:
        await websocket.send_json({
            "event": "connected",
            "bot_status": bot_state["status"],
            "timestamp": datetime.now().isoformat()
        })

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
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
