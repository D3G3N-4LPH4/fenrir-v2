#!/usr/bin/env python3
"""
FENRIR - Core Configuration

Trading modes, bot configuration, and environment management.
"""

import importlib  # FIX 1: was used below but never imported
import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

# Dependency check (single location for the whole package)
_deps_missing = False
try:
    # Import solana and solders modules dynamically to avoid static analysis
    # issues in environments where the packages aren't installed.
    _solana_rpc = importlib.import_module("solana.rpc.async_api")
    AsyncClient = _solana_rpc.AsyncClient  # type: ignore

    _solders_kp = importlib.import_module("solders.keypair")
    Keypair = _solders_kp.Keypair  # noqa: F401
except ImportError:
    print("Missing dependencies. Install with:")
    print("   pip install solana solders base58 aiohttp python-dotenv websockets")
    _deps_missing = True


_dotenv_loaded = False


def _ensure_dotenv() -> None:
    """Load .env file exactly once, on first call."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv()
        _dotenv_loaded = True


class TradingMode(Enum):
    """Trading modes reflecting different risk appetites."""

    SIMULATION = "simulation"      # Paper trading - no real txs
    CONSERVATIVE = "conservative"  # Small positions, strict stops
    AGGRESSIVE = "aggressive"      # Larger positions, wider stops
    DEGEN = "degen"                # YOLO mode - maximum risk


# ── Mode-specific trading presets ──────────────────────────────────────
TRADING_PRESETS: dict[TradingMode, dict] = {  # type: ignore[type-arg]
    TradingMode.SIMULATION: {
        "buy_amount_sol": 0.1,
        "max_slippage_bps": 500,
        "stop_loss_pct": 25.0,
        "take_profit_pct": 100.0,
        "trailing_stop_pct": 15.0,
        "max_position_age_minutes": 30,
        "min_initial_liquidity_sol": 0.0,
        "max_initial_market_cap_sol": 80.0,
        "priority_fee_lamports": 500_000,
        "ai_entry_timeout_seconds": 12.0,
        "ai_exit_timeout_seconds": 10.0,
        "ai_min_confidence_to_buy": 0.6,
        "ai_temperature": 0.3,
    },
    TradingMode.CONSERVATIVE: {
        "buy_amount_sol": 0.05,
        "max_slippage_bps": 300,
        "stop_loss_pct": 15.0,
        "take_profit_pct": 75.0,
        "trailing_stop_pct": 10.0,
        "max_position_age_minutes": 20,
        "min_initial_liquidity_sol": 5.0,
        "max_initial_market_cap_sol": 60.0,
        "priority_fee_lamports": 500_000,
        "ai_entry_timeout_seconds": 12.0,
        "ai_exit_timeout_seconds": 10.0,
        "ai_min_confidence_to_buy": 0.75,
        "ai_temperature": 0.2,
    },
    TradingMode.AGGRESSIVE: {
        "buy_amount_sol": 0.2,
        "max_slippage_bps": 800,
        "stop_loss_pct": 30.0,
        "take_profit_pct": 200.0,
        "trailing_stop_pct": 20.0,
        "max_position_age_minutes": 45,
        "min_initial_liquidity_sol": 2.0,
        "max_initial_market_cap_sol": 120.0,
        "priority_fee_lamports": 1_000_000,
        "ai_entry_timeout_seconds": 10.0,
        "ai_exit_timeout_seconds": 8.0,
        "ai_min_confidence_to_buy": 0.55,
        "ai_temperature": 0.4,
    },
    TradingMode.DEGEN: {
        "buy_amount_sol": 0.5,
        "max_slippage_bps": 1500,
        "stop_loss_pct": 50.0,
        "take_profit_pct": 500.0,
        "trailing_stop_pct": 30.0,
        "max_position_age_minutes": 60,
        "min_initial_liquidity_sol": 1.0,
        "max_initial_market_cap_sol": 200.0,
        "priority_fee_lamports": 2_000_000,
        "ai_entry_timeout_seconds": 10.0,
        "ai_exit_timeout_seconds": 8.0,
        "ai_min_confidence_to_buy": 0.4,
        "ai_temperature": 0.5,
    },
}


@dataclass
class BotConfig:
    """
    The brain of FENRIR. Every parameter tuned for elegance and control.
    Think of this as the DNA of your trading strategy.
    """

    # Network & Connection
    rpc_url: str = ""
    ws_url: str = ""

    # Wallet Configuration
    private_key: str = ""  # Base58 encoded

    # Pump.fun Program
    pumpfun_program: str = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

    # Trading Parameters
    mode: TradingMode = TradingMode.SIMULATION
    buy_amount_sol: float = 0.1       # SOL per trade
    max_slippage_bps: int = 500       # 5% max slippage

    # Risk Management
    stop_loss_pct: float = 25.0           # Exit if down 25%
    take_profit_pct: float = 100.0        # Exit if up 100%
    trailing_stop_pct: float = 15.0       # Trail by 15% from peak
    max_position_age_minutes: int = 30    # Memecoins pump/dump in minutes

    # Launch Criteria
    min_initial_liquidity_sol: float = 0.0      # Minimum SOL in bonding curve (0 = snipe at creation)
    max_initial_market_cap_sol: float = 80.0    # Don't buy if already too big

    # Execution Settings
    priority_fee_lamports: int = 500_000  # 0.0005 SOL for competitive inclusion
    use_jito: bool = False                # MEV protection via Jito bundles
    jito_tip_lamports: int = 10000        # Tip for Jito validators

    # Monitoring
    websocket_enabled: bool = True        # Real-time vs polling
    poll_interval_seconds: float = 2.0   # If WebSocket fails

    # Strategy budgets
    sniper_daily_budget_sol: float = 0.0  # 0 = auto (10 × buy_amount_sol)

    # AI Integration - Claude Brain
    ai_analysis_enabled: bool = True      # Master switch for AI decisions
    ai_api_key: str = ""
    ai_model: str = "anthropic/claude-haiku-4-5"
    ai_provider: str = "openrouter"       # "openrouter" or "anthropic_direct"
    ai_entry_timeout_seconds: float = 12.0
    ai_exit_timeout_seconds: float = 10.0
    ai_exit_eval_interval_seconds: float = 60.0
    ai_min_confidence_to_buy: float = 0.6
    ai_memory_size: int = 15
    ai_temperature: float = 0.3
    ai_fallback_to_rules: bool = True
    ai_dynamic_position_sizing: bool = False

    # Local Model Backend
    ai_local_model_enabled: bool = False
    ai_local_model_url: str = "http://localhost:8000/v1/chat/completions"
    ai_local_model_name: str = "fenrir-brain"

    # Logging — default is INFO; override via LOG_LEVEL env var
    # NOTE: do NOT use os.getenv() here — .env isn't loaded yet at class
    # definition time. The __post_init__ method handles the env override
    # after _ensure_dotenv() has run.
    log_level: str = "INFO"
    log_file: str = "fenrir_bot.log"

    def __post_init__(self) -> None:
        """Fill env-based defaults after dataclass init."""
        _ensure_dotenv()

        if not self.rpc_url:
            self.rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        if not self.ws_url:
            self.ws_url = os.getenv("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com")
        if not self.private_key:
            self.private_key = os.getenv("WALLET_PRIVATE_KEY", "")

        # FIX 2: ai_analysis_enabled controllable from .env
        env_ai_enabled = os.getenv("AI_ANALYSIS_ENABLED", "")
        if env_ai_enabled != "":
            self.ai_analysis_enabled = env_ai_enabled.lower() == "true"

        # FIX 3: support both OpenRouter and direct Anthropic API keys
        if not self.ai_api_key:
            self.ai_api_key = (
                os.getenv("OPENROUTER_API_KEY", "")
                or os.getenv("ANTHROPIC_API_KEY", "")
            )

        if not self.ai_local_model_url:
            self.ai_local_model_url = os.getenv(
                "AI_LOCAL_MODEL_URL", "http://localhost:8000/v1/chat/completions"
            )
        if not self.ai_local_model_name:
            self.ai_local_model_name = os.getenv("AI_LOCAL_MODEL_NAME", "fenrir-brain")
        if not self.ai_local_model_enabled:
            self.ai_local_model_enabled = (
                os.getenv("AI_LOCAL_MODEL_ENABLED", "").lower() == "true"
            )

        # FIX 4: LOG_LEVEL controllable from .env
        # Always defer to env var — .env is now loaded so os.getenv is reliable
        env_log_level = os.getenv("LOG_LEVEL", "").upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if env_log_level in valid_levels:
            self.log_level = env_log_level

    @classmethod
    def from_mode(cls, mode: TradingMode, **overrides) -> "BotConfig":  # type: ignore[override]
        """Create a BotConfig pre-tuned for a specific trading mode."""
        preset = TRADING_PRESETS.get(mode, {})  # type: ignore[arg-type]
        merged = {**preset, "mode": mode, **overrides}
        return cls(**merged)

    def __repr__(self) -> str:
        """Redact sensitive fields to prevent accidental secret leakage in logs."""
        pk_display = "***" if self.private_key else "(empty)"
        api_key_display = "***" if self.ai_api_key else "(empty)"
        return (
            f"BotConfig(mode={self.mode.value}, "
            f"rpc_url='{self.rpc_url[:30]}...', "
            f"private_key='{pk_display}', "
            f"ai_api_key='{api_key_display}', "
            f"buy_amount_sol={self.buy_amount_sol})"
        )

    def validate(self) -> list[str]:
        """Validation that feels like a caring mentor checking your homework."""
        errors = []

        if not self.rpc_url:
            errors.append("RPC URL required - get one from QuickNode or Helius")

        if self.rpc_url and not self.rpc_url.startswith("https://"):
            if (
                not self.rpc_url.startswith("http://127.0.0.1")
                and not self.rpc_url.startswith("http://localhost")
            ):
                errors.append("RPC URL must use HTTPS (plaintext HTTP leaks wallet data)")

        if self.mode != TradingMode.SIMULATION and not self.private_key:
            errors.append("Private key required for live trading (set WALLET_PRIVATE_KEY)")

        if self.buy_amount_sol <= 0:
            errors.append("Buy amount must be positive")

        if self.stop_loss_pct >= 100:
            errors.append("Stop loss too aggressive - you'd lose everything")

        if self.take_profit_pct <= 0:
            errors.append("Take profit must be positive - you want gains, right?")

        if self.priority_fee_lamports < 0:
            errors.append("Priority fee must be non-negative")

        if self.use_jito and self.jito_tip_lamports <= 0:
            errors.append("Jito tip must be positive when use_jito=True")

        if self.ai_analysis_enabled and not self.ai_api_key:
            errors.append(
                "AI API key required when ai_analysis_enabled=True "
                "(set OPENROUTER_API_KEY or ANTHROPIC_API_KEY)"
            )

        if not (0.0 <= self.ai_temperature <= 2.0):
            errors.append("AI temperature must be between 0.0 and 2.0")

        if not (0.0 <= self.ai_min_confidence_to_buy <= 1.0):
            errors.append("AI min confidence must be between 0.0 and 1.0")

        if self.ai_local_model_enabled and self.ai_analysis_enabled:
            if not self.ai_local_model_url.startswith("http"):
                errors.append("ai_local_model_url must be a valid HTTP URL")
            if not self.ai_local_model_name:
                errors.append(
                    "ai_local_model_name must be set when ai_local_model_enabled=True"
                )

        return errors