#!/usr/bin/env python3
"""
FENRIR - Core Configuration

Trading modes, bot configuration, and environment management.
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

# Dependency check (single location for the whole package)
try:
    from solana.rpc.async_api import AsyncClient  # noqa: F401
    from solders.keypair import Keypair  # noqa: F401
except ImportError:
    print("Missing dependencies. Install with:")
    print("   pip install solana solders base58 aiohttp python-dotenv websockets")
    sys.exit(1)


_dotenv_loaded = False


def _ensure_dotenv() -> None:
    """Load .env file exactly once, on first call."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv()
        _dotenv_loaded = True


class TradingMode(Enum):
    """Trading modes reflecting different risk appetites."""

    SIMULATION = "simulation"  # Paper trading - no real txs
    CONSERVATIVE = "conservative"  # Small positions, strict stops
    AGGRESSIVE = "aggressive"  # Larger positions, wider stops
    DEGEN = "degen"  # YOLO mode - maximum risk


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
    buy_amount_sol: float = 0.1  # SOL per trade
    max_slippage_bps: int = 500  # 5% max slippage

    # Risk Management - The guardrails that keep you alive
    stop_loss_pct: float = 25.0  # Exit if down 25%
    take_profit_pct: float = 100.0  # Exit if up 100%
    trailing_stop_pct: float = 15.0  # Trail by 15% from peak
    max_position_age_minutes: int = 60  # Auto-sell after 1 hour

    # Launch Criteria - What makes a token worth sniping?
    min_initial_liquidity_sol: float = 5.0  # Minimum SOL in bonding curve
    max_initial_market_cap_sol: float = 100.0  # Don't buy if already too big

    # Execution Settings
    priority_fee_lamports: int = 100000  # 0.0001 SOL priority (adjust for speed)
    use_jito: bool = False  # MEV protection via Jito bundles
    jito_tip_lamports: int = 10000  # Tip for Jito validators

    # Monitoring
    websocket_enabled: bool = True  # Real-time vs polling
    poll_interval_seconds: float = 2.0  # If WebSocket fails

    # AI Integration - Claude Brain (autonomous LLM decision engine)
    ai_analysis_enabled: bool = False  # Master switch for AI decisions
    ai_api_key: str = ""
    ai_model: str = "anthropic/claude-sonnet-4"
    ai_provider: str = "openrouter"  # "openrouter" or "anthropic_direct"
    ai_entry_timeout_seconds: float = 8.0  # Max wait for entry analysis
    ai_exit_timeout_seconds: float = 5.0  # Max wait for exit evaluation
    ai_exit_eval_interval_seconds: float = 60.0  # Proactive exit check cadence
    ai_min_confidence_to_buy: float = 0.6  # Minimum confidence for BUY
    ai_memory_size: int = 15  # Rolling decision history size
    ai_temperature: float = 0.3  # LLM temperature (lower = more conservative)
    ai_fallback_to_rules: bool = True  # Auto-buy on AI failure/timeout?
    ai_dynamic_position_sizing: bool = False  # Let AI set buy amount?

    # Logging
    log_level: str = "INFO"
    log_file: str = "fenrir_bot.log"

    def __post_init__(self):
        """Fill env-based defaults after dataclass init (avoids module-level side effects)."""
        _ensure_dotenv()
        if not self.rpc_url:
            self.rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        if not self.ws_url:
            self.ws_url = os.getenv("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com")
        if not self.private_key:
            self.private_key = os.getenv("WALLET_PRIVATE_KEY", "")
        if not self.ai_api_key:
            self.ai_api_key = os.getenv("OPENROUTER_API_KEY", "")

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
            if not self.rpc_url.startswith("http://127.0.0.1") and not self.rpc_url.startswith(
                "http://localhost"
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

        # AI configuration validation
        if self.ai_analysis_enabled and not self.ai_api_key:
            errors.append(
                "AI API key required when ai_analysis_enabled=True (set OPENROUTER_API_KEY)"
            )

        if not (0.0 <= self.ai_temperature <= 2.0):
            errors.append("AI temperature must be between 0.0 and 2.0")

        if not (0.0 <= self.ai_min_confidence_to_buy <= 1.0):
            errors.append("AI min confidence must be between 0.0 and 1.0")

        return errors
