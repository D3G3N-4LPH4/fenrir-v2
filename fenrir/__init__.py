"""
FENRIR - Pump.fun Trading Bot

"The wolf that breaks chains and hunts opportunities in the memecoin
wilderness of Solana. Not just code - a philosophy of elegant automation."

Usage:
    from fenrir import FenrirBot, BotConfig, TradingMode

    config = BotConfig(mode=TradingMode.SIMULATION)
    bot = FenrirBot(config)
    await bot.start()
"""

__version__ = "1.0.0"

# Core configuration
# AI
from fenrir.ai.brain import ClaudeBrain
from fenrir.ai.decision_engine import AIDecision, AITradingAnalyst, TokenAnalysis, TokenMetadata
from fenrir.ai.memory import AISessionMemory, DecisionRecord

# Bot
from fenrir.bot import FenrirBot
from fenrir.config import BotConfig, TradingMode
from fenrir.core.client import SolanaClient
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.core.positions import Position, PositionManager

# Core components
from fenrir.core.wallet import WalletManager
from fenrir.data.analytics import PerformanceAnalyzer
from fenrir.data.database import PositionRecord, TradeDatabase

# Data
from fenrir.data.price_feed import PriceFeedManager

# Exceptions
from fenrir.exceptions import (
    AIError,
    AITimeoutError,
    BondingCurveMigratedError,
    ConfigError,
    ExecutionError,
    FenrirError,
    InsufficientLiquidityError,
    SlippageExceededError,
    WalletError,
)

# Logger
from fenrir.logger import FenrirLogger
from fenrir.protocol.jito import JitoMEVProtection

# Protocol
from fenrir.protocol.pumpfun import BondingCurveState, PumpFunProgram, TokenLaunchDetector

# Trading
from fenrir.trading.engine import TradingEngine
from fenrir.trading.monitor import PumpFunMonitor

__all__ = [
    # Config
    "BotConfig",
    "TradingMode",
    # Exceptions
    "FenrirError",
    "ConfigError",
    "ExecutionError",
    "InsufficientLiquidityError",
    "BondingCurveMigratedError",
    "SlippageExceededError",
    "AIError",
    "AITimeoutError",
    "WalletError",
    # Logger
    "FenrirLogger",
    # Core
    "WalletManager",
    "SolanaClient",
    "Position",
    "PositionManager",
    "JupiterSwapEngine",
    # Trading
    "TradingEngine",
    "PumpFunMonitor",
    # Protocol
    "PumpFunProgram",
    "TokenLaunchDetector",
    "BondingCurveState",
    "JitoMEVProtection",
    # AI
    "ClaudeBrain",
    "AITradingAnalyst",
    "TokenAnalysis",
    "TokenMetadata",
    "AIDecision",
    "AISessionMemory",
    "DecisionRecord",
    # Data
    "PriceFeedManager",
    "TradeDatabase",
    "PositionRecord",
    "PerformanceAnalyzer",
    # Bot
    "FenrirBot",
]
