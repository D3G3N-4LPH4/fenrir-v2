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
from fenrir.config import BotConfig, TradingMode

# Exceptions
from fenrir.exceptions import (
    FenrirError,
    ConfigError,
    ExecutionError,
    InsufficientLiquidityError,
    BondingCurveMigratedError,
    SlippageExceededError,
    AIError,
    AITimeoutError,
    WalletError,
)

# Logger
from fenrir.logger import FenrirLogger

# Core components
from fenrir.core.wallet import WalletManager
from fenrir.core.client import SolanaClient
from fenrir.core.positions import Position, PositionManager
from fenrir.core.jupiter import JupiterSwapEngine

# Trading
from fenrir.trading.engine import TradingEngine
from fenrir.trading.monitor import PumpFunMonitor

# Protocol
from fenrir.protocol.pumpfun import PumpFunProgram, TokenLaunchDetector, BondingCurveState
from fenrir.protocol.jito import JitoMEVProtection

# AI
from fenrir.ai.brain import ClaudeBrain
from fenrir.ai.decision_engine import AITradingAnalyst, TokenAnalysis, TokenMetadata, AIDecision
from fenrir.ai.memory import AISessionMemory, DecisionRecord

# Data
from fenrir.data.price_feed import PriceFeedManager
from fenrir.data.database import TradeDatabase, PositionRecord
from fenrir.data.analytics import PerformanceAnalyzer

# Bot
from fenrir.bot import FenrirBot

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
