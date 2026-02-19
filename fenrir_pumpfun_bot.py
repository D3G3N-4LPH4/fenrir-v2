#!/usr/bin/env python3
"""
FENRIR PUMP.FUN TRADING BOT - Backwards Compatibility Shim

This file re-exports all classes and functions from the fenrir/ package
for backwards compatibility. New code should import from fenrir directly:

    from fenrir import FenrirBot, BotConfig, TradingMode
    from fenrir.core.positions import Position, PositionManager

Or run via:
    python -m fenrir --mode simulation
"""

import asyncio

# Re-export everything from fenrir package
from fenrir.config import BotConfig, TradingMode
from fenrir.logger import FenrirLogger
from fenrir.core.wallet import WalletManager
from fenrir.core.client import SolanaClient
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.core.positions import Position, PositionManager
from fenrir.trading.engine import TradingEngine
from fenrir.trading.monitor import PumpFunMonitor
from fenrir.bot import FenrirBot, main

__all__ = [
    "TradingMode",
    "BotConfig",
    "FenrirLogger",
    "WalletManager",
    "SolanaClient",
    "JupiterSwapEngine",
    "Position",
    "PositionManager",
    "PumpFunMonitor",
    "TradingEngine",
    "FenrirBot",
    "main",
]


if __name__ == "__main__":
    asyncio.run(main())
