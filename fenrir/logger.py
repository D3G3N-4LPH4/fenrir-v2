#!/usr/bin/env python3
"""
FENRIR - Elegant Logging System

Logs that read like a story, not error dumps.
"""

import logging
from logging.handlers import RotatingFileHandler

from .config import BotConfig


class FenrirLogger:
    """
    Logs that read like a story, not error dumps.
    Every message should inform and inspire.
    """

    def __init__(self, config: BotConfig):
        self.logger = logging.getLogger("FENRIR")
        self.logger.setLevel(getattr(logging, config.log_level))

        # Guard against duplicate handlers when FenrirLogger is instantiated
        # multiple times (e.g. in tests). logging.getLogger returns the same
        # logger instance, but handlers stack if not checked.
        if not self.logger.handlers:
            # Console handler - beautiful terminal output
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(message)s',
                datefmt='%H:%M:%S'
            ))
            self.logger.addHandler(console)

            # File handler - persistent history
            file_handler = RotatingFileHandler(
                config.log_file,
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
            ))
            self.logger.addHandler(file_handler)

    def launch_detected(self, token_address: str, initial_liq: float):
        """A new hunt begins."""
        self.logger.info(f"NEW LAUNCH DETECTED: {token_address}")
        self.logger.info(f"   Initial Liquidity: {initial_liq:.2f} SOL")

    def buy_executed(self, token: str, amount_sol: float, price: float):
        """The snipe lands."""
        self.logger.info(f"BUY EXECUTED: {amount_sol:.4f} SOL -> {token[:8]}...")
        self.logger.info(f"   Entry Price: ${price:.8f}")

    def sell_executed(self, token: str, pnl_pct: float, reason: str):
        """Exit strategy activated."""
        emoji = "+" if pnl_pct > 0 else ""
        self.logger.info(f"SELL EXECUTED: {token[:8]}... | PnL: {emoji}{pnl_pct:.2f}%")
        self.logger.info(f"   Reason: {reason}")

    def error(self, context: str, error: Exception):
        """Failures happen. Learn from them."""
        self.logger.error(f"{context}: {str(error)}")

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def debug(self, message: str):
        self.logger.debug(message)
