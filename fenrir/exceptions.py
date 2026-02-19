#!/usr/bin/env python3
"""
FENRIR - Custom Exception Hierarchy

Structured error types for precise error handling.
"""


class FenrirError(Exception):
    """Base exception for all FENRIR errors."""

    pass


class ConfigError(FenrirError):
    """Invalid or missing configuration."""

    pass


class ExecutionError(FenrirError):
    """Trade execution failure."""

    pass


class InsufficientLiquidityError(ExecutionError):
    """Not enough liquidity for the requested trade."""

    pass


class BondingCurveMigratedError(ExecutionError):
    """Token has already migrated to Raydium; bonding curve is complete."""

    pass


class SlippageExceededError(ExecutionError):
    """Price impact exceeds maximum allowed slippage."""

    pass


class AIError(FenrirError):
    """AI decision engine failure."""

    pass


class AITimeoutError(AIError):
    """AI analysis timed out."""

    pass


class WalletError(FenrirError):
    """Wallet or key management error."""

    pass
