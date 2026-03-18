"""
protocols.py — FENRIR v2 Duality Enforcement Layer
====================================================
Abstract interfaces that form the contract between strategy logic
and the execution environment (backtest vs. live).

Rules enforced by this file:
  1. Strategy code ONLY imports from this file — never from live or backtest impls
  2. Every method has a typed signature — no dict/Any escapes
  3. All methods are async — backtest impls use asyncio.sleep() to model latency

This is the boundary. If strategy code reaches past it, the duality breaks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

import polars as pl


# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class DecisionAction(str, Enum):
    BUY    = "BUY"
    SELL   = "SELL"
    HOLD   = "HOLD"
    ADD    = "ADD"   # scale into existing position
    EXIT   = "EXIT"  # emergency exit — skip Claude, route direct


@dataclass(frozen=True)
class TokenInfo:
    mint:        str
    ticker:      str
    name:        str
    creator:     str
    launch_ts:   datetime
    initial_sol: float = 0.0


@dataclass(frozen=True)
class Position:
    mint:          str
    tokens:        float   # token units held
    entry_price:   float   # SOL per token at entry
    entry_ts:      datetime
    sol_invested:  float
    current_price: float = 0.0

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def current_value_sol(self) -> float:
        return self.tokens * self.current_price


@dataclass
class OrderResult:
    success:        bool
    side:           OrderSide
    mint:           str
    sol_amount:     float
    tokens:         float
    effective_price: float
    slippage_pct:   float
    latency_ms:     float
    tx_signature:   str | None = None
    error:          str | None = None
    timestamp:      datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def failed(cls, mint: str, side: OrderSide, reason: str) -> "OrderResult":
        return cls(
            success=False, side=side, mint=mint,
            sol_amount=0, tokens=0, effective_price=0,
            slippage_pct=0, latency_ms=0, error=reason,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"]      = self.side.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class ClaudeDecision:
    action:        DecisionAction
    confidence:    float          # 0.0 – 1.0
    reasoning:     str
    sol_amount:    float = 0.0    # populated for BUY / ADD
    sell_pct:      float = 1.0    # populated for SELL / EXIT (1.0 = full)
    raw_prompt:    str = ""       # the exact prompt sent — stored for ART
    raw_response:  str = ""       # the exact response — stored for ART
    model:         str = ""
    tokens_used:   int = 0
    latency_ms:    float = 0.0
    timestamp:     datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"]    = self.action.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class OnChainSnapshot:
    mint:              str
    price_sol:         float
    market_cap_sol:    float
    liquidity_sol:     float
    holder_count:      int
    top10_pct:         float    # % supply in top 10 wallets
    creator_holdings:  float    # % supply held by creator
    bonding_progress:  float    # 0.0 – 1.0 toward graduation
    volume_5m:         float
    volume_1h:         float
    buy_sell_ratio:    float    # buys / (buys + sells) in last 5m
    timestamp:         datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class SocialSnapshot:
    mint:             str
    ticker:           str
    tweet_count:      int
    sentiment_score:  float     # -1.0 to +1.0
    bull_count:       int
    bear_count:       int
    bear_warnings:    list[str] = field(default_factory=list)
    bull_signals:     list[str] = field(default_factory=list)
    kol_mentions:     list[str] = field(default_factory=list)
    scanned_at:       datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# The 4 core protocols — strategy only touches these
# ---------------------------------------------------------------------------

@runtime_checkable
class DataFeed(Protocol):
    """Provides OHLCV candle data — historical replay or live stream."""

    async def get_ohlcv(
        self,
        mint: str,
        lookback: int = 100,
    ) -> pl.DataFrame:
        """
        Returns a Polars DataFrame with columns:
        [timestamp, open, high, low, close, volume]
        Sorted ascending by timestamp, exactly `lookback` rows (or fewer
        if not enough history exists).
        """
        ...

    async def get_current_price(self, mint: str) -> float:
        """Latest close price in SOL."""
        ...

    async def get_token_info(self, mint: str) -> TokenInfo | None:
        """Static metadata about the token."""
        ...

    async def subscribe_candles(
        self,
        mint: str,
        callback,   # async callable(candle: dict) -> None
    ) -> None:
        """
        Subscribe to new candle closes.
        In backtest: drives the replay loop.
        In live: connects to WebSocket feed.
        """
        ...


@runtime_checkable
class OrderRouter(Protocol):
    """Executes orders — simulated or live broker."""

    async def buy(
        self,
        mint: str,
        sol_amount: float,
        max_slippage_pct: float = 0.05,
    ) -> OrderResult:
        ...

    async def sell(
        self,
        mint: str,
        sell_pct: float = 1.0,    # 1.0 = full position
        max_slippage_pct: float = 0.05,
    ) -> OrderResult:
        ...

    async def get_position(self, mint: str) -> Position | None:
        ...

    async def get_balance_sol(self) -> float:
        ...

    async def get_all_positions(self) -> dict[str, Position]:
        ...


@runtime_checkable
class ContextProvider(Protocol):
    """Provides enriched context beyond raw price data."""

    async def get_on_chain(self, mint: str) -> OnChainSnapshot | None:
        ...

    async def get_social(self, mint: str) -> SocialSnapshot | None:
        ...

    async def get_smc_context(self, mint: str) -> str:
        """Pre-formatted SMC signal string for Claude prompt."""
        ...


@runtime_checkable
class ExecutionRecorder(Protocol):
    """
    Records every decision event for replay and ART training.
    In live: writes to append-only JSONL file.
    In backtest: writes to in-memory buffer (optionally to file).
    Strategy never checks mode — it just calls record().
    """

    async def record(
        self,
        mint:      str,
        decision:  ClaudeDecision,
        order:     OrderResult | None,
        context:   dict,          # on_chain + social + smc at decision time
        outcome:   dict | None,   # filled in retrospectively during backtest
    ) -> None:
        ...

    async def flush(self) -> None:
        """Force write any buffered events."""
        ...
