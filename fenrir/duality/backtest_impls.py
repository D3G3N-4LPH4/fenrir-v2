"""
backtest_impls.py — FENRIR v2 Backtest Concrete Implementations
================================================================
Implements DataFeed, OrderRouter, ContextProvider, ExecutionRecorder
for backtest mode. Strategy code never imports this directly — it
receives these via dependency injection at the entrypoint.

Key design choices:
  - SlippageModel is pluggable — swap in a real distribution fit to
    your live pump.fun execution data once you have enough history
  - HistoricalDataFeed drives the replay loop — the strategy's
    subscribe_candles callback IS the backtest event loop
  - SimulatedRouter logs every trade with full detail for ART export
  - SnapshotProvider replays on-chain + social snapshots from disk
    (or generates synthetic ones if no snapshots exist yet)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, AsyncIterator

import polars as pl

from .protocols import (
    DataFeed, OrderRouter, ContextProvider, ExecutionRecorder,
    TokenInfo, Position, OrderResult, OrderSide,
    OnChainSnapshot, SocialSnapshot, ClaudeDecision,
)

logger = logging.getLogger("fenrir.backtest")


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------

class SlippageModel:
    """
    Base class. Override apply_buy / apply_sell with your distribution.
    Default: log-normal slippage calibrated for thin pump.fun liquidity.
    """

    def __init__(
        self,
        base_slippage: float = 0.02,   # 2% base
        vol_multiplier: float = 0.5,   # scales with volume ratio
        seed: int | None = None,
    ):
        self.base = base_slippage
        self.vol_mult = vol_multiplier
        self._rng = random.Random(seed)

    def apply_buy(
        self,
        price: float,
        sol_amount: float,
        liquidity_sol: float,
    ) -> tuple[float, float]:
        """Returns (effective_price, slippage_pct)."""
        # Market impact grows with order size relative to liquidity
        impact = (sol_amount / max(liquidity_sol, 1.0)) * self.vol_mult
        slippage = self.base + impact + self._rng.gauss(0, 0.005)
        slippage = max(0.001, min(slippage, 0.30))   # cap at 30%
        effective = price * (1 + slippage)
        return effective, slippage

    def apply_sell(
        self,
        price: float,
        tokens: float,
        liquidity_sol: float,
    ) -> tuple[float, float]:
        """Returns (effective_price, slippage_pct). Sells push price down."""
        sol_value = tokens * price
        impact = (sol_value / max(liquidity_sol, 1.0)) * self.vol_mult
        slippage = self.base + impact + self._rng.gauss(0, 0.005)
        slippage = max(0.001, min(slippage, 0.30))
        effective = price * (1 - slippage)
        return effective, slippage


class ZeroSlippageModel(SlippageModel):
    """Use for debugging only — produces optimistic unrealistic results."""

    def apply_buy(self, price, sol_amount, liquidity_sol):
        return price, 0.0

    def apply_sell(self, price, tokens, liquidity_sol):
        return price, 0.0


# ---------------------------------------------------------------------------
# Simulated Order Router
# ---------------------------------------------------------------------------

class SimulatedRouter:
    """
    Backtest implementation of OrderRouter.
    Models: balance, positions, slippage, execution latency (400ms Solana).
    Records every trade for ART export.
    """

    SOLANA_BLOCK_MS = 400   # realistic Solana block time

    def __init__(
        self,
        initial_sol: float = 10.0,
        slippage: SlippageModel | None = None,
        simulate_latency: bool = True,
    ):
        self.balance        = initial_sol
        self.initial_sol    = initial_sol
        self.slippage       = slippage or SlippageModel(seed=42)
        self.simulate_latency = simulate_latency

        self._positions: dict[str, Position]  = {}
        self.trade_log:  list[OrderResult]    = []

        # Live price feed injected by HistoricalDataFeed during replay
        self._price_fn:     Callable[[str], float] | None = None
        self._liquidity_fn: Callable[[str], float] | None = None

    def _set_price_fn(self, fn: Callable[[str], float]) -> None:
        self._price_fn = fn

    def _set_liquidity_fn(self, fn: Callable[[str], float]) -> None:
        self._liquidity_fn = fn

    def _current_price(self, mint: str) -> float:
        if self._price_fn:
            return self._price_fn(mint)
        raise RuntimeError("Price function not injected into SimulatedRouter")

    def _current_liquidity(self, mint: str) -> float:
        if self._liquidity_fn:
            return self._liquidity_fn(mint)
        return 100.0   # default fallback

    async def buy(
        self,
        mint: str,
        sol_amount: float,
        max_slippage_pct: float = 0.05,
    ) -> OrderResult:
        if self.simulate_latency:
            await asyncio.sleep(self.SOLANA_BLOCK_MS / 1000)

        if sol_amount > self.balance:
            return OrderResult.failed(mint, OrderSide.BUY, "Insufficient SOL balance")
        if sol_amount <= 0:
            return OrderResult.failed(mint, OrderSide.BUY, "Invalid sol_amount <= 0")

        price     = self._current_price(mint)
        liquidity = self._current_liquidity(mint)
        t0        = datetime.now(timezone.utc)

        effective_price, slippage = self.slippage.apply_buy(
            price, sol_amount, liquidity
        )

        if slippage > max_slippage_pct:
            return OrderResult.failed(
                mint, OrderSide.BUY,
                f"Slippage {slippage:.2%} exceeds max {max_slippage_pct:.2%}"
            )

        tokens_received = sol_amount / effective_price
        self.balance   -= sol_amount

        # Merge with existing position if any
        existing = self._positions.get(mint)
        if existing:
            total_tokens = existing.tokens + tokens_received
            total_sol    = existing.sol_invested + sol_amount
            avg_entry    = total_sol / total_tokens
            self._positions[mint] = Position(
                mint=mint, tokens=total_tokens, entry_price=avg_entry,
                entry_ts=existing.entry_ts, sol_invested=total_sol,
                current_price=effective_price,
            )
        else:
            self._positions[mint] = Position(
                mint=mint, tokens=tokens_received, entry_price=effective_price,
                entry_ts=t0, sol_invested=sol_amount, current_price=effective_price,
            )

        result = OrderResult(
            success=True, side=OrderSide.BUY, mint=mint,
            sol_amount=sol_amount, tokens=tokens_received,
            effective_price=effective_price, slippage_pct=slippage,
            latency_ms=self.SOLANA_BLOCK_MS if self.simulate_latency else 0,
            tx_signature=f"SIM_BUY_{mint[:8]}_{int(t0.timestamp())}",
            timestamp=t0,
        )
        self.trade_log.append(result)
        logger.debug(f"SIM BUY {mint[:8]} {sol_amount:.3f}SOL @ {effective_price:.6f} slip={slippage:.2%}")
        return result

    async def sell(
        self,
        mint: str,
        sell_pct: float = 1.0,
        max_slippage_pct: float = 0.05,
    ) -> OrderResult:
        if self.simulate_latency:
            await asyncio.sleep(self.SOLANA_BLOCK_MS / 1000)

        pos = self._positions.get(mint)
        if not pos or pos.tokens <= 0:
            return OrderResult.failed(mint, OrderSide.SELL, "No position to sell")

        price     = self._current_price(mint)
        liquidity = self._current_liquidity(mint)
        t0        = datetime.now(timezone.utc)

        tokens_to_sell = pos.tokens * sell_pct
        effective_price, slippage = self.slippage.apply_sell(
            price, tokens_to_sell, liquidity
        )

        if slippage > max_slippage_pct:
            # On EXIT decisions we force through regardless
            logger.warning(f"Sell slippage {slippage:.2%} exceeds max — executing anyway")

        sol_received = tokens_to_sell * effective_price
        self.balance += sol_received

        if sell_pct >= 1.0:
            del self._positions[mint]
        else:
            remaining = pos.tokens * (1 - sell_pct)
            self._positions[mint] = Position(
                mint=mint, tokens=remaining, entry_price=pos.entry_price,
                entry_ts=pos.entry_ts,
                sol_invested=pos.sol_invested * (1 - sell_pct),
                current_price=effective_price,
            )

        result = OrderResult(
            success=True, side=OrderSide.SELL, mint=mint,
            sol_amount=sol_received, tokens=tokens_to_sell,
            effective_price=effective_price, slippage_pct=slippage,
            latency_ms=self.SOLANA_BLOCK_MS if self.simulate_latency else 0,
            tx_signature=f"SIM_SELL_{mint[:8]}_{int(t0.timestamp())}",
            timestamp=t0,
        )
        self.trade_log.append(result)
        pnl = ((effective_price / pos.entry_price) - 1) * 100
        logger.debug(f"SIM SELL {mint[:8]} {sell_pct:.0%} @ {effective_price:.6f} PnL={pnl:+.1f}%")
        return result

    async def get_position(self, mint: str) -> Position | None:
        pos = self._positions.get(mint)
        if pos:
            price = self._current_price(mint)
            # Return with updated current_price
            return Position(
                mint=pos.mint, tokens=pos.tokens, entry_price=pos.entry_price,
                entry_ts=pos.entry_ts, sol_invested=pos.sol_invested,
                current_price=price,
            )
        return None

    async def get_balance_sol(self) -> float:
        return self.balance

    async def get_all_positions(self) -> dict[str, Position]:
        return {
            mint: Position(
                mint=p.mint, tokens=p.tokens, entry_price=p.entry_price,
                entry_ts=p.entry_ts, sol_invested=p.sol_invested,
                current_price=self._current_price(mint),
            )
            for mint, p in self._positions.items()
        }

    def summary(self) -> dict:
        """Call at end of backtest for performance metrics."""
        total_trades  = len(self.trade_log)
        buys          = [t for t in self.trade_log if t.side == OrderSide.BUY]
        sells         = [t for t in self.trade_log if t.side == OrderSide.SELL]
        total_sol_in  = sum(t.sol_amount for t in buys)
        total_sol_out = sum(t.sol_amount for t in sells)
        pnl_sol       = (self.balance - self.initial_sol)
        pnl_pct       = pnl_sol / self.initial_sol * 100

        return {
            "initial_sol":    self.initial_sol,
            "final_sol":      self.balance,
            "pnl_sol":        round(pnl_sol, 4),
            "pnl_pct":        round(pnl_pct, 2),
            "total_trades":   total_trades,
            "buys":           len(buys),
            "sells":          len(sells),
            "total_sol_in":   round(total_sol_in, 4),
            "total_sol_out":  round(total_sol_out, 4),
            "avg_slippage":   round(
                sum(t.slippage_pct for t in self.trade_log) / max(total_trades, 1), 4
            ),
        }


# ---------------------------------------------------------------------------
# Historical Data Feed
# ---------------------------------------------------------------------------

class HistoricalDataFeed:
    """
    Backtest implementation of DataFeed.
    Reads Polars Parquet files and drives the replay event loop.

    Expected file layout:
        data/
          candles/
            {mint}.parquet   ← columns: timestamp, open, high, low, close, volume
          snapshots/
            {mint}_onchain.parquet   ← OnChainSnapshot rows keyed by timestamp
            {mint}_social.parquet    ← SocialSnapshot rows keyed by timestamp
          tokens.parquet             ← TokenInfo rows

    If snapshot files don't exist, synthetic snapshots are generated.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir     = Path(data_dir)
        self._candles:    dict[str, pl.DataFrame] = {}
        self._onchain:    dict[str, pl.DataFrame] = {}
        self._social:     dict[str, pl.DataFrame] = {}
        self._tokens:     dict[str, TokenInfo]    = {}
        self._current_ts: datetime | None         = None
        # Current candle index per mint — used by router price_fn
        self._price_now:  dict[str, float]        = {}
        self._liq_now:    dict[str, float]        = {}

    def load(self, mints: list[str]) -> None:
        """Pre-load all data for the specified mints."""
        for mint in mints:
            candle_path = self.data_dir / "candles" / f"{mint}.parquet"
            if not candle_path.exists():
                logger.warning(f"No candle data for {mint}, skipping")
                continue

            df = pl.read_parquet(candle_path).sort("timestamp")
            self._candles[mint] = df
            logger.info(f"Loaded {len(df)} candles for {mint[:8]}")

            # Load on-chain snapshots (optional)
            oc_path = self.data_dir / "snapshots" / f"{mint}_onchain.parquet"
            if oc_path.exists():
                self._onchain[mint] = pl.read_parquet(oc_path).sort("timestamp")
            else:
                self._onchain[mint] = self._synthetic_onchain(df)

            # Load social snapshots (optional)
            soc_path = self.data_dir / "snapshots" / f"{mint}_social.parquet"
            if soc_path.exists():
                self._social[mint] = pl.read_parquet(soc_path).sort("timestamp")
            else:
                self._social[mint] = self._synthetic_social()

    async def get_ohlcv(self, mint: str, lookback: int = 100) -> pl.DataFrame:
        df = self._candles.get(mint)
        if df is None:
            return pl.DataFrame(schema={
                "timestamp": pl.Datetime, "open": pl.Float64,
                "high": pl.Float64, "low": pl.Float64,
                "close": pl.Float64, "volume": pl.Float64,
            })

        if self._current_ts:
            df = df.filter(pl.col("timestamp") <= self._current_ts)
        return df.tail(lookback)

    async def get_current_price(self, mint: str) -> float:
        return self._price_now.get(mint, 0.0)

    async def get_token_info(self, mint: str) -> TokenInfo | None:
        return self._tokens.get(mint)

    async def subscribe_candles(self, mint: str, callback) -> None:
        """
        Drives the backtest replay loop for a single token.
        Calls callback(candle) for each candle in chronological order,
        updating internal price state so the router sees the right prices.
        """
        df = self._candles.get(mint)
        if df is None:
            logger.warning(f"subscribe_candles: no data for {mint[:8]}")
            return

        logger.info(f"Replaying {len(df)} candles for {mint[:8]}")
        for row in df.iter_rows(named=True):
            self._current_ts         = row["timestamp"]
            self._price_now[mint]    = row["close"]

            # Update liquidity estimate from on-chain snapshot
            oc = self._get_onchain_at(mint, self._current_ts)
            self._liq_now[mint] = oc.liquidity_sol if oc else 50.0

            await callback(row)

    def get_price_fn(self, mint: str) -> Callable[[str], float]:
        """Returns a closure for the SimulatedRouter to call."""
        return lambda _: self._price_now.get(mint, 0.0)

    def get_liquidity_fn(self) -> Callable[[str], float]:
        return lambda mint: self._liq_now.get(mint, 50.0)

    def _get_onchain_at(
        self, mint: str, ts: datetime
    ) -> OnChainSnapshot | None:
        df = self._onchain.get(mint)
        if df is None or df.is_empty():
            return None
        # Find most recent snapshot at or before ts
        filtered = df.filter(pl.col("timestamp") <= ts)
        if filtered.is_empty():
            filtered = df.head(1)
        row = filtered.tail(1).to_dicts()[0]
        return OnChainSnapshot(
            mint=mint,
            price_sol=row.get("price_sol", self._price_now.get(mint, 0)),
            market_cap_sol=row.get("market_cap_sol", 0),
            liquidity_sol=row.get("liquidity_sol", 50),
            holder_count=row.get("holder_count", 0),
            top10_pct=row.get("top10_pct", 0),
            creator_holdings=row.get("creator_holdings", 0),
            bonding_progress=row.get("bonding_progress", 0),
            volume_5m=row.get("volume_5m", 0),
            volume_1h=row.get("volume_1h", 0),
            buy_sell_ratio=row.get("buy_sell_ratio", 0.5),
            timestamp=row.get("timestamp", ts),
        )

    @staticmethod
    def _synthetic_onchain(candle_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate plausible on-chain snapshots from candle data.
        Used when real snapshot data isn't available yet.
        Lets you backtest strategy logic before you have full data capture.
        """
        rows = []
        for row in candle_df.iter_rows(named=True):
            price  = row["close"]
            vol    = row["volume"]
            mcap   = price * 1_000_000_000   # assume 1B supply
            liq    = max(vol * 0.1, 5.0)
            rows.append({
                "timestamp":       row["timestamp"],
                "price_sol":       price,
                "market_cap_sol":  mcap,
                "liquidity_sol":   liq,
                "holder_count":    random.randint(50, 500),
                "top10_pct":       random.uniform(0.3, 0.7),
                "creator_holdings": random.uniform(0.0, 0.15),
                "bonding_progress": min(mcap / 1000, 1.0),
                "volume_5m":       vol,
                "volume_1h":       vol * 12,
                "buy_sell_ratio":  random.uniform(0.3, 0.7),
            })
        return pl.DataFrame(rows)

    @staticmethod
    def _synthetic_social() -> pl.DataFrame:
        """Empty social snapshot frame — safe default when no data exists."""
        return pl.DataFrame(schema={
            "timestamp":      pl.Datetime,
            "sentiment_score": pl.Float64,
            "tweet_count":    pl.Int64,
            "bull_count":     pl.Int64,
            "bear_count":     pl.Int64,
        })


# ---------------------------------------------------------------------------
# Backtest Context Provider
# ---------------------------------------------------------------------------

class BacktestContextProvider:
    """
    Backtest implementation of ContextProvider.
    Sources on-chain and social data from the HistoricalDataFeed's
    snapshot tables, so Claude sees the same data it would have seen live.
    """

    def __init__(self, feed: HistoricalDataFeed):
        self._feed = feed

    async def get_on_chain(self, mint: str) -> OnChainSnapshot | None:
        return self._feed._get_onchain_at(mint, self._feed._current_ts)

    async def get_social(self, mint: str) -> SocialSnapshot | None:
        # Return a neutral snapshot if no social data
        df = self._feed._social.get(mint, pl.DataFrame())
        if df.is_empty() or self._feed._current_ts is None:
            return SocialSnapshot(mint=mint, ticker="?", tweet_count=0,
                                  sentiment_score=0.0, bull_count=0, bear_count=0)
        filtered = df.filter(pl.col("timestamp") <= self._feed._current_ts)
        if filtered.is_empty():
            return None
        row = filtered.tail(1).to_dicts()[0]
        return SocialSnapshot(
            mint=mint, ticker=row.get("ticker", "?"),
            tweet_count=row.get("tweet_count", 0),
            sentiment_score=row.get("sentiment_score", 0.0),
            bull_count=row.get("bull_count", 0),
            bear_count=row.get("bear_count", 0),
        )

    async def get_smc_context(self, mint: str) -> str:
        ohlcv = await self._feed.get_ohlcv(mint, lookback=50)
        if ohlcv.is_empty():
            return "[SMC] Insufficient data"
        # Lightweight SMC summary without full adapter overhead
        close = ohlcv["close"].to_list()
        high  = ohlcv["high"].to_list()
        low   = ohlcv["low"].to_list()
        if len(close) < 10:
            return "[SMC] Insufficient candles"
        recent_high  = max(high[-10:])
        recent_low   = min(low[-10:])
        current      = close[-1]
        pos_in_range = (current - recent_low) / max(recent_high - recent_low, 1e-10)
        trend        = "BULLISH" if close[-1] > close[-5] else "BEARISH"
        return (
            f"[SMC BACKTEST] Trend={trend} | "
            f"Range=[{recent_low:.6f}–{recent_high:.6f}] | "
            f"Position={pos_in_range:.0%} of range"
        )


# ---------------------------------------------------------------------------
# Backtest Execution Recorder
# ---------------------------------------------------------------------------

class BacktestRecorder:
    """
    Backtest implementation of ExecutionRecorder.
    Buffers all events in memory for ART export at end of run.
    Optionally also writes to JSONL for large backtests.
    """

    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path
        self._events:    list[dict] = []
        self._file = None

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(output_path, "w")

    async def record(
        self,
        mint:     str,
        decision: ClaudeDecision,
        order:    OrderResult | None,
        context:  dict,
        outcome:  dict | None = None,
    ) -> None:
        event = {
            "mint":     mint,
            "decision": decision.to_dict(),
            "order":    order.to_dict() if order else None,
            "context":  context,
            "outcome":  outcome,
        }
        self._events.append(event)
        if self._file:
            self._file.write(json.dumps(event) + "\n")

    async def flush(self) -> None:
        if self._file:
            self._file.flush()

    def backfill_outcomes(self, router: SimulatedRouter) -> None:
        """
        After replay completes, walk the trade log and attach outcomes
        to each decision event. Outcome = what actually happened to price
        in the N candles after the decision.

        This is what makes ART training labels — the model sees:
          "I said BUY, here's what the prompt looked like,
           and here's what actually happened."
        """
        trade_map: dict[str, list[OrderResult]] = {}
        for trade in router.trade_log:
            trade_map.setdefault(trade.mint, []).append(trade)

        for event in self._events:
            mint      = event["mint"]
            dec_ts_str = event["decision"].get("timestamp", "")
            try:
                dec_ts = datetime.fromisoformat(dec_ts_str)
            except Exception:
                continue

            # Find the sell that closed this position after this decision
            sells = [
                t for t in trade_map.get(mint, [])
                if t.side == OrderSide.SELL and t.timestamp > dec_ts
            ]
            buys = [
                t for t in trade_map.get(mint, [])
                if t.side == OrderSide.BUY and t.timestamp <= dec_ts
            ]
            if sells and buys:
                last_buy  = buys[-1]
                first_sell = sells[0]
                pnl_pct   = (
                    (first_sell.effective_price / last_buy.effective_price) - 1
                ) * 100
                hold_secs = (
                    first_sell.timestamp - last_buy.timestamp
                ).total_seconds()
                event["outcome"] = {
                    "pnl_pct":       round(pnl_pct, 2),
                    "hold_secs":     round(hold_secs, 1),
                    "exit_price":    first_sell.effective_price,
                    "exit_slippage": first_sell.slippage_pct,
                    "exit_ts":       first_sell.timestamp.isoformat(),
                    "label":         self._label(pnl_pct, event["decision"]["action"]),
                }

    @staticmethod
    def _label(pnl_pct: float, action: str) -> str:
        """
        Ground-truth label for ART training.
        Converts outcome into a quality signal for the decision.
        """
        if action == "BUY":
            if pnl_pct > 20:   return "GOOD_BUY"
            if pnl_pct > 5:    return "OK_BUY"
            if pnl_pct > -5:   return "BREAK_EVEN_BUY"
            return "BAD_BUY"
        if action in ("SELL", "EXIT"):
            if pnl_pct > 20:   return "EARLY_SELL"   # left money on table
            if pnl_pct > 0:    return "GOOD_SELL"
            return "LOSS_SELL"
        if action == "HOLD":
            if pnl_pct > 10:   return "GOOD_HOLD"
            if pnl_pct < -10:  return "BAD_HOLD"     # should have exited
            return "OK_HOLD"
        return "UNKNOWN"

    def get_events(self) -> list[dict]:
        return self._events.copy()

    def close(self) -> None:
        if self._file:
            self._file.close()
