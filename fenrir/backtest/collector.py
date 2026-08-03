#!/usr/bin/env python3
"""
FENRIR - Forward-price sample collector (empirical phase)

Turns live observation into backtester fuel. When a strategy flags a token, this
records the entry MarketData snapshot and then samples the token's price forward over
a window, writing one ``BacktestSample``-shaped record (see fenrir.backtest.loader) to a
JSONL file. Feed that file back through the backtester to calibrate the strategies on
real outcomes.

Read-only: it only reads prices and appends to a file — it never trades. The price
source is injected (``get_price(token) -> SOL/token | None``) so this is fully testable
with no network; the bot passes a closure over the aggregated price feed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

# The MarketData fields worth persisting for a replay (numeric + venue ids; not `raw`).
_MD_FIELDS = (
    "token_address",
    "pair_address",
    "dex_id",
    "price_usd",
    "price_sol",
    "liquidity_usd",
    "market_cap_usd",
    "fdv_usd",
    "volume_5m_usd",
    "volume_1h_usd",
    "volume_6h_usd",
    "volume_24h_usd",
    "txns_5m_buys",
    "txns_5m_sells",
    "txns_1h_buys",
    "txns_1h_sells",
    "unique_buyers_1h",
    "price_change_5m_pct",
    "price_change_1h_pct",
    "price_change_6h_pct",
    "price_change_24h_pct",
    "age_minutes",
)

PriceGetter = Callable[[str], Awaitable[float | None]]


def market_data_to_dict(market_data: Any) -> dict[str, Any]:
    """Serialize the replay-relevant MarketData fields (JSON-safe, no ``raw``)."""
    return {f: getattr(market_data, f) for f in _MD_FIELDS if hasattr(market_data, f)}


def build_record(
    token_address: str,
    market_data: Any,
    forward_prices: list[float],
    symbol: str,
    frame_seconds: float,
) -> dict[str, Any]:
    """Assemble a BacktestSample-shaped record (see fenrir.backtest.loader)."""
    return {
        "token_address": token_address,
        "symbol": symbol,
        "market_data": market_data_to_dict(market_data),
        "forward_prices": forward_prices,
        "frame_seconds": frame_seconds,
    }


class ForwardPriceCollector:
    """Samples a flagged token's price forward and appends a replay record to JSONL."""

    def __init__(
        self,
        get_price: PriceGetter,
        out_path: str | Path,
        frame_seconds: float = 60.0,
        max_frames: int = 30,
        logger: Any = None,
    ) -> None:
        self._get_price = get_price
        self.out_path = Path(out_path)
        self.frame_seconds = frame_seconds
        self.max_frames = max_frames
        self._logger = logger
        self.collected = 0

    async def collect(self, token_address: str, market_data: Any, symbol: str = "") -> dict | None:
        """Sample the price forward for ``max_frames`` (first sample = entry) and append
        the record. Returns the record, or None if fewer than two prices were captured
        (nothing to replay). Never raises on a feed hiccup — bad reads are skipped."""
        prices: list[float] = []
        for i in range(self.max_frames):
            try:
                price = await self._get_price(token_address)
            except Exception as e:  # noqa: BLE001 - a feed hiccup must not sink the sample
                self._log("warning", f"collector price error for {token_address[:8]}...: {e}")
                price = None
            if price is not None and price > 0:
                prices.append(float(price))
            if i < self.max_frames - 1 and self.frame_seconds > 0:
                await asyncio.sleep(self.frame_seconds)

        if len(prices) < 2:
            return None

        record = build_record(token_address, market_data, prices, symbol, self.frame_seconds)
        self._append(record)
        self.collected += 1
        self._log(
            "info",
            f"collected sample {token_address[:8]}... ({len(prices)} frames) -> {self.out_path.name}",
        )
        return record

    def _append(self, record: dict) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        method = getattr(self._logger, level, None)
        try:
            if callable(method):
                method(msg)
                return
        except TypeError:
            pass
        fallback = getattr(self._logger, "warning", None) or getattr(self._logger, "info", None)
        if callable(fallback):
            fallback(msg)
