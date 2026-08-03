#!/usr/bin/env python3
"""
FENRIR - Backtest sample loader (Phase 6.2)

Builds ``BacktestSample`` objects from plain dict records (and JSON files) so real
historical data can feed the drift-free backtester. Each record:

    {
      "token_address": "...",
      "symbol": "ABC",
      "market_data": { ...MarketData fields... },   # entry snapshot
      "forward_prices": [1.0, 1.02, ...],            # SOL/token path, [0] = entry
      "frame_seconds": 60.0                          # optional, price spacing
    }

Malformed records are skipped (fail-open) so one bad row can't sink a whole dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fenrir.backtest.models import BacktestSample
from fenrir.filters import MarketData


def sample_from_dict(record: dict[str, Any]) -> BacktestSample | None:
    """Build one BacktestSample from a record, or None if it is unusable."""
    token = record.get("token_address")
    prices = record.get("forward_prices")
    if not token or not isinstance(prices, list) or not prices:
        return None
    try:
        forward_prices = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None

    md_fields = dict(record.get("market_data") or {})
    md_fields.setdefault("token_address", token)
    try:
        market_data = MarketData(**md_fields)
    except TypeError:
        # Unknown keys / bad types in the snapshot — skip rather than guess.
        return None

    symbol = record.get("symbol", "") or ""
    token_data = {"token_address": token, "symbol": symbol}
    return BacktestSample(
        token_address=token,
        token_data=token_data,
        market_data=market_data,
        forward_prices=forward_prices,
        frame_seconds=float(record.get("frame_seconds", 60.0) or 60.0),
        symbol=symbol,
    )


def samples_from_dicts(records: list[dict[str, Any]]) -> list[BacktestSample]:
    """Parse many records, dropping any that are malformed."""
    out: list[BacktestSample] = []
    for record in records:
        sample = sample_from_dict(record)
        if sample is not None:
            out.append(sample)
    return out


def load_samples(path: str | Path) -> list[BacktestSample]:
    """Load samples from a JSON file — either a top-level list, or an object with a
    ``"samples"`` list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("samples", []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        return []
    return samples_from_dicts(records)
