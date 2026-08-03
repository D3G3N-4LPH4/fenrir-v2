#!/usr/bin/env python3
"""
FENRIR - Signal backtest CLI (Phase 6.3)

Runs the drift-free portfolio backtester over historical samples loaded from a JSON
file and prints a report. Unlike the legacy tools/backtest.py, entry/exit come from the
real strategies (evaluate_token + TradeParams) via fenrir.backtest, so results reflect
what actually trades.

Usage:
    python -m tools.signal_backtest --samples history.json \
        --strategies momentum,mean_reversion [--min-sources 2]

The samples file is a JSON list (or {"samples": [...]}) of records shaped like
fenrir.backtest.loader (token_address, market_data, forward_prices, ...).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from fenrir.backtest import PortfolioBacktester, format_report, load_samples
from fenrir.config import BotConfig
from fenrir.strategies import STRATEGY_REGISTRY


def _build_strategies(ids: list[str], config: BotConfig) -> list[object]:
    strategies: list[object] = []
    for sid in ids:
        cls: Any = STRATEGY_REGISTRY.get(sid)  # concrete ctor takes a BotConfig
        if cls is None:
            print(f"unknown strategy '{sid}' (available: {', '.join(sorted(STRATEGY_REGISTRY))})")
            continue
        strategies.append(cls(config))
    return strategies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FENRIR signal backtester")
    parser.add_argument("--samples", required=True, help="JSON file of historical samples")
    parser.add_argument(
        "--strategies",
        default="momentum,mean_reversion",
        help="comma-separated strategy ids (default: momentum,mean_reversion)",
    )
    parser.add_argument("--min-sources", type=int, default=2, help="confluence threshold")
    args = parser.parse_args(argv)

    samples = load_samples(args.samples)
    if not samples:
        print(f"no usable samples in {args.samples}")
        return 1

    ids = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies = _build_strategies(ids, BotConfig())
    if not strategies:
        print("no valid strategies selected")
        return 1

    result = PortfolioBacktester().run(
        strategies, samples, confluence_min_sources=args.min_sources
    )
    print(f"\nLoaded {len(samples)} samples; ran {len(strategies)} strategies.\n")
    print(format_report(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
