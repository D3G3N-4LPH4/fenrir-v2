#!/usr/bin/env python3
"""
FENRIR - EVM collect-and-report CLI (Phase 7, read-only)

One command to produce a real EVM backtest report: run the EVM read loop against LIVE
DexScreener data for a bounded window, collecting each flagged token's forward-price path
to JSONL, then backtest those samples and print the report. Read-only end to end — it
fetches public market data and evaluates; it never signs or sends a transaction.

Usage:
    python -m tools.evm_collect --minutes 120
    python -m tools.evm_collect --minutes 60 --watchlist 0xabc...,0xdef... \
        --chains ethereum,base --strategies momentum,mean_reversion

Token source: DexScreener trending/boosted tokens for the enabled chains (younger, moving
— what the strategies target), plus any --watchlist addresses. Leave it running for a real
window: a strategy only fires on a token caught young + in the right regime, which is a
function of sampling continuously over time.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

from fenrir.backtest import PortfolioBacktester, format_report, load_jsonl
from fenrir.config import BotConfig
from fenrir.strategies import STRATEGY_REGISTRY


def _build_strategies(ids: list[str], config: BotConfig) -> list[Any]:
    out: list[Any] = []
    for sid in ids:
        cls: Any = STRATEGY_REGISTRY.get(sid)  # concrete ctor takes a BotConfig
        if cls is None:
            print(f"unknown strategy '{sid}' (skipped)")
            continue
        out.append(cls(config))
    return out


def _report_from_samples(path: str, strategy_ids: list[str], config: BotConfig) -> str:
    """Backtest whatever samples were collected and render the report. Returns a plain
    message when nothing was collected (rather than a misleading empty report)."""
    samples = load_jsonl(path)
    if not samples:
        return f"No EVM samples collected in {path} — nothing to report yet."
    strategies = _build_strategies(strategy_ids, config)
    result = PortfolioBacktester().run(strategies, samples)
    return f"Collected {len(samples)} EVM samples → {path}\n\n" + format_report(result)


def _trending_source(provider: Any, chains: list[Any], watchlist: list[str]) -> Any:
    """Async token source: DexScreener boosted/trending addresses for the chains, plus
    the static watchlist, de-duplicated."""

    async def _get() -> list[str]:
        tokens: list[str] = list(watchlist)
        for chain in chains:
            try:
                tokens += await provider.fetch_boosted_addresses(chain)
            except Exception:  # noqa: BLE001,S112 - one chain's provider hiccup must not stop the rest
                continue
        return list(dict.fromkeys(tokens))

    return _get


async def _run_window(scanner: Any, minutes: float, interval: float) -> None:
    """Scan for the window, then drain in-flight collections so late samples aren't lost."""
    deadline = time.monotonic() + minutes * 60.0
    cycle = 0
    while time.monotonic() < deadline:
        cycle += 1
        await scanner.scan_once()
        print(
            f"[cycle {cycle}] evaluated={scanner.tokens_evaluated} "
            f"surfaced={scanner.signals_surfaced}",
            flush=True,
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(interval, remaining))
    print("window over — draining in-flight collections...", flush=True)
    await scanner.drain_collections(timeout=None)


async def _main(args: argparse.Namespace) -> int:
    from fenrir.backtest import ForwardPriceCollector
    from fenrir.discovery.models import Chain
    from fenrir.discovery.providers.dexscreener import DexScreenerProvider
    from fenrir.evm import EvmEvaluatorScanner

    config = BotConfig()
    strategy_ids = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies = _build_strategies(strategy_ids, config)
    if not strategies:
        print("no valid strategies")
        return 1

    chain_names = [c.strip().lower() for c in args.chains.split(",") if c.strip()]
    chains: list[Any] = []
    for c in chain_names:
        try:
            chains.append(Chain(c))
        except ValueError:
            print(f"unknown chain '{c}' (skipped)")
    watchlist = [w.strip() for w in args.watchlist.split(",") if w.strip()]

    provider = DexScreenerProvider()

    async def get_price(token: str) -> float | None:
        snap = await provider.fetch_snapshot(token)
        return float(snap.price_usd) if snap is not None and snap.price_usd > 0 else None

    evaluator = config.build_evm_evaluator(
        strategies=strategies, fetch_snapshot=provider.fetch_snapshot
    )
    collector = ForwardPriceCollector(
        get_price=get_price,
        out_path=args.out,
        frame_seconds=args.frame_seconds,
        max_frames=args.frames,
    )
    scanner = EvmEvaluatorScanner(
        evaluator=evaluator,
        token_source=_trending_source(provider, chains, watchlist),
        enabled_chains={c.value for c in chains} if chains else None,
        interval_seconds=args.interval,
        max_tokens_per_cycle=args.max_tokens,
        collector=collector,
    )

    print(
        f"EVM collect (READ-ONLY): {args.minutes} min, chains={chain_names}, "
        f"strategies={strategy_ids}, out={args.out}",
        flush=True,
    )
    try:
        await _run_window(scanner, args.minutes, args.interval)
    finally:
        try:
            await provider.close()
        except Exception:  # noqa: BLE001,S110
            pass

    print("\n" + _report_from_samples(args.out, strategy_ids, config))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FENRIR EVM collect-and-report (read-only)")
    p.add_argument("--minutes", type=float, default=60.0, help="window length (minutes)")
    p.add_argument("--chains", default="ethereum,base,bnb")
    p.add_argument("--watchlist", default="", help="extra comma-separated token addresses")
    p.add_argument("--strategies", default="momentum,mean_reversion")
    p.add_argument("--interval", type=float, default=60.0, help="seconds between scan cycles")
    p.add_argument("--frames", type=int, default=30, help="price samples per flagged token")
    p.add_argument("--frame-seconds", type=float, default=60.0, dest="frame_seconds")
    p.add_argument("--max-tokens", type=int, default=25, dest="max_tokens")
    p.add_argument("--out", default="evm_samples.jsonl")
    args = p.parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
