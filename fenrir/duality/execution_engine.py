"""
execution_engine.py — FENRIR v2 Unified Execution Engine
=========================================================
The single entrypoint that enforces the backtest/live duality.
Strategy code never calls this directly — it receives the injected
interfaces. This file owns the wiring.

Two public functions:
  run_backtest(strategy_cls, data_dir, config) → BacktestResult
  run_live(strategy_cls, config)               → runs until stopped

Both call the exact same strategy code. The only difference is which
concrete implementations get injected.

Also provides: PromptTester — runs a strategy through N historical
scenarios and grades Claude's decision quality before any real SOL
is at risk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Type

from .protocols import (
    DataFeed, OrderRouter, ContextProvider, ExecutionRecorder,
    ClaudeDecision, DecisionAction,
)
from .backtest_impls import (
    HistoricalDataFeed, SimulatedRouter, BacktestContextProvider,
    BacktestRecorder, SlippageModel,
)
from .replay_and_art import ARTExporter, LiveExecutionRecorder

logger = logging.getLogger("fenrir.engine")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    config:          dict
    performance:     dict   # from SimulatedRouter.summary()
    art_summary:     dict   # from ARTExporter.export()
    recorder_path:   Path
    duration_secs:   float
    candles_replayed: int
    decisions_made:  int

    def print_summary(self) -> None:
        p = self.performance
        print("\n" + "═" * 55)
        print("  FENRIR BACKTEST RESULT")
        print("═" * 55)
        print(f"  Duration:     {self.duration_secs:.1f}s")
        print(f"  Candles:      {self.candles_replayed:,}")
        print(f"  Decisions:    {self.decisions_made}")
        print(f"  ─────────────────────────────────────")
        print(f"  Initial SOL:  {p['initial_sol']:.4f}")
        print(f"  Final SOL:    {p['final_sol']:.4f}")
        print(f"  PnL:          {p['pnl_sol']:+.4f} SOL ({p['pnl_pct']:+.2f}%)")
        print(f"  Trades:       {p['buys']} buys / {p['sells']} sells")
        print(f"  Avg slippage: {p['avg_slippage']:.2%}")
        print(f"  ─────────────────────────────────────")
        a = self.art_summary
        print(f"  ART samples:  {a.get('exportable_samples', 0)}")
        print(f"  DPO pairs:    {a.get('preference_pairs', 0)}")
        print(f"  Avg reward:   {a.get('avg_reward', 0):+.3f}")
        print("═" * 55)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

async def run_backtest(
    strategy_cls,
    mints:      list[str],
    data_dir:   str | Path,
    config:     dict | None = None,
    output_dir: str | Path | None = None,
    slippage:   SlippageModel | None = None,
) -> BacktestResult:
    """
    Wire up backtest implementations and run strategy over historical data.

    Parameters
    ----------
    strategy_cls : class
        Your strategy class. Must accept (data, router, context, recorder, config).
    mints : list[str]
        Token mint addresses to replay.
    data_dir : Path
        Directory containing candles/ and snapshots/ parquet files.
    config : dict
        Strategy configuration (position sizing, thresholds, etc.).
    output_dir : Path
        Where to write recorder JSONL and ART export. Defaults to data_dir/runs/.
    slippage : SlippageModel
        Pluggable slippage model. Defaults to calibrated pump.fun model.
    """
    config     = config or {}
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir / "runs" / _run_id()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = datetime.now(timezone.utc)
    logger.info(f"Starting backtest | mints={len(mints)} | output={output_dir}")

    # ── Build concrete implementations ────────────────────────────────────
    feed     = HistoricalDataFeed(data_dir)
    feed.load(mints)

    router   = SimulatedRouter(
        initial_sol=config.get("initial_sol", 10.0),
        slippage=slippage or SlippageModel(seed=config.get("seed", 42)),
        simulate_latency=config.get("simulate_latency", True),
    )
    # Wire price/liquidity into router so it sees replay-current prices
    router._set_price_fn(lambda mint: feed._price_now.get(mint, 0.0))
    router._set_liquidity_fn(feed.get_liquidity_fn())

    context  = BacktestContextProvider(feed)
    recorder = BacktestRecorder(output_path=output_dir / "events.jsonl")

    # ── Instantiate strategy with injected dependencies ────────────────────
    strategy = strategy_cls(
        data=feed,
        router=router,
        context=context,
        recorder=recorder,
        config=config,
    )

    # ── Replay loop ────────────────────────────────────────────────────────
    candles_total  = 0
    decisions_total = 0

    async def candle_callback(mint: str, candle: dict) -> None:
        nonlocal candles_total, decisions_total
        candles_total += 1
        n = await strategy.on_candle(mint, candle)
        decisions_total += n or 0

    # Run tokens sequentially (or in parallel for speed — your choice)
    for mint in mints:
        await feed.subscribe_candles(
            mint,
            lambda c, m=mint: candle_callback(m, c),
        )

    # ── Backfill outcomes ──────────────────────────────────────────────────
    recorder.backfill_outcomes(router)
    await recorder.flush()
    recorder.close()

    # ── Export ART data ────────────────────────────────────────────────────
    exporter   = ARTExporter(recorder)
    art_summary = exporter.export(output_dir / "art")

    # ── Performance summary ────────────────────────────────────────────────
    duration = (datetime.now(timezone.utc) - t0).total_seconds()
    result   = BacktestResult(
        config=config,
        performance=router.summary(),
        art_summary=art_summary,
        recorder_path=output_dir / "events.jsonl",
        duration_secs=duration,
        candles_replayed=candles_total,
        decisions_made=decisions_total,
    )
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------

async def run_live(
    strategy_cls,
    live_data:    DataFeed,        # HeliusLiveFeed or your WebSocket feed
    live_router:  OrderRouter,     # JupiterRouter with real wallet
    live_context: ContextProvider, # LiveContextProvider
    log_dir:      str | Path,
    config:       dict | None = None,
) -> None:
    """
    Wire up live implementations and run strategy indefinitely.
    Identical strategy code to run_backtest — only the implementations differ.
    """
    config  = config or {}
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    recorder = LiveExecutionRecorder(
        log_path=log_dir / f"live_events_{_run_id()}.jsonl"
    )

    strategy = strategy_cls(
        data=live_data,
        router=live_router,
        context=live_context,
        recorder=recorder,
        config=config,
    )

    logger.info("FENRIR live mode starting — strategy running")
    try:
        await strategy.run()   # strategy owns its own loop in live mode
    except KeyboardInterrupt:
        logger.info("Live mode stopped by user")
    except Exception as e:
        logger.exception(f"Live mode crashed: {e}")
        raise
    finally:
        await recorder.flush()
        recorder.close()


# ---------------------------------------------------------------------------
# Prompt Tester — grade Claude decisions before spending SOL
# ---------------------------------------------------------------------------

class PromptTester:
    """
    Runs a strategy through N historical scenarios, grades Claude's
    decision quality, and identifies systematic prompt failures.

    Use this before deploying a prompt change to live:
      tester = PromptTester(strategy_cls, data_dir)
      report = await tester.run(mints=test_mints, scenarios=50)
      if report["avg_reward"] < 0.3:
          raise ValueError("Prompt quality too low — do not deploy")

    This is the "test your prompts, don't assume them" gate.
    """

    def __init__(self, strategy_cls, data_dir: str | Path, config: dict | None = None):
        self.strategy_cls = strategy_cls
        self.data_dir     = Path(data_dir)
        self.config       = config or {}

    async def run(
        self,
        mints:     list[str],
        scenarios: int = 50,
        output_dir: str | Path | None = None,
    ) -> dict:
        """
        Run backtest and return prompt quality report.
        """
        run_output = Path(output_dir) if output_dir else (
            self.data_dir / "prompt_tests" / _run_id()
        )

        result = await run_backtest(
            strategy_cls=self.strategy_cls,
            mints=mints,
            data_dir=self.data_dir,
            config={**self.config, "max_decisions": scenarios},
            output_dir=run_output,
        )

        art = result.art_summary
        dist = art.get("label_distribution", {})

        bad_decisions = sum(
            dist.get(l, 0) for l in ["BAD_BUY", "BAD_HOLD", "LOSS_SELL"]
        )
        good_decisions = sum(
            dist.get(l, 0) for l in ["GOOD_BUY", "GOOD_SELL", "GOOD_HOLD", "OK_BUY", "OK_HOLD"]
        )
        total = max(art.get("exportable_samples", 1), 1)

        report = {
            "avg_reward":       art.get("avg_reward", 0),
            "avg_pnl_pct":      art.get("avg_pnl_pct", 0),
            "pnl_pct":          result.performance.get("pnl_pct", 0),
            "good_rate":        round(good_decisions / total, 3),
            "bad_rate":         round(bad_decisions / total, 3),
            "total_decisions":  total,
            "label_dist":       dist,
            "deploy_recommended": art.get("avg_reward", 0) > 0.3 and
                                  result.performance.get("pnl_pct", -999) > 0,
            "output_dir":       str(run_output),
        }

        self._print_prompt_report(report)
        return report

    @staticmethod
    def _print_prompt_report(report: dict) -> None:
        deploy = "✅ DEPLOY" if report["deploy_recommended"] else "❌ DO NOT DEPLOY"
        print("\n" + "═" * 55)
        print(f"  PROMPT QUALITY REPORT  {deploy}")
        print("═" * 55)
        print(f"  Avg reward:    {report['avg_reward']:+.3f}  (target: > 0.30)")
        print(f"  Avg PnL:       {report['avg_pnl_pct']:+.2f}%  (target: > 0%)")
        print(f"  Good rate:     {report['good_rate']:.0%}")
        print(f"  Bad rate:      {report['bad_rate']:.0%}")
        print(f"  Total samples: {report['total_decisions']}")
        print("═" * 55)


# ---------------------------------------------------------------------------
# Example strategy skeleton — shows exactly what a strategy must look like
# ---------------------------------------------------------------------------

class ExampleSniperStrategy:
    """
    Minimal strategy skeleton that satisfies the duality contract.
    Replace this with your real SniperStrategy — keeping this exact
    constructor signature is all that's required.
    """

    def __init__(
        self,
        data:     DataFeed,
        router:   OrderRouter,
        context:  ContextProvider,
        recorder: ExecutionRecorder,
        config:   dict,
    ):
        self.data     = data
        self.router   = router
        self.context  = context
        self.recorder = recorder
        self.config   = config
        self._candle_count: dict[str, int] = {}

    async def on_candle(self, mint: str, candle: dict) -> int:
        """
        Called on every candle close — in backtest by the replay loop,
        in live by the WebSocket callback.
        Returns number of decisions made (for stats).
        """
        count = self._candle_count.get(mint, 0) + 1
        self._candle_count[mint] = count

        # Don't trade until we have enough history
        if count < self.config.get("min_candles", 30):
            return 0

        ohlcv    = await self.data.get_ohlcv(mint, lookback=100)
        price    = await self.data.get_current_price(mint)
        on_chain = await self.context.get_on_chain(mint)
        social   = await self.context.get_social(mint)
        smc_ctx  = await self.context.get_smc_context(mint)

        # ── Build Claude prompt (same in backtest and live) ──────────────
        prompt = self._build_prompt(mint, price, ohlcv, on_chain, social, smc_ctx)

        # ── Call Claude (you inject your ClaudeClient the same way) ──────
        # decision = await self.claude.decide(prompt)
        # For skeleton: simulate a decision
        decision = ClaudeDecision(
            action=DecisionAction.HOLD,
            confidence=0.6,
            reasoning="Skeleton strategy — always holds",
            raw_prompt=prompt,
            raw_response='{"action": "HOLD", "confidence": 0.6, "reasoning": "..."}',
        )

        # ── Execute order ────────────────────────────────────────────────
        order = None
        if decision.action == DecisionAction.BUY:
            order = await self.router.buy(mint, decision.sol_amount)
        elif decision.action in (DecisionAction.SELL, DecisionAction.EXIT):
            order = await self.router.sell(mint, decision.sell_pct)

        # ── Record (mandatory — this feeds ART and incident replay) ──────
        context_snapshot = {
            "on_chain": on_chain.__dict__ if on_chain else {},
            "social":   social.__dict__   if social   else {},
            "smc":      smc_ctx,
            "candle":   candle,
        }
        await self.recorder.record(
            mint=mint,
            decision=decision,
            order=order,
            context=context_snapshot,
        )

        return 1

    async def run(self) -> None:
        """Live mode: strategy owns its loop via subscribe_candles."""
        raise NotImplementedError("Implement run() for live mode")

    def _build_prompt(self, mint, price, ohlcv, on_chain, social, smc_ctx) -> str:
        return f"""
You are FENRIR's trade decision engine for a Solana memecoin.

Token: {mint[:8]}
Price: {price:.8f} SOL

{smc_ctx}

On-chain: {on_chain}
Social: {social}

Decide: BUY, HOLD, SELL, or EXIT.
Respond with JSON: {{"action": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
