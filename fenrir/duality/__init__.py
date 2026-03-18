"""
fenrir.duality — FENRIR v2 Backtest/Live Duality Layer
=======================================================

File layout:
  protocols.py        — Abstract interfaces (DataFeed, OrderRouter, etc.)
  backtest_impls.py   — Concrete backtest implementations
  replay_and_art.py   — Incident replay + OpenPipe ART export
  execution_engine.py — Unified entrypoint + PromptTester

─────────────────────────────────────────────────────
QUICK START
─────────────────────────────────────────────────────

1. Run a backtest:

    from fenrir.duality.execution_engine import run_backtest
    import asyncio

    result = asyncio.run(run_backtest(
        strategy_cls=SniperStrategy,
        mints=["mint1", "mint2"],
        data_dir="./data",
        config={"initial_sol": 10.0, "min_candles": 25},
    ))

2. Test prompt quality before deploying:

    from fenrir.duality.execution_engine import PromptTester

    tester = PromptTester(SniperStrategy, "./data")
    report = asyncio.run(tester.run(mints=test_mints, scenarios=50))
    assert report["deploy_recommended"], "Prompt quality too low"

3. Replay a live incident:

    from fenrir.duality.replay_and_art import IncidentReplayer

    replayer = IncidentReplayer("./logs/live_events_20250110.jsonl")
    losses   = replayer.find_losses(min_loss_pct=-15.0)
    for event in losses:
        print(replayer.generate_incident_report(event))

4. Export ART training data from a backtest:

    from fenrir.duality.replay_and_art import ARTExporter
    from fenrir.duality.backtest_impls import BacktestRecorder

    # recorder is returned from run_backtest internally
    # or load from an existing events.jsonl:
    recorder = BacktestRecorder()
    # ... reload events ...
    exporter = ARTExporter(recorder)
    summary  = exporter.export("./art_data/run_001/")

─────────────────────────────────────────────────────
RULES FOR STRATEGY AUTHORS
─────────────────────────────────────────────────────

✅ DO:
  - Import ONLY from protocols.py in your strategy file
  - Always call recorder.record() after every decision
  - Always store raw_prompt and raw_response in ClaudeDecision
  - Keep strategy __init__ signature: (data, router, context, recorder, config)

❌ DON'T:
  - Import from backtest_impls.py or replay_and_art.py in strategy code
  - Add if self.mode == 'live': branches in strategy logic
  - Call Jupiter/Helius/WebSocket directly from strategy methods
  - Skip recorder.record() — you lose incident replay capability
"""

from .protocols import (
    DataFeed, OrderRouter, ContextProvider, ExecutionRecorder,
    TokenInfo, Position, OrderResult, OrderSide,
    ClaudeDecision, DecisionAction,
    OnChainSnapshot, SocialSnapshot,
)
from .backtest_impls import (
    HistoricalDataFeed, SimulatedRouter, BacktestContextProvider,
    BacktestRecorder, SlippageModel, ZeroSlippageModel,
)
from .replay_and_art import (
    ARTExporter, ARTSample, PreferencePair,
    IncidentReplayer, LiveExecutionRecorder,
    REWARD_MAP,
)
from .execution_engine import (
    run_backtest, run_live, BacktestResult, PromptTester,
    ExampleSniperStrategy,
)

__all__ = [
    # Protocols
    "DataFeed", "OrderRouter", "ContextProvider", "ExecutionRecorder",
    "TokenInfo", "Position", "OrderResult", "OrderSide",
    "ClaudeDecision", "DecisionAction",
    "OnChainSnapshot", "SocialSnapshot",
    # Backtest
    "HistoricalDataFeed", "SimulatedRouter", "BacktestContextProvider",
    "BacktestRecorder", "SlippageModel", "ZeroSlippageModel",
    # ART
    "ARTExporter", "ARTSample", "PreferencePair",
    "IncidentReplayer", "LiveExecutionRecorder", "REWARD_MAP",
    # Engine
    "run_backtest", "run_live", "BacktestResult", "PromptTester",
    "ExampleSniperStrategy",
]
