#!/usr/bin/env python3
"""
FENRIR Bot v2 - The Orchestrator

Where all components harmonize into a single, elegant system.

v2 additions (inspired by OpenFang patterns):
- Event Bus: decoupled alerting via pluggable adapters
- Audit Trail: Merkle hash-chain tamper-evident logging
- Strategy System: pluggable, self-contained trading strategies
- Historical Memory: cross-session AI learning
- Budget Tracker: per-strategy SOL budget enforcement
"""

import asyncio
import json
import os
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from fenrir.ai.brain import ClaudeBrain
from fenrir.ai.market_geometry import MarketGeometryAnalyzer
from fenrir.config import BotConfig, TradingMode
from fenrir.core.agents import AgentPipeline
from fenrir.core.circuit_breaker import ServiceBreakers
from fenrir.core.client import SolanaClient
from fenrir.core.dump_recovery import (
    OuroborosConfig,
    PostDumpRecoveryDetector,
    ouroboros_detected_event,
)
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.core.positions import Position, PositionManager
from fenrir.data.audit import AuditChain, AuditEventType
from fenrir.data.historical_memory import HistoricalMemory
from fenrir.data.price_feed import PriceFeedManager
from fenrir.events.adapters.audit import AuditAdapter
from fenrir.events.adapters.health import AIHealthMonitor, HealthMonitorConfig
from fenrir.events.adapters.log import LogAdapter
from fenrir.events.adapters.telegram import TelegramAdapter
from fenrir.events.bus import EventBus
from fenrir.events.types import (
    ai_decision_event,
    ai_override_event,
    bot_lifecycle_event,
    budget_exhausted_event,
    buy_executed_event,
    error_event,
    sell_executed_event,
    signal_confluence_event,
    token_detected_event,
    trade_failed_event,
)
from fenrir.filters import MarketFilter, MarketFilterConfig, SecurityFilter
from fenrir.logger import FenrirLogger
from fenrir.protocol.jito import JitoMEVProtection
from fenrir.strategies import STRATEGY_REGISTRY, SniperStrategy
from fenrir.strategies.base import TradingStrategy
from fenrir.strategies.sniper import AIScoutStrategy
from fenrir.trading.engine import TradingEngine
from fenrir.trading.monitor import PumpFunMonitor
from fenrir.trading.scanner import MarketScanner
from fenrir.trading.smart_money import SmartMoneyTracker


class SignalStrategy(Protocol):
    """Structural type for market-data-aware (uses_market_data=True) strategies."""

    def evaluate_token(self, token_data: dict[str, Any], market_data: Any = None) -> Any: ...

    def build_ai_context(self, signal: Any) -> str: ...


class FenrirBot:
    """
    The conductor of this symphony.
    Where all components harmonize into a single, elegant system.

    v2: Now with event bus, audit trail, strategy system,
    historical memory, and budget enforcement.
    """

    def __init__(self, config: BotConfig, strategies: list[str] | None = None):
        self.config = config
        self.logger = FenrirLogger(config)

        # Validate configuration
        errors = config.validate()
        if errors:
            for err in errors:
                self.logger.error("Configuration error", Exception(err))
            raise ValueError("Invalid configuration")

        # ── Circuit breakers (per-service fault isolation) ──────
        self.breakers = ServiceBreakers()

        # ── Core infrastructure ─────────────────────────────────
        # Multi-wallet rotation pool so concurrent positions aren't capped by a
        # single wallet's balance/limits. `self.wallet` stays the primary wallet for
        # backward compatibility (the engine signs with it); the pool is available
        # for per-trade rotation + balance monitoring, and attributes trades to the
        # executing wallet in the audit chain.
        self.wallet_pool = config.build_wallet_pool(self.logger)
        self.wallet = self.wallet_pool.primary
        self.solana_client = SolanaClient(config, self.logger, breaker=self.breakers.solana_rpc)
        self.jupiter = JupiterSwapEngine(config, self.logger, breaker=self.breakers.jupiter)
        self.positions = PositionManager(config, self.logger)

        # Jito MEV protection. Needed when the flat use_jito flag is set OR when
        # per-strategy profiles are active — a profile (e.g. the sniper's
        # UltraEarlySnipe) can enable atomic Jito bundles even if the flat flag is
        # off, and _resolve_use_jito returns False when this instance is None.
        self.jito: JitoMEVProtection | None = None
        if config.use_jito or config.tx_profiles_enabled:
            self.jito = JitoMEVProtection(
                region="mainnet",
                tip_lamports=config.jito_tip_lamports,
            )

        self.price_feed = PriceFeedManager()

        # Trading engine
        self.trading_engine = TradingEngine(
            config,
            self.wallet,
            self.solana_client,
            self.jupiter,
            self.positions,
            self.logger,
            jito=self.jito,
            price_feed=self.price_feed,
            wallet_pool=self.wallet_pool,
        )

        # Monitor
        self.monitor = PumpFunMonitor(config, self.solana_client, self.logger)

        # Market scanner (multi-tier discovery; started only when enabled)
        self.scanner = MarketScanner(config, self.jupiter, self.logger)
        # Handle to the running scanner task so it can be started/stopped live
        # from apply_config_update() without restarting the bot.
        self._scanner_task: asyncio.Task | None = None

        # Multi-chain discovery scanner (discovery-only; built in start() when
        # config.discovery_enabled, since it needs the shared providers).
        self.discovery_scanner: Any = None
        self._discovery_task: asyncio.Task | None = None
        self._discovery_dex: Any = None  # shared DexScreener provider (closed on stop)

        # Read-only arbitrage monitor (built in start() when arbitrage_monitor_enabled).
        # Checks recently-seen + held tokens for cross-pool divergence; never executes.
        self.arbitrage_monitor: Any = None
        self.arbitrage_scanner: Any = None
        self._arbitrage_task: asyncio.Task | None = None
        self._arb_dex: Any = None  # DexScreener provider for the arb monitor (closed on stop)
        # Bounded ring of recently-detected tokens the arb monitor can sample.
        self._recent_tokens: deque[str] = deque(maxlen=100)

        # Read-only signal confluence surfacing (Phase 5.3). Collects normalized signals
        # over a window and emits SIGNAL_CONFLUENCE when independent strategies agree.
        # Never changes sizing/execution. Built only when enabled.
        self.signal_aggregator: Any = None
        if config.signal_confluence_enabled:
            from fenrir.signals import SignalAggregator

            self.signal_aggregator = SignalAggregator(
                ttl_seconds=config.signal_confluence_ttl_seconds
            )

        # Smart-money / whale wallet tracker (follow curated wallets; opt-in).
        self.smart_money = SmartMoneyTracker(
            config, self.solana_client, self.trading_engine.pumpfun, self.logger
        )
        self._smart_money_task: asyncio.Task | None = None

        # Always-on multi-lens AI evaluator, used as a fallback for launches that
        # no active strategy claimed (opt-in via ai_evaluate_all_launches).
        self._ai_scout = AIScoutStrategy(config)

        # ── NEW: Event Bus ──────────────────────────────────────
        self.event_bus = EventBus()
        self._setup_event_bus()

        # ── NEW: Audit Trail ────────────────────────────────────
        db_path = config.log_file.replace(".log", ".db") if config.log_file else "fenrir_trades.db"
        self.audit = AuditChain(db_path=db_path)

        # Register audit adapter on the event bus
        self.event_bus.register(AuditAdapter(self.audit))

        # ── NEW: AI Health Monitor ──────────────────────────────
        self.health_monitor = AIHealthMonitor(
            config=HealthMonitorConfig(),
            event_bus=self.event_bus,
        )
        self.event_bus.register(self.health_monitor)

        # ── NEW: Historical Memory ──────────────────────────────
        self.historical_memory = HistoricalMemory(db_path=db_path)

        # ── NEW: Budget Tracker ─────────────────────────────────
        # Dynamic allocation (config-driven) scales each strategy's budget by its
        # trailing win rate, within floor/ceiling multipliers.
        self.budget_tracker = config.build_budget_tracker()
        # Master safety valve: cap net live SOL exposure across ALL strategies.
        if config.global_daily_sol_limit > 0:
            self.budget_tracker.set_global_limit(config.global_daily_sol_limit)
            self.logger.info(
                f"Global daily SOL limit: {config.global_daily_sol_limit} SOL (all strategies)"
            )

        # ── Portfolio-level risk (above per-strategy budgets) ───
        # Aggregate exposure cap, creator/launch-window correlation limits, and a
        # portfolio drawdown circuit breaker that halts ALL buys.
        self.portfolio_risk = config.build_portfolio_risk()

        # Ouroboros / dump recovery detector
        self.dump_detector = PostDumpRecoveryDetector(
            config=OuroborosConfig(
                dump_threshold_pct=30.0,  # Flag if drops 30%+ from peak
                recovery_threshold_pct=10.0,  # And then recovers 10%+
                max_recovery_to_flag_pct=60.0,  # But less than 60% (full recovery = legit)
                tightened_trailing_stop_pct=8.0,  # Tighten trail to 8% on Ouroboros
            )
        )

        # Market geometry analyzer (pre-entry informed pipeline)
        self.market_geometry = MarketGeometryAnalyzer()

        # ── NEW: Strategy System ────────────────────────────────
        self.strategies: list[TradingStrategy] = []
        self._init_strategies(strategies)

        # ── Multi-agent decision pipeline (Phase 3, strangler) ──────
        # When enabled, detections are handed to queue-backed Scanner/Sizing/Execution
        # agents instead of the inline loop, decoupling ingestion from slow AI evals and
        # trades. Off by default; the callables delegate to the same scan/size/execute
        # helpers the inline path uses, so decisions are identical. Started in start().
        self.agent_pipeline: AgentPipeline | None = None
        if config.multi_agent_pipeline_enabled:
            self.agent_pipeline = AgentPipeline(
                self.event_bus,
                claims=self._pipeline_claims,
                size=self._pipeline_size,
                execute=self._pipeline_execute,
                logger=self.logger,
                scanner_from_bus=False,  # fed full token_data via submit(), not the lossy bus event
            )

        # ── Pre-trade filters (off by default; see BotConfig flags) ──
        # Security hard-gate: instantiated only when enabled.
        self.security_filter: SecurityFilter | None = None
        if config.security_filter_enabled:
            self.security_filter = SecurityFilter(
                config.build_security_filter_config(),
                config.rpc_url,
                config.helius_api_key,
            )

        # Market data provider: needed when the market gate is on OR any active
        # strategy is market-data-aware (its evaluate_token needs a snapshot).
        # Built with enabled=True as a pure data source; the *gate* (skip on
        # not-passed) is applied separately, only when market_filter_enabled.
        self._needs_market_data = config.market_filter_enabled or any(
            s.uses_market_data for s in self.strategies
        )
        self.market_filter: MarketFilter | None = None
        if self._needs_market_data:
            self.market_filter = MarketFilter(
                MarketFilterConfig(
                    enabled=True,
                    fail_open_on_fetch_error=config.market_fail_open_on_fetch_error,
                )
            )

        # ── AI Brain (with historical memory) ───────────────────
        self.claude_brain = ClaudeBrain(
            config, self.logger, breaker=self.breakers.openrouter, db_path=db_path, audit=self.audit
        )

        self.running = False

    def _setup_event_bus(self) -> None:
        """Register event adapters based on configuration."""
        # Always register the log adapter
        self.event_bus.register(LogAdapter())

        # Telegram adapter (if configured)
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        tg_chat = os.getenv("TELEGRAM_CHAT_ID", "")
        if tg_token and tg_chat:
            self.event_bus.register(
                TelegramAdapter(
                    bot_token=tg_token,
                    chat_id=tg_chat,
                    alert_on_trades=True,
                    alert_on_errors=True,
                    breaker=self.breakers.telegram,
                )
            )

    def _init_strategies(self, strategy_ids: list[str] | None) -> None:
        """Initialize trading strategies."""
        if not strategy_ids:
            # Fall back to the config surface (ENABLED_STRATEGIES / default
            # ["sniper"]). The signal strategies stay off unless opted in.
            strategy_ids = self.config.enabled_strategies or ["sniper"]

        for sid in strategy_ids:
            cls = STRATEGY_REGISTRY.get(sid)
            if cls:
                # Concrete strategies accept the bot config; the abstract base's
                # __init__ signature doesn't reflect that, so narrow the type.
                strategy = cast(Callable[[BotConfig], TradingStrategy], cls)(self.config)
                self.strategies.append(strategy)
                self.logger.info(
                    f"Strategy loaded: {strategy.display_name} ({strategy.strategy_id})"
                )
            else:
                self.logger.warning(
                    f"Unknown strategy: {sid} " f"(available: {list(STRATEGY_REGISTRY.keys())})"
                )

        if not self.strategies:
            # Fallback: always have at least the sniper
            self.strategies.append(SniperStrategy(self.config))
            self.logger.info("Fallback: loaded default Sniper strategy")

    def _find_strategy(self, strategy_id: str) -> TradingStrategy | None:
        return next((s for s in self.strategies if s.strategy_id == strategy_id), None)

    async def set_strategy_enabled(self, strategy_id: str, enabled: bool) -> tuple[bool, str]:
        """Switch a strategy on/off live from the dashboard.

        enable: resume it if already loaded, else instantiate and add it to the
        active set (building the market-data provider on demand if the strategy
        needs it). disable: pause it (keeps existing positions; stops new entries)
        — we pause rather than unload so state/positions aren't stranded.

        Returns (ok, message). ok=False only for an unknown strategy id.
        """
        cls = STRATEGY_REGISTRY.get(strategy_id)
        if cls is None:
            return False, f"Unknown strategy '{strategy_id}'"

        existing = self._find_strategy(strategy_id)
        if enabled:
            if existing is not None:
                existing.resume()
                return True, "resumed"
            strategy = cast(Callable[[BotConfig], TradingStrategy], cls)(self.config)
            self.strategies.append(strategy)
            # Signal strategies need a MarketData provider; build it if the bot
            # started without one (no market-data strategy was enabled at boot).
            if strategy.uses_market_data and self.market_filter is None:
                self._needs_market_data = True
                self.market_filter = MarketFilter(
                    MarketFilterConfig(
                        enabled=True,
                        fail_open_on_fetch_error=self.config.market_fail_open_on_fetch_error,
                    )
                )
            self.logger.info(f"Strategy activated live: {strategy.display_name} ({strategy_id})")
            return True, "loaded"

        if existing is None:
            return True, "already inactive"
        existing.pause()
        return True, "paused"

    async def start(self):
        """
        Unleash the wolf.
        Begin the hunt for opportunities.
        """
        self.running = True
        self._print_banner()

        # Record bot start in audit trail
        self.audit.record(
            AuditEventType.BOT_STARTED,
            payload={
                "mode": self.config.mode.value,
                "strategies": [s.strategy_id for s in self.strategies],
                "ai_enabled": self.config.ai_analysis_enabled,
            },
        )
        await self.event_bus.emit(
            bot_lifecycle_event(
                "started",
                {
                    "mode": self.config.mode.value,
                    "strategies": [s.strategy_id for s in self.strategies],
                },
            )
        )

        # Initialize async sessions
        await self.jupiter.initialize()
        await self.price_feed.initialize()
        await self.claude_brain.initialize()
        if self.jito:
            await self.jito.initialize()

        # Load real wallet balances so the pool can gate selection on funding
        # (no-op in simulation, where balances are seeded). Refreshed again each
        # scan cycle would keep them current; a live buy also re-checks on-chain.
        await self.wallet_pool.refresh_balances(self.solana_client)
        if self.wallet_pool.size > 1:
            self.logger.info(
                f"Wallet pool: {self.wallet_pool.size} wallets, "
                f"{self.wallet_pool.total_balance_sol():.3f} SOL total"
            )

        # Spin up the multi-agent pipeline workers before ingestion starts, so the
        # first detection has somewhere to go (no-op when disabled).
        if self.agent_pipeline is not None:
            await self.agent_pipeline.start()
            self.logger.info("Multi-agent decision pipeline: ON (Scanner/Sizing/Execution)")

        # Start monitoring and position management
        tasks = [
            asyncio.create_task(self.monitor.start_monitoring(self._on_token_launch)),
            asyncio.create_task(self._position_management_loop()),
        ]
        # Multi-tier market scanner (mid/large caps) — opt-in.
        if self.config.market_scanner_enabled:
            self._scanner_task = asyncio.create_task(
                self.scanner.start_scanning(self._on_candidate)
            )
            tasks.append(self._scanner_task)

        # Smart-money / whale wallet tracker — opt-in, needs a wallet list.
        if self.config.smart_money_enabled and self.config.smart_money_wallets:
            self._smart_money_task = asyncio.create_task(
                self.smart_money.start_tracking(self._on_candidate, self._on_smart_money_sell)
            )
            tasks.append(self._smart_money_task)

        # Multi-chain discovery scanner — opt-in, discovery-only (no execution).
        if self.config.discovery_enabled:
            from fenrir.discovery.providers.dexscreener import DexScreenerProvider
            from fenrir.discovery.scanner import DiscoveryScanner, build_adapters

            disc_cfg = self.config.build_discovery_config()
            self._discovery_dex = DexScreenerProvider()
            adapters = build_adapters(disc_cfg, self._discovery_dex, jupiter=self.jupiter)
            self.discovery_scanner = DiscoveryScanner(disc_cfg, adapters, event_bus=self.event_bus)
            self._discovery_task = asyncio.create_task(self.discovery_scanner.start_scanning())
            tasks.append(self._discovery_task)

        # Read-only arbitrage monitor — opt-in, surfaces cross-pool divergence, no execution.
        if self.config.arbitrage_monitor_enabled:
            from fenrir.discovery.providers.dexscreener import DexScreenerProvider
            from fenrir.trading.arbitrage_scanner import ArbitrageScanner

            self._arb_dex = DexScreenerProvider()
            self.arbitrage_monitor = self.config.build_arbitrage_monitor(
                event_bus=self.event_bus, logger=self.logger
            )
            self.arbitrage_scanner = ArbitrageScanner(
                monitor=self.arbitrage_monitor,
                fetch_pairs=self._arb_dex.fetch_pairs,
                token_source=self._arb_token_source,
                interval_seconds=self.config.arbitrage_interval_seconds,
                logger=self.logger,
            )
            self._arbitrage_task = asyncio.create_task(self.arbitrage_scanner.start_scanning())
            tasks.append(self._arbitrage_task)

        await asyncio.gather(*tasks)

    async def _arb_token_source(self) -> list[str]:
        """Tokens for the arbitrage monitor to check: currently-held positions plus
        recently-detected launches, de-duplicated. Read-only sampling."""
        held = list(self.positions.positions.keys())
        recent = list(self._recent_tokens)
        return list(dict.fromkeys(held + recent))

    async def _on_token_launch(self, token_data: dict):
        """
        Callback when new token is detected.

        v2: Routes through all active strategies. Each strategy
        independently decides whether to evaluate, and the budget
        tracker enforces spending limits.
        """
        token_addr = token_data["token_address"]
        symbol = token_data.get("symbol", "???")
        name = token_data.get("name", "Unknown")
        liq = token_data.get("initial_liquidity_sol", 0)
        mcap = token_data.get("market_cap_sol", 0)
        creator = token_data.get("creator")

        # Emit detection event
        await self.event_bus.emit(
            token_detected_event(
                token_address=token_addr,
                symbol=symbol,
                name=name,
                liquidity_sol=liq,
                market_cap_sol=mcap,
                creator=creator,
            )
        )
        # Feed the read-only arbitrage monitor's sampling ring (no-op when disabled).
        self._recent_tokens.append(token_addr)

        # Enrich + route. When the multi-agent pipeline is enabled, hand the fully
        # populated token_data to the ScannerAgent (queue-backed) and return
        # immediately — a slow AI eval or trade never stalls ingestion of the next
        # launch. Otherwise run the classic inline loop. Both call the same
        # _scan_and_route / _size_for_strategy / _execute_sized helpers, so the two
        # paths make identical decisions.
        if self.agent_pipeline is not None:
            await self.agent_pipeline.submit(token_data)
            return

        for strategy, signal_context in await self._scan_and_route(token_data):
            try:
                if signal_context is None:
                    await self._evaluate_and_execute(strategy, token_data)
                else:
                    await self._evaluate_and_execute(
                        strategy, token_data, signal_context=signal_context
                    )
            except Exception as e:
                await self.event_bus.emit(
                    error_event(
                        context=f"Strategy {strategy.strategy_id} evaluation",
                        error=str(e),
                        token_address=token_addr,
                        strategy_id=strategy.strategy_id,
                    )
                )

    async def _scan_and_route(self, token_data: dict) -> list[tuple[TradingStrategy, str | None]]:
        """Security/market pre-gates + enrichment + strategy routing.

        Returns the ``(strategy, signal_context)`` pairs that claim this launch — empty
        when a hard-gate rejects it or nothing claims it. Enriches ``token_data`` in
        place (RugCheck score, DexScreener momentum) for the AI context. Shared verbatim
        by the inline loop and the pipeline's ScannerAgent so both route identically.
        """
        token_addr = token_data["token_address"]
        symbol = token_data.get("symbol", "???")

        # ── Pre-trade security hard-gate (only when enabled) ──────────
        if self.security_filter is not None:
            lp_mint = token_data.get("lp_mint") or token_data.get("lp_mint_address")
            sec = await self.security_filter.check(token_addr, lp_mint)
            if not sec.passed:
                self.logger.info(f"Security filter rejected {symbol}: {sec}")
                return []
            # Surface RugCheck signals to the AI decision context.
            if "rugcheck_score" in sec.details:
                token_data["rugcheck_score"] = sec.details.get("rugcheck_score")
                token_data["rugcheck_risks"] = sec.details.get("rugcheck_risks")

        # ── Market data snapshot (data source + optional gate) ────────
        market_data = None
        if self.market_filter is not None:
            mkt = await self.market_filter.check(token_addr)
            market_data = mkt.market_data
            # Enforce the tier gate only when the operator turned it on.
            if self.config.market_filter_enabled and not mkt.passed:
                self.logger.info(f"Market filter rejected {symbol}: {mkt}")
                return []
            # Surface DexScreener momentum to the AI decision context.
            if market_data is not None:
                token_data["dex_volume_5m_usd"] = market_data.volume_5m_usd
                token_data["dex_txns_5m_buys"] = market_data.txns_5m_buys
                token_data["dex_txns_5m_sells"] = market_data.txns_5m_sells
                token_data["dex_buy_pressure_5m"] = market_data.buy_pressure_5m
                token_data["dex_price_change_1h_pct"] = market_data.price_change_1h_pct
                token_data["dex_liquidity_usd"] = market_data.liquidity_usd

        # Route through each active strategy.
        routed: list[tuple[TradingStrategy, str | None]] = []
        active_ids: list[str] = []
        claimed_signals: list[Any] = []  # bespoke signals, for confluence surfacing
        for strategy in self.strategies:
            if not strategy.state.active or strategy.state.paused:
                continue
            active_ids.append(strategy.strategy_id)

            try:
                if strategy.uses_market_data:
                    # Signal path: gate on the MarketData snapshot.
                    sig_strat = cast(SignalStrategy, strategy)
                    signal = sig_strat.evaluate_token(token_data, market_data)
                    if signal is None:
                        continue
                    claimed_signals.append(signal)
                    routed.append((strategy, sig_strat.build_ai_context(signal)))
                else:
                    # Classic path: cheap token_data pre-filter, then AI.
                    if not await strategy.should_evaluate(token_data):
                        continue
                    routed.append((strategy, None))

            except Exception as e:
                await self.event_bus.emit(
                    error_event(
                        context=f"Strategy {strategy.strategy_id} evaluation",
                        error=str(e),
                        token_address=token_addr,
                        strategy_id=strategy.strategy_id,
                    )
                )

        # No active strategy claimed this launch. Optionally fall back to the
        # always-on multi-lens AI Scout so the brain still evaluates it (fixes the
        # case where only a market-data strategy is active and fresh launches —
        # which have no market data yet — would otherwise never reach the AI).
        if not routed:
            if self.config.ai_evaluate_all_launches and await self._ai_scout.should_evaluate(
                token_data
            ):
                routed.append((self._ai_scout, None))
            else:
                self.logger.debug(
                    f"No active strategy evaluated ${symbol} "
                    f"(active: {', '.join(active_ids) or 'none'})"
                )

        # Read-only: surface when independent strategies agree on this token.
        if self.signal_aggregator is not None and claimed_signals:
            await self._surface_confluence(token_addr, symbol, claimed_signals)

        return routed

    async def _surface_confluence(self, token_addr: str, symbol: str, signals: list[Any]) -> None:
        """Normalize this scan's strategy signals into the aggregator and emit a
        SIGNAL_CONFLUENCE event when enough distinct strategies agree. Read-only —
        never changes sizing or execution."""
        from fenrir.signals import SignalDirection, normalize_strategy_signal
        from fenrir.signals.models import Signal as _Signal

        for sig in signals:
            try:
                self.signal_aggregator.add(normalize_strategy_signal(sig))
            except Exception as e:  # noqa: BLE001 - surfacing must never break routing
                self.logger.debug(f"confluence normalize skipped: {e}")

        confluence = self.signal_aggregator.confluence_for(token_addr, SignalDirection.LONG)
        if confluence is None:
            return
        if confluence.source_count >= self.config.signal_confluence_min_sources:
            top: _Signal | None = confluence.signals[0] if confluence.signals else None
            self.logger.info(
                f"Signal confluence ${symbol}: {confluence.source_count} agree "
                f"({', '.join(confluence.sources)}) conviction={confluence.combined_strength:.2f}"
            )
            await self.event_bus.emit(
                signal_confluence_event(
                    token_address=token_addr,
                    symbol=(top.symbol if top and top.symbol else symbol),
                    sources=confluence.sources,
                    combined_strength=confluence.combined_strength,
                    direction=confluence.direction.value,
                )
            )

    def _find_any_strategy(self, strategy_id: str) -> TradingStrategy | None:
        """Strategy lookup that also resolves the always-on AI Scout (which lives
        outside ``self.strategies`` but can claim launches via the fallback)."""
        if strategy_id == self._ai_scout.strategy_id:
            return self._ai_scout
        return self._find_strategy(strategy_id)

    async def _pipeline_claims(self, token_address: str, token_data: dict) -> list[str]:
        """ScannerAgent hook: route the launch, stash each strategy's signal_context on
        token_data so it reaches the sizing stage (token_data travels the events)."""
        routed = await self._scan_and_route(token_data)
        contexts: dict[str, str | None] = {}
        ids: list[str] = []
        for strategy, ctx in routed:
            ids.append(strategy.strategy_id)
            contexts[strategy.strategy_id] = ctx
        token_data["_signal_contexts"] = contexts
        return ids

    async def _pipeline_size(self, strategy_id: str, token_data: dict) -> float | None:
        """SizingAgent hook: run the shared sizing/risk gates for one claimed strategy."""
        strategy = self._find_any_strategy(strategy_id)
        if strategy is None:
            return None
        ctx = token_data.get("_signal_contexts", {}).get(strategy_id)
        return await self._size_for_strategy(strategy, token_data, signal_context=ctx)

    async def _pipeline_execute(
        self, token_data: dict, amount_sol: float, strategy_id: str
    ) -> bool:
        """ExecutionAgent hook: execute a sized buy via the shared executor."""
        strategy = self._find_any_strategy(strategy_id)
        if strategy is None:
            return False
        return await self._execute_sized(strategy, token_data, amount_sol)

    async def _evaluate_and_execute(
        self,
        strategy: TradingStrategy,
        token_data: dict,
        *,
        signal_context: str | None = None,
    ) -> None:
        """Evaluate a token for a specific strategy and execute if approved.

        ``signal_context`` overrides the strategy's static AI context with a
        per-signal context block (from a market-data-aware strategy's
        ``build_ai_context(signal)``); when None, the classic static context is
        used.

        Thin composition of the two shared phase-helpers ``_size_for_strategy`` (AI
        eval + geometry + budget + portfolio-risk gates) and ``_execute_sized``
        (execute + record). The multi-agent pipeline calls the exact same helpers, so
        the two paths make identical decisions — only their concurrency differs.
        """
        effective_amount = await self._size_for_strategy(
            strategy, token_data, signal_context=signal_context
        )
        if effective_amount is None:
            return
        await self._execute_sized(strategy, token_data, effective_amount)

    async def _size_for_strategy(
        self,
        strategy: TradingStrategy,
        token_data: dict,
        *,
        signal_context: str | None = None,
    ) -> float | None:
        """Run AI evaluation + geometry + budget + portfolio-risk gates for a strategy.

        Returns the effective buy amount in SOL when the trade clears every gate, or
        ``None`` when the AI declines or any gate blocks it (emitting the same decision
        / budget-exhausted events as before). Pure of execution side effects.
        """
        token_addr = token_data["token_address"]
        symbol = token_data.get("symbol", "???")

        # Get strategy-specific positions for context
        strategy_positions = self.positions.get_by_strategy(strategy.strategy_id)

        # Build historical context for AI
        historical_context = self.historical_memory.build_historical_context(
            creator_address=token_data.get("creator"),
            initial_liquidity_sol=token_data.get("initial_liquidity_sol", 0),
            market_cap_sol=token_data.get("market_cap_sol", 0),
        )

        # Get strategy-specific AI context (per-signal block if provided).
        base_strategy_context = (
            signal_context if signal_context is not None else strategy.get_ai_context()
        )

        # Run pre-entry market geometry analysis (informed pipeline pattern)
        # Derives token-specific TradeParams and enriches AI context
        geometry_report = self.market_geometry.analyze(token_data, strategy)
        strategy_context = base_strategy_context + "\n\n" + geometry_report.ai_context_block

        # AI evaluation (with strategy + geometry + historical context)
        should_buy, analysis, buy_amount_override = await self.claude_brain.evaluate_entry(
            token_data,
            strategy_positions,
            strategy_context=strategy_context,
            historical_context=historical_context,
        )

        # Emit AI decision event
        if analysis:
            await self.event_bus.emit(
                ai_decision_event(
                    token_address=token_addr,
                    symbol=symbol,
                    decision=analysis.decision.value,
                    confidence=analysis.confidence,
                    risk_score=analysis.risk_score,
                    reasoning=analysis.reasoning,
                    strategy_id=strategy.strategy_id,
                )
            )

        if not should_buy:
            return None

        # Use geometry-derived params if available, else fall back to strategy defaults
        params = (
            geometry_report.derived_params
            if geometry_report.derived_params
            else strategy.get_trade_params()
        )
        effective_amount = params.buy_amount_sol
        if buy_amount_override is not None:
            effective_amount = min(buy_amount_override, params.buy_amount_sol * 2)

        # Budget authorization gate
        auth = self.budget_tracker.authorize_trade(
            strategy_id=strategy.strategy_id,
            amount_sol=effective_amount,
            budget_sol=self.budget_tracker.effective_budget(
                strategy.strategy_id, strategy.budget_sol
            ),
            max_positions=strategy.max_concurrent_positions,
            is_active=strategy.state.active,
            is_paused=strategy.state.paused,
        )

        if not auth.allowed:
            await self.event_bus.emit(
                budget_exhausted_event(
                    strategy_id=strategy.strategy_id,
                    budget_sol=strategy.budget_sol,
                    spent_sol=self.budget_tracker.get_strategy_budget_status(
                        strategy.strategy_id, strategy.budget_sol
                    )["sol_spent"],
                )
            )
            return None

        # Portfolio-level risk gate — across ALL strategies (exposure cap,
        # creator/launch-window correlation, drawdown breaker).
        creator = token_data.get("creator")
        risk = self.portfolio_risk.check(strategy.strategy_id, effective_amount, creator=creator)
        if not risk.allowed:
            self.logger.info(f"Portfolio risk blocked {symbol}: {risk.reason}")
            return None

        return effective_amount

    async def _execute_sized(
        self,
        strategy: TradingStrategy,
        token_data: dict,
        effective_amount: float,
    ) -> bool:
        """Execute a sized buy and record it. Returns whether the buy succeeded.

        Runs only after ``_size_for_strategy`` has cleared every gate; shared by the
        inline loop and the ExecutionAgent so both record identically.
        """
        token_addr = token_data["token_address"]
        symbol = token_data.get("symbol", "???")
        creator = token_data.get("creator")

        # Execute the buy
        success = await self.trading_engine.execute_buy(
            token_data,
            amount_sol=effective_amount,
            strategy_id=strategy.strategy_id,
        )

        if success:
            self.budget_tracker.record_buy(strategy.strategy_id, effective_amount)
            self.portfolio_risk.record_open(
                token_addr, strategy.strategy_id, effective_amount, creator=creator
            )
            strategy.record_spend(effective_amount)

            await self.event_bus.emit(
                buy_executed_event(
                    token_address=token_addr,
                    symbol=symbol,
                    amount_sol=effective_amount,
                    entry_price=self._opened_entry_price(token_addr),
                    simulation=(self.config.mode == TradingMode.SIMULATION),
                    strategy_id=strategy.strategy_id,
                    wallet=self._opened_wallet(token_addr),
                )
            )
        else:
            await self.event_bus.emit(
                trade_failed_event(
                    token_address=token_addr,
                    symbol=symbol,
                    trade_type="BUY",
                    error="Execution failed",
                    strategy_id=strategy.strategy_id,
                )
            )

        return success

    async def apply_config_update(self, updates: dict) -> list[str]:
        """Apply a live config patch and perform the side-effects that a bare
        attribute set would miss, so dashboard toggles take effect without a
        restart. Returns config validation errors (empty list = OK).

        Live-effective fields:
        - market_scanner_enabled: starts/stops the scanner task on demand.
        - global_daily_sol_limit: re-applies the budget tracker's global cap.
        - buy_amount_sol: refreshes each strategy's cached trade params.
        - scanner tier thresholds / ai_min_confidence_to_buy / scanner_max_positions:
          read fresh each cycle, so a plain setattr is enough.
        """
        for key, value in updates.items():
            setattr(self.config, key, value)

        errors = self.config.validate()
        if errors:
            return errors

        if "global_daily_sol_limit" in updates:
            self.budget_tracker.set_global_limit(self.config.global_daily_sol_limit)

        if "buy_amount_sol" in updates:
            for strat in self.strategies:
                params = getattr(strat, "_params", None)
                if params is not None and hasattr(params, "buy_amount_sol"):
                    params.buy_amount_sol = self.config.buy_amount_sol

        if "market_scanner_enabled" in updates:
            if self.config.market_scanner_enabled:
                if self._scanner_task is None or self._scanner_task.done():
                    self._scanner_task = asyncio.create_task(
                        self.scanner.start_scanning(self._on_candidate)
                    )
                    self.logger.info("Market scanner started (live config update)")
            else:
                await self.scanner.stop()
                if self._scanner_task is not None:
                    self._scanner_task.cancel()
                    self._scanner_task = None
                self.logger.info("Market scanner stopped (live config update)")

        return []

    def _opened_entry_price(self, token_address: str) -> float:
        """Entry price of the position the engine just opened.

        The buy event used to read the price off the bonding curve (or hardcode
        0.0), so every established/AMM buy — which has no curve — reported an entry
        price of 0 to the event feed and dashboard while the position itself held
        the correct price. The engine already resolved it; just read it back.
        """
        position = self.positions.positions.get(token_address)
        return float(getattr(position, "entry_price", 0.0) or 0.0)

    def _opened_wallet(self, token_address: str) -> str | None:
        """Address of the pool wallet that executed this buy (for the audit chain)."""
        position = self.positions.positions.get(token_address)
        return (getattr(position, "wallet_address", "") or None) if position else None

    def _tier_context(self, token_data: dict) -> str:
        """Reframe the AI prompt for a scanned mid/large-cap so it isn't judged by
        launch-sniping heuristics (which treat high mcap as a 'FOMO trap')."""
        tier = token_data.get("tier")
        mcap = token_data.get("market_cap_usd") or 0
        liq = token_data.get("liquidity_usd") or 0
        holders = token_data.get("holder_count") or 0
        label = "ESTABLISHED LARGE-CAP" if tier == "large" else "MID-CAP (graduated off the curve)"
        return (
            f"MARKET-SCAN CANDIDATE — {label}. This is NOT a fresh pump.fun launch; it trades on "
            f"an AMM. Market cap ~${mcap:,.0f}, liquidity ~${liq:,.0f}, {holders:,} holders. "
            f"Evaluate it as a momentum/swing trade based on trend strength, liquidity and downside "
            f"risk — a high market cap is EXPECTED for this tier and is NOT itself a red flag or "
            f"'FOMO trap'. Judge on relative strength, not on launch-sniping heuristics."
        )

    def _smart_money_context(self, token_data: dict) -> str:
        """Frame the AI prompt for a token a tracked smart-money wallet just bought."""
        wallet = token_data.get("smart_money_wallet", "?")
        tier = token_data.get("smart_money_tier", "B")
        sol = token_data.get("smart_money_sol") or 0.0
        venue = "an AMM (migrated)" if token_data.get("migrated") else "the pump.fun bonding curve"
        tier_note = (
            "This is an A-TIER wallet (your highest-conviction list) — weight it very heavily."
            if tier == "A"
            else "This is a standard tracked wallet."
        )
        size_note = f" and spent ~{sol:.2f} SOL on it" if sol > 0 else ""
        return (
            f"SMART-MONEY SIGNAL — a wallet you track as a proven early buyer "
            f"({wallet[:6]}…{wallet[-4:] if len(wallet) > 10 else ''}) just BOUGHT this token on "
            f"{venue}{size_note}. {tier_note} Treat this as a STRONG positive signal (you follow "
            f"this wallet into early entries), but still assess rug/liquidity risk independently — "
            f"smart wallets take losers too. Weight the wallet's conviction and position size, not "
            f"launch-sniping heuristics."
        )

    async def _on_candidate(self, token_data: dict) -> None:
        """Evaluate a discovery candidate (scanner mid/large-cap OR smart-money buy).

        Parallel to _on_token_launch: reframes the AI for the source, gates on the
        per-source budget, and executes via the engine (which routes curve vs
        Jupiter automatically).
        """
        token_addr = token_data["token_address"]
        symbol = token_data.get("symbol", "???")
        if token_addr in self.positions.positions:
            return  # already holding

        await self.event_bus.emit(
            token_detected_event(
                token_address=token_addr,
                symbol=symbol,
                name=token_data.get("name", "Unknown"),
                liquidity_sol=0.0,
                market_cap_sol=0.0,
                creator=None,
            )
        )

        # Source-specific framing + budget: smart-money follows tracked wallets,
        # the scanner surfaces mid/large caps by market cap.
        if token_data.get("source") == "smart_money":
            strat_id = "smart_money"
            context = self._smart_money_context(token_data)
            budget = self.config.smart_money_daily_budget_sol or (self.config.buy_amount_sol * 5)
            max_positions = self.config.smart_money_max_positions
        else:
            strat_id = "scanner"
            context = self._tier_context(token_data)
            budget = self.config.scanner_daily_budget_sol or (self.config.buy_amount_sol * 5)
            max_positions = self.config.scanner_max_positions

        strategy_positions = self.positions.get_by_strategy(strat_id)
        should_buy, analysis, buy_amount_override = await self.claude_brain.evaluate_entry(
            token_data, strategy_positions, strategy_context=context
        )
        if analysis:
            await self.event_bus.emit(
                ai_decision_event(
                    token_address=token_addr,
                    symbol=symbol,
                    decision=analysis.decision.value,
                    confidence=analysis.confidence,
                    risk_score=analysis.risk_score,
                    reasoning=analysis.reasoning,
                    strategy_id=strat_id,
                )
            )
        if not should_buy:
            return

        amount = self.config.buy_amount_sol
        if buy_amount_override is not None and self.config.ai_dynamic_position_sizing:
            amount = min(buy_amount_override, self.config.buy_amount_sol * 2)
        # A-tier smart-money wallets get a size bump (capped at 2× base).
        if token_data.get("smart_money_tier") == "A":
            amount = min(
                amount * self.config.smart_money_a_tier_size_mult, self.config.buy_amount_sol * 2
            )

        auth = self.budget_tracker.authorize_trade(
            strategy_id=strat_id,
            amount_sol=amount,
            budget_sol=self.budget_tracker.effective_budget(strat_id, budget),
            max_positions=max_positions,
        )
        if not auth.allowed:
            self.logger.info(f"{strat_id} trade blocked for {symbol}: {auth.reason}")
            return

        creator = token_data.get("creator")
        risk = self.portfolio_risk.check(strat_id, amount, creator=creator)
        if not risk.allowed:
            self.logger.info(f"Portfolio risk blocked {symbol}: {risk.reason}")
            return

        success = await self.trading_engine.execute_buy(
            token_data, amount_sol=amount, strategy_id=strat_id
        )
        if success:
            self.budget_tracker.record_buy(strat_id, amount)
            self.portfolio_risk.record_open(token_addr, strat_id, amount, creator=creator)
            await self.event_bus.emit(
                buy_executed_event(
                    token_address=token_addr,
                    symbol=symbol,
                    amount_sol=amount,
                    entry_price=self._opened_entry_price(token_addr),
                    simulation=(self.config.mode == TradingMode.SIMULATION),
                    strategy_id=strat_id,
                    wallet=self._opened_wallet(token_addr),
                )
            )

    async def _on_smart_money_sell(self, mint: str, wallet: str) -> None:
        """A tracked wallet SOLD a token — if we hold it, route to the AI exit evaluator.

        We don't blindly dump: the sell is fed to the AI as a strong exit trigger,
        and the AI can still OVERRIDE_HOLD (e.g. strong momentum, other holders in).
        """
        position = self.positions.positions.get(mint)
        if not position:
            return  # we don't hold it — nothing to do
        self.logger.warning(
            f"Smart money {wallet[:6]}… SOLD {mint[:8]}… which we hold → AI exit check"
        )
        trigger = (
            f"A tracked smart-money wallet ({wallet[:6]}…) just SOLD/reduced this token — "
            f"a strong exit signal; decide whether to follow it out."
        )
        action, ai_reason = await self.claude_brain.evaluate_exit(
            mint, position, mechanical_trigger=trigger
        )
        if action == "OVERRIDE_HOLD":
            self.logger.info(f"AI holds {mint[:8]}… despite smart-money exit: {ai_reason}")
            return
        await self._execute_exit(mint, position, ai_reason or f"smart-money exit ({wallet[:6]}…)")

    async def _position_management_loop(self):
        """
        Continuous monitoring of open positions with AI-enhanced exits.

        Three phases each cycle:
        1. Fetch live prices
        2. Mechanical exit check -> AI override chance
        3. Proactive AI exit check (cadence-based)
        """
        while self.running:
            try:
                # Phase 1: Update prices
                open_addrs = list(self.positions.positions.keys())
                if open_addrs:
                    # Pre-migration, the pump.fun bonding curve is the
                    # authoritative price (and the only source for freshly
                    # launched tokens). Mark against it for BOTH sim and live so
                    # entry and current price share one scale — PnL then tracks
                    # the price ratio. Fall back to the external feed only for
                    # migrated/uncurved tokens (external feeds return wrong-scale
                    # or no data for fresh launches, which produced absurd PnL).
                    prices: dict[str, float] = {}
                    feed_addrs: list[str] = []
                    for addr in open_addrs:
                        curve = await self.trading_engine.fetch_curve_state(addr)
                        if curve and not curve.complete:
                            price = curve.get_price()
                            if price > 0:
                                prices[addr] = price
                                continue
                        feed_addrs.append(addr)

                    if feed_addrs:
                        results = await asyncio.gather(
                            *[self.price_feed.get_price(addr) for addr in feed_addrs],
                            return_exceptions=True,
                        )
                        for addr, result in zip(feed_addrs, results, strict=False):
                            if isinstance(result, BaseException) or not result:
                                continue
                            prices[addr] = result.price

                    for addr, new_price in prices.items():
                        pos = self.positions.positions.get(addr)
                        if not pos:
                            continue
                        pos.update_price(new_price)

                        # Ouroboros detection: dump → fake recovery → second dump
                        alert = self.dump_detector.update(addr, pos.current_price, pos.entry_price)
                        if alert.ouroboros_detected:
                            # Tighten trailing stop on the Position object
                            if hasattr(pos, "trailing_stop_override_pct"):
                                pos.trailing_stop_override_pct = alert.tightened_trailing_stop_pct
                            await self.event_bus.emit(
                                ouroboros_detected_event(
                                    addr,
                                    alert,
                                    symbol=pos.token_symbol,
                                    strategy_id=pos.strategy_id,
                                )
                            )

                # Phase 2: Mechanical exit conditions
                mechanical_exits = self.positions.check_exit_conditions()
                triggered_tokens = set()

                for token_address, reason in mechanical_exits:
                    triggered_tokens.add(token_address)
                    position = self.positions.positions.get(token_address)
                    if not position:
                        continue

                    action, ai_reason = await self.claude_brain.evaluate_exit(
                        token_address, position, mechanical_trigger=reason
                    )

                    if action == "OVERRIDE_HOLD":
                        await self.event_bus.emit(
                            ai_override_event(
                                token_address=token_address,
                                symbol=position.token_symbol,
                                mechanical_trigger=reason,
                                reasoning=ai_reason or "",
                                strategy_id=position.strategy_id,
                            )
                        )
                        continue

                    await self._execute_exit(token_address, position, ai_reason or reason)

                # Phase 3: Proactive AI exits
                for token_address in list(self.positions.positions.keys()):
                    if token_address in triggered_tokens:
                        continue
                    position = self.positions.positions.get(token_address)
                    if not position:
                        continue

                    action, ai_reason = await self.claude_brain.evaluate_exit(
                        token_address, position, mechanical_trigger=None
                    )

                    if action == "EXIT":
                        await self._execute_exit(token_address, position, f"AI: {ai_reason}")

                # Log portfolio status
                summary = self.positions.get_portfolio_summary()
                if summary["num_positions"] > 0:
                    self.logger.info(
                        f"Portfolio: {summary['num_positions']} positions | "
                        f"PnL: {summary['total_pnl_pct']:+.2f}% "
                        f"({summary['total_pnl_sol']:+.4f} SOL)"
                    )

                await asyncio.sleep(10)

            except Exception as e:
                self.logger.error("Position management error", e)
                await asyncio.sleep(10)

    async def _execute_exit(
        self,
        token_address: str,
        position: Position,
        reason: str,
    ) -> None:
        """Execute an exit and record outcomes across all systems."""
        pnl_pct = position.get_pnl_percent()
        pnl_sol = position.get_pnl_sol()
        hold_minutes = int((datetime.now() - position.entry_time).total_seconds() / 60)

        sold = await self.trading_engine.execute_sell(token_address, reason)
        if not sold:
            self.logger.warning(
                f"Sell failed for {token_address[:8]}... — skipping outcome recording"
            )
            return
        self.dump_detector.remove_position(token_address)

        # Record in AI session memory
        self.claude_brain.record_trade_outcome(
            token_address, pnl_pct, reason, hold_minutes, pnl_sol
        )

        # Record in historical memory (persists across sessions)
        self.historical_memory.record_outcome(
            token_address=token_address,
            token_symbol=position.token_symbol,
            creator_address=None,  # TODO: track creator on position
            ai_decision="BUY",
            was_bought=True,
            pnl_pct=pnl_pct,
            pnl_sol=pnl_sol,
            hold_time_minutes=hold_minutes,
            exit_reason=reason,
            strategy_id=position.strategy_id,
            session_id=self.audit.session_id,
        )

        # Update budget tracker
        returned_sol = position.amount_sol_invested + pnl_sol
        self.budget_tracker.record_sell(
            position.strategy_id,
            max(0, returned_sol),
            pnl_pct,
        )
        # Portfolio-level: free the exposure + fold realized PnL into the drawdown
        # breaker. Net-of-fees PnL, so the breaker tracks the wallet, not raw price.
        net_pnl_sol = position.get_net_pnl_sol()
        self.portfolio_risk.record_close(token_address, net_pnl_sol)
        if self.portfolio_risk.breaker_tripped:
            self.logger.warning(
                f"Portfolio drawdown breaker TRIPPED "
                f"(drawdown {self.portfolio_risk.drawdown_sol:.3f} SOL) — new buys halted"
            )

        # Update strategy state
        for strategy in self.strategies:
            if strategy.strategy_id == position.strategy_id:
                strategy.record_close(pnl_pct)
                break

        # Emit sell event
        await self.event_bus.emit(
            sell_executed_event(
                token_address=token_address,
                symbol=position.token_symbol,
                pnl_pct=pnl_pct,
                pnl_sol=pnl_sol,
                reason=reason,
                hold_minutes=hold_minutes,
                simulation=(self.config.mode == TradingMode.SIMULATION),
                strategy_id=position.strategy_id,
            )
        )

    async def stop(self):
        """Graceful shutdown."""
        self.running = False

        # Record bot stop
        self.audit.record(
            AuditEventType.BOT_STOPPED,
            payload={
                "budget_status": self.budget_tracker.get_global_status(),
                "strategies": [s.get_status() for s in self.strategies],
            },
        )
        await self.event_bus.emit(bot_lifecycle_event("stopped"))

        # Shutdown all components
        if self.agent_pipeline is not None:
            await self.agent_pipeline.stop()
        await self.monitor.stop()
        await self.scanner.stop()
        if self._scanner_task is not None:
            self._scanner_task.cancel()
            self._scanner_task = None
        await self.smart_money.stop()
        if self._smart_money_task is not None:
            self._smart_money_task.cancel()
            self._smart_money_task = None
        if self.discovery_scanner is not None:
            await self.discovery_scanner.stop()
        if self._discovery_task is not None:
            self._discovery_task.cancel()
            self._discovery_task = None
        if self._discovery_dex is not None:
            await self._discovery_dex.close()
            self._discovery_dex = None
        if self.arbitrage_scanner is not None:
            await self.arbitrage_scanner.stop()
        if self._arbitrage_task is not None:
            self._arbitrage_task.cancel()
            self._arbitrage_task = None
        if self._arb_dex is not None:
            await self._arb_dex.close()
            self._arb_dex = None
        await self.trading_engine.close()
        await self.claude_brain.close()
        await self.solana_client.close()
        await self.jupiter.close()
        await self.price_feed.close()
        if self.jito:
            await self.jito.close()
        if self.security_filter is not None:
            await self.security_filter.close()
        if self.market_filter is not None:
            await self.market_filter.close()

        # Shutdown new systems
        await self.event_bus.shutdown()
        self.historical_memory.close()
        self.audit.close()

        self.logger.info("FENRIR rests. Until next time.")

    def _print_banner(self):
        """Because every legend needs an introduction."""
        strategy_names = ", ".join(s.display_name for s in self.strategies)
        banner = """
=====================================================================
                     FENRIR TRADING BOT v2.0
          "In the memecoin wilderness, only the swift survive."
=====================================================================
  Mode: {mode:<58}
  Wallet: {wallet:<55}
  Buy Amount: {buy_sol:<51}
  Stop Loss: {stop_loss:<53}
  Take Profit: {take_profit:<50}
  Strategies: {strategies:<49}
  Audit Session: {session:<47}
=====================================================================
        """.format(
            mode=self.config.mode.value.upper(),
            wallet=self.wallet.get_address()[:40] + "...",
            buy_sol=f"{self.config.buy_amount_sol} SOL",
            stop_loss=f"{self.config.stop_loss_pct}%",
            take_profit=f"{self.config.take_profit_pct}%",
            strategies=strategy_names[:49],
            session=self.audit.session_id,
        )
        print(banner)

    # ── Public API for external control ─────────────────────────

    def get_full_status(self) -> dict:
        """Complete bot status including all new systems."""
        return {
            "running": self.running,
            "mode": self.config.mode.value,
            "portfolio": self.positions.get_portfolio_summary(),
            "strategies": [s.get_status() for s in self.strategies],
            "budget": self.budget_tracker.get_global_status(),
            "ai_brain": self.claude_brain.get_performance_report(),
            "audit": self.audit.get_chain_stats(),
            "event_bus": self.event_bus.get_stats(),
            "historical_memory": {
                "total_outcomes": self.historical_memory.get_total_outcomes(),
            },
            "ai_health": self.health_monitor.get_health_report(),
            "circuit_breakers": self.breakers.get_all_stats(),
        }

    def get_ai_health_report(self) -> dict:
        """Get detailed AI drift detection report."""
        return self.health_monitor.get_health_report()

    def reset_ai_health(self, strategy_id: str | None = None) -> None:
        """Reset AI health state after parameter changes."""
        self.health_monitor.reset(strategy_id)

    def verify_audit_chain(self) -> tuple[bool, int | None]:
        """Verify audit trail integrity."""
        return self.audit.verify_chain()


async def main():
    """
    The entry point.
    Where intention becomes reality.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="FENRIR Pump.fun Trading Bot v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in simulation mode with default sniper strategy
  python -m fenrir --mode simulation

  # Run with multiple strategies
  python -m fenrir --mode aggressive --strategies sniper graduation

  # Conservative mode with custom risk parameters
  python -m fenrir --mode conservative --stop-loss 15 --take-profit 200

Environment Variables (or use .env file):
  SOLANA_RPC_URL          - Your Solana RPC endpoint
  WALLET_PRIVATE_KEY      - Your wallet's base58-encoded private key
  OPENROUTER_API_KEY      - For AI-powered trading decisions
  TELEGRAM_BOT_TOKEN      - For Telegram alerts (optional)
  TELEGRAM_CHAT_ID        - Telegram chat to send alerts to (optional)
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "conservative", "aggressive", "degen"],
        default="simulation",
        help="Trading mode (default: simulation)",
    )
    parser.add_argument("--buy-amount", type=float, help="SOL amount per buy")
    parser.add_argument("--stop-loss", type=float, help="Stop loss percentage")
    parser.add_argument("--take-profit", type=float, help="Take profit percentage")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Active strategies (e.g., sniper graduation)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Launch live terminal dashboard",
    )

    args = parser.parse_args()

    # ── Single-instance guard ───────────────────────────────────
    # Prevent duplicate/orphaned bots (e.g. left over from an interrupted
    # restart) from trading concurrently against the same DB and doubling
    # RPC/AI load. Hold an exclusive loopback port as the lock; the OS frees
    # it automatically when this process exits, so stale locks can't linger.
    import socket as _socket
    import sys as _sys

    _lock_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    # SO_EXCLUSIVEADDRUSE is Windows-only; fetch via getattr so type checkers on
    # other platforms (e.g. CI on Linux) don't flag a missing socket attribute.
    _exclusive_opt = getattr(_socket, "SO_EXCLUSIVEADDRUSE", None)
    if _exclusive_opt is not None:
        _lock_sock.setsockopt(_socket.SOL_SOCKET, _exclusive_opt, 1)
    try:
        _lock_sock.bind(("127.0.0.1", 47653))
    except OSError:
        print(
            "FENRIR is already running (single-instance lock 127.0.0.1:47653). "
            "Stop the existing instance first, or check for an orphaned "
            "'python -m fenrir' process.",
            file=_sys.stderr,
        )
        return
    # Keep a reference alive for the whole process so the lock is held.
    globals()["_FENRIR_SINGLETON_LOCK"] = _lock_sock

    # Load config
    if args.config and Path(args.config).exists():
        config_path = Path(args.config)
        config_data = json.loads(config_path.read_text())
        config = BotConfig(**config_data)
    else:
        config = BotConfig()

    # Override with CLI args
    config.mode = TradingMode(args.mode)
    if args.buy_amount:
        config.buy_amount_sol = args.buy_amount
    if args.stop_loss:
        config.stop_loss_pct = args.stop_loss
    if args.take_profit:
        config.take_profit_pct = args.take_profit

    # Create bot with strategy selection
    bot = FenrirBot(config, strategies=args.strategies)

    # Optional: Launch terminal dashboard
    dashboard = None
    if args.dashboard:
        from fenrir.data.database import TradeDatabase
        from fenrir.ui.dashboard import Dashboard

        db = TradeDatabase()
        dashboard = Dashboard(bot, db=db)
        dashboard.start()

        import logging

        fenrir_logger = logging.getLogger("FENRIR")
        for handler in fenrir_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                fenrir_logger.removeHandler(handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        if dashboard:
            dashboard.stop()
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
