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
from datetime import datetime
from pathlib import Path

from fenrir.ai.brain import ClaudeBrain
from fenrir.config import BotConfig, TradingMode
from fenrir.core.budget import BudgetTracker
from fenrir.core.client import SolanaClient
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.core.positions import Position, PositionManager
from fenrir.core.wallet import WalletManager
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
    token_detected_event,
    trade_failed_event,
)
from fenrir.logger import FenrirLogger
from fenrir.protocol.jito import JitoMEVProtection
from fenrir.strategies import STRATEGY_REGISTRY, SniperStrategy
from fenrir.strategies.base import TradingStrategy
from fenrir.trading.engine import TradingEngine
from fenrir.trading.monitor import PumpFunMonitor


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

        # ── Core infrastructure ─────────────────────────────────
        self.wallet = WalletManager(
            config.private_key, simulation_mode=(config.mode == TradingMode.SIMULATION)
        )
        self.solana_client = SolanaClient(config, self.logger)
        self.jupiter = JupiterSwapEngine(config, self.logger)
        self.positions = PositionManager(config, self.logger)

        # Jito MEV protection (optional)
        self.jito: JitoMEVProtection | None = None
        if config.use_jito:
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
        )

        # Monitor
        self.monitor = PumpFunMonitor(config, self.solana_client, self.logger)

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
        self.budget_tracker = BudgetTracker()

        # ── NEW: Strategy System ────────────────────────────────
        self.strategies: list[TradingStrategy] = []
        self._init_strategies(strategies)

        # ── AI Brain (with historical memory) ───────────────────
        self.claude_brain = ClaudeBrain(config, self.logger)

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
                )
            )

    def _init_strategies(self, strategy_ids: list[str] | None) -> None:
        """Initialize trading strategies."""
        if not strategy_ids:
            # Default: just the sniper strategy matching current config
            strategy_ids = ["sniper"]

        for sid in strategy_ids:
            cls = STRATEGY_REGISTRY.get(sid)
            if cls:
                strategy = cls(self.config)
                self.strategies.append(strategy)
                self.logger.info(
                    f"Strategy loaded: {strategy.display_name} ({strategy.strategy_id})"
                )
            else:
                self.logger.warning(
                    f"Unknown strategy: {sid} "
                    f"(available: {list(STRATEGY_REGISTRY.keys())})"
                )

        if not self.strategies:
            # Fallback: always have at least the sniper
            self.strategies.append(SniperStrategy(self.config))
            self.logger.info("Fallback: loaded default Sniper strategy")

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
        await self.event_bus.emit(bot_lifecycle_event("started", {
            "mode": self.config.mode.value,
            "strategies": [s.strategy_id for s in self.strategies],
        }))

        # Initialize async sessions
        await self.jupiter.initialize()
        await self.price_feed.initialize()
        await self.claude_brain.initialize()
        if self.jito:
            await self.jito.initialize()

        # Start monitoring and position management
        monitor_task = asyncio.create_task(
            self.monitor.start_monitoring(self._on_token_launch)
        )
        management_task = asyncio.create_task(self._position_management_loop())

        await asyncio.gather(monitor_task, management_task)

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
        await self.event_bus.emit(token_detected_event(
            token_address=token_addr,
            symbol=symbol,
            name=name,
            liquidity_sol=liq,
            market_cap_sol=mcap,
            creator=creator,
        ))

        # Route through each active strategy
        for strategy in self.strategies:
            if not strategy.state.active or strategy.state.paused:
                continue

            try:
                if not await strategy.should_evaluate(token_data):
                    continue

                await self._evaluate_and_execute(strategy, token_data)

            except Exception as e:
                await self.event_bus.emit(error_event(
                    context=f"Strategy {strategy.strategy_id} evaluation",
                    error=str(e),
                    token_address=token_addr,
                    strategy_id=strategy.strategy_id,
                ))

    async def _evaluate_and_execute(
        self,
        strategy: TradingStrategy,
        token_data: dict,
    ) -> None:
        """Evaluate a token for a specific strategy and execute if approved."""
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

        # Get strategy-specific AI context
        strategy_context = strategy.get_ai_context()

        # AI evaluation (with strategy + historical context)
        should_buy, analysis, buy_amount_override = await self.claude_brain.evaluate_entry(
            token_data,
            strategy_positions,
            strategy_context=strategy_context,
            historical_context=historical_context,
        )

        # Emit AI decision event
        if analysis:
            await self.event_bus.emit(ai_decision_event(
                token_address=token_addr,
                symbol=symbol,
                decision=analysis.decision.value,
                confidence=analysis.confidence,
                risk_score=analysis.risk_score,
                reasoning=analysis.reasoning,
                strategy_id=strategy.strategy_id,
            ))

        if not should_buy:
            return

        # Determine trade parameters
        params = strategy.get_trade_params()
        effective_amount = params.buy_amount_sol
        if buy_amount_override is not None:
            effective_amount = min(buy_amount_override, params.buy_amount_sol * 2)

        # Budget authorization gate
        auth = self.budget_tracker.authorize_trade(
            strategy_id=strategy.strategy_id,
            amount_sol=effective_amount,
            budget_sol=strategy.budget_sol,
            max_positions=strategy.max_concurrent_positions,
            is_active=strategy.state.active,
            is_paused=strategy.state.paused,
        )

        if not auth.allowed:
            await self.event_bus.emit(budget_exhausted_event(
                strategy_id=strategy.strategy_id,
                budget_sol=strategy.budget_sol,
                spent_sol=self.budget_tracker._get_state(strategy.strategy_id).sol_spent,
            ))
            return

        # Execute the buy
        success = await self.trading_engine.execute_buy(
            token_data,
            amount_sol=effective_amount,
            strategy_id=strategy.strategy_id,
        )

        if success:
            self.budget_tracker.record_buy(strategy.strategy_id, effective_amount)
            strategy.record_spend(effective_amount)

            # Get entry price from bonding curve state if available
            entry_price = 0.0
            bc = token_data.get("bonding_curve_state")
            if bc and hasattr(bc, "get_price"):
                entry_price = bc.get_price()

            await self.event_bus.emit(buy_executed_event(
                token_address=token_addr,
                symbol=symbol,
                amount_sol=effective_amount,
                entry_price=entry_price,
                simulation=(self.config.mode == TradingMode.SIMULATION),
                strategy_id=strategy.strategy_id,
            ))
        else:
            await self.event_bus.emit(trade_failed_event(
                token_address=token_addr,
                symbol=symbol,
                trade_type="BUY",
                error="Execution failed",
                strategy_id=strategy.strategy_id,
            ))

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
                    results = await asyncio.gather(
                        *[self.price_feed.get_price(addr) for addr in open_addrs],
                        return_exceptions=True,
                    )
                    for addr, result in zip(open_addrs, results, strict=False):
                        if isinstance(result, Exception):
                            continue
                        elif result:
                            pos = self.positions.positions.get(addr)
                            if pos:
                                pos.update_price(result.price)

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
                        await self.event_bus.emit(ai_override_event(
                            token_address=token_address,
                            symbol=position.token_symbol,
                            mechanical_trigger=reason,
                            reasoning=ai_reason or "",
                            strategy_id=position.strategy_id,
                        ))
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
                        await self._execute_exit(
                            token_address, position, f"AI: {ai_reason}"
                        )

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
        hold_minutes = int(
            (datetime.now() - position.entry_time).total_seconds() / 60
        )

        await self.trading_engine.execute_sell(token_address, reason)

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

        # Update strategy state
        for strategy in self.strategies:
            if strategy.strategy_id == position.strategy_id:
                strategy.record_close(pnl_pct)
                break

        # Emit sell event
        await self.event_bus.emit(sell_executed_event(
            token_address=token_address,
            symbol=position.token_symbol,
            pnl_pct=pnl_pct,
            pnl_sol=pnl_sol,
            reason=reason,
            hold_minutes=hold_minutes,
            simulation=(self.config.mode == TradingMode.SIMULATION),
            strategy_id=position.strategy_id,
        ))

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
        await self.monitor.stop()
        await self.claude_brain.close()
        await self.solana_client.close()
        await self.jupiter.close()
        await self.price_feed.close()
        if self.jito:
            await self.jito.close()

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
