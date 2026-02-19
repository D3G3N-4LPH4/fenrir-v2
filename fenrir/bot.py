#!/usr/bin/env python3
"""
FENRIR Bot - The Orchestrator

Where all components harmonize into a single, elegant system.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from fenrir.ai.brain import ClaudeBrain
from fenrir.config import BotConfig, TradingMode
from fenrir.core.client import SolanaClient
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.core.positions import Position, PositionManager
from fenrir.core.wallet import WalletManager
from fenrir.data.price_feed import PriceFeedManager
from fenrir.logger import FenrirLogger
from fenrir.protocol.jito import JitoMEVProtection
from fenrir.trading.engine import TradingEngine
from fenrir.trading.monitor import PumpFunMonitor


class FenrirBot:
    """
    The conductor of this symphony.
    Where all components harmonize into a single, elegant system.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = FenrirLogger(config)

        # Validate configuration
        errors = config.validate()
        if errors:
            for error in errors:
                self.logger.error("Configuration error", Exception(error))
            raise ValueError("Invalid configuration")

        # Initialize core components
        self.wallet = WalletManager(
            config.private_key, simulation_mode=(config.mode == TradingMode.SIMULATION)
        )
        self.solana_client = SolanaClient(config, self.logger)
        self.jupiter = JupiterSwapEngine(config, self.logger)
        self.positions = PositionManager(config, self.logger)

        # Initialize Jito MEV protection (optional)
        self.jito: JitoMEVProtection | None = None
        if config.use_jito:
            self.jito = JitoMEVProtection(
                region="mainnet",
                tip_lamports=config.jito_tip_lamports,
            )

        # Initialize price feed manager
        self.price_feed = PriceFeedManager()

        # Initialize trading engine with all dependencies
        self.trading_engine = TradingEngine(
            config,
            self.wallet,
            self.solana_client,
            self.jupiter,
            self.positions,
            self.logger,
            jito=self.jito,
        )

        # Initialize monitor with real launch detection
        self.monitor = PumpFunMonitor(config, self.solana_client, self.logger)

        # Initialize AI Brain (autonomous Claude decision engine)
        self.claude_brain = ClaudeBrain(config, self.logger)

        self.running = False

    async def start(self):
        """
        Unleash the wolf.
        Begin the hunt for opportunities.
        """
        self.running = True

        # Print startup banner
        self._print_banner()

        # Initialize async sessions
        await self.jupiter.initialize()
        await self.price_feed.initialize()
        await self.claude_brain.initialize()
        if self.jito:
            await self.jito.initialize()

        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor.start_monitoring(self._on_token_launch))

        # Start position management loop
        management_task = asyncio.create_task(self._position_management_loop())

        # Wait for tasks (KeyboardInterrupt is handled by the outer asyncio.run
        # in main(); catching it here would not work reliably inside gather)
        await asyncio.gather(monitor_task, management_task)

    async def _on_token_launch(self, token_data: dict):
        """
        Callback when new token is detected and passes criteria.
        If AI Brain is enabled, Claude evaluates before buying.
        """
        self.logger.launch_detected(
            token_data["token_address"], token_data["initial_liquidity_sol"]
        )
        self.logger.info(
            f"   Token: {token_data.get('name', 'Unknown')} " f"({token_data.get('symbol', '???')})"
        )

        # AI evaluation gate
        should_buy, analysis, buy_amount_override = await self.claude_brain.evaluate_entry(
            token_data,
            self.positions.positions,
        )

        if not should_buy:
            reason = analysis.reasoning[:80] if analysis else "AI disabled fallback"
            self.logger.info(f"   SKIP: {reason}")
            return

        # Determine effective buy amount (pass explicitly instead of mutating config)
        effective_amount = self.config.buy_amount_sol
        if buy_amount_override is not None:
            effective_amount = min(buy_amount_override, self.config.buy_amount_sol * 2)

        await self.trading_engine.execute_buy(token_data, amount_sol=effective_amount)

    async def _position_management_loop(self):
        """
        Continuous monitoring of open positions with AI-enhanced exits.

        Three phases each cycle:
        1. Fetch live prices (unchanged)
        2. Mechanical exit check -> AI gets override chance before executing
        3. Proactive AI exit check (cadence-based, for non-triggered positions)
        """
        while self.running:
            try:
                # Phase 1: Update prices for all open positions (concurrent)
                open_addrs = list(self.positions.positions.keys())
                if open_addrs:
                    results = await asyncio.gather(
                        *[self.price_feed.get_price(addr) for addr in open_addrs],
                        return_exceptions=True,
                    )
                    for addr, result in zip(open_addrs, results, strict=False):
                        if isinstance(result, Exception):
                            self.logger.debug(f"Price fetch failed for {addr[:8]}: {result}")
                        elif result:
                            pos = self.positions.positions.get(addr)
                            if pos:
                                pos.update_price(result.price)

                # Phase 2: Check mechanical exit conditions
                mechanical_exits = self.positions.check_exit_conditions()
                triggered_tokens = set()

                for token_address, reason in mechanical_exits:
                    triggered_tokens.add(token_address)
                    position = self.positions.positions.get(token_address)
                    if not position:
                        continue

                    # Give AI a chance to override the mechanical trigger
                    action, ai_reason = await self.claude_brain.evaluate_exit(
                        token_address, position, mechanical_trigger=reason
                    )

                    if action == "OVERRIDE_HOLD":
                        continue

                    await self._execute_exit(token_address, position, ai_reason or reason)

                # Phase 3: Proactive AI exit evaluation (cadence-based)
                for token_address in list(self.positions.positions.keys()):
                    if token_address in triggered_tokens:
                        continue  # Already handled above
                    position = self.positions.positions.get(token_address)
                    if not position:
                        continue

                    action, ai_reason = await self.claude_brain.evaluate_exit(
                        token_address, position, mechanical_trigger=None
                    )

                    if action == "EXIT":
                        self.logger.info(
                            f"AI proactive exit: {token_address[:8]}... " f"- {ai_reason}"
                        )
                        await self._execute_exit(token_address, position, f"AI: {ai_reason}")

                # Log portfolio status periodically
                summary = self.positions.get_portfolio_summary()
                if summary["num_positions"] > 0:
                    self.logger.info(
                        f"Portfolio: {summary['num_positions']} positions | "
                        f"PnL: {summary['total_pnl_pct']:+.2f}% "
                        f"({summary['total_pnl_sol']:+.4f} SOL)"
                    )

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error("Position management error", e)
                await asyncio.sleep(10)

    async def _execute_exit(
        self,
        token_address: str,
        position: Position,
        reason: str,
    ) -> None:
        """Execute an exit and record the outcome in AI memory."""
        pnl_pct = position.get_pnl_percent()
        hold_minutes = int((datetime.now() - position.entry_time).total_seconds() / 60)
        pnl_sol = position.get_pnl_sol()

        await self.trading_engine.execute_sell(token_address, reason)

        self.claude_brain.record_trade_outcome(
            token_address,
            pnl_pct,
            reason,
            hold_minutes,
            pnl_sol,
        )

    async def stop(self):
        """Graceful shutdown."""
        self.running = False
        await self.monitor.stop()
        await self.claude_brain.close()
        await self.solana_client.close()
        await self.jupiter.close()
        await self.price_feed.close()
        if self.jito:
            await self.jito.close()
        self.logger.info("FENRIR rests. Until next time.")

    def _print_banner(self):
        """Because every legend needs an introduction."""
        banner = """
=====================================================================
                     FENRIR TRADING BOT v1.0
          "In the memecoin wilderness, only the swift survive."
=====================================================================
  Mode: {mode:<58}
  Wallet: {wallet:<55}
  Buy Amount: {buy_sol:<51}
  Stop Loss: {stop_loss:<53}
  Take Profit: {take_profit:<50}
=====================================================================
        """.format(
            mode=self.config.mode.value.upper(),
            wallet=self.wallet.get_address()[:40] + "...",
            buy_sol=f"{self.config.buy_amount_sol} SOL",
            stop_loss=f"{self.config.stop_loss_pct}%",
            take_profit=f"{self.config.take_profit_pct}%",
        )
        print(banner)


async def main():
    """
    The entry point.
    Where intention becomes reality.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="FENRIR Pump.fun Trading Bot - Elegant DeFi Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in simulation mode (safe testing)
  python -m fenrir --mode simulation

  # Run live with custom buy amount
  python -m fenrir --mode conservative --buy-amount 0.05

  # Aggressive mode with custom risk parameters
  python -m fenrir --mode aggressive --stop-loss 15 --take-profit 200

Environment Variables (or use .env file):
  SOLANA_RPC_URL          - Your Solana RPC endpoint (QuickNode, Helius, etc.)
  WALLET_PRIVATE_KEY      - Your wallet's base58-encoded private key
  OPENROUTER_API_KEY      - Optional: for AI-powered trading decisions
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "conservative", "aggressive", "degen"],
        default="simulation",
        help="Trading mode (default: simulation)",
    )
    parser.add_argument("--buy-amount", type=float, help="SOL amount per buy (default: 0.1)")
    parser.add_argument("--stop-loss", type=float, help="Stop loss percentage (default: 25)")
    parser.add_argument("--take-profit", type=float, help="Take profit percentage (default: 100)")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--dashboard", action="store_true", default=False, help="Launch live terminal dashboard"
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

    # Create and run bot
    bot = FenrirBot(config)

    # Optional: Launch terminal dashboard
    dashboard = None
    if args.dashboard:
        from fenrir.data.database import TradeDatabase
        from fenrir.ui.dashboard import Dashboard

        db = TradeDatabase()
        dashboard = Dashboard(bot, db=db)
        dashboard.start()

        # Suppress console log handler so it doesn't corrupt the Rich display
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
