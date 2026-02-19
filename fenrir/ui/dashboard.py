#!/usr/bin/env python3
"""
FENRIR - Terminal Dashboard

Rich-based live terminal dashboard showing:
- Bot status and portfolio summary
- Open positions with real-time P&L
- Recent trade history
- Performance metrics

Runs in a background thread alongside the async bot loop.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from fenrir.bot import FenrirBot
    from fenrir.data.database import TradeDatabase


class Dashboard:
    """
    Terminal dashboard that runs alongside FenrirBot.

    Reads bot state from a background thread and renders
    a Rich Live display with portfolio, positions, trades, and P&L.
    """

    REFRESH_INTERVAL = 2.5  # seconds

    def __init__(
        self,
        bot: "FenrirBot",
        db: Optional["TradeDatabase"] = None,
    ):
        self.bot = bot
        self.db = db
        self.console = Console()
        self.start_time = datetime.now()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Launch the dashboard in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run_loop,
            name="fenrir-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Signal the dashboard thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        """Main loop: Rich Live display until stop is signaled."""
        with Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=0,
            screen=True,
        ) as live:
            while not self._stop_event.is_set():
                try:
                    live.update(self._build_layout())
                except Exception:
                    pass  # never crash the dashboard thread
                self._stop_event.wait(timeout=self.REFRESH_INTERVAL)

    # ── Layout ────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        """Compose the full dashboard layout from panels."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="portfolio", size=4),
            Layout(name="positions", ratio=2),
            Layout(name="trades", ratio=2),
            Layout(name="performance", size=5),
        )

        layout["header"].update(self._build_header_panel())
        layout["portfolio"].update(self._build_portfolio_panel())
        layout["positions"].update(self._build_positions_panel())
        layout["trades"].update(self._build_trades_panel())
        layout["performance"].update(self._build_performance_panel())

        return layout

    # ── Panels ────────────────────────────────────────────

    def _build_header_panel(self) -> Panel:
        """Top bar: mode, status, uptime."""
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split(".")[0]

        status = "RUNNING" if self.bot.running else "STOPPED"
        status_color = "green" if self.bot.running else "red"
        mode = self.bot.config.mode.value.upper()

        header = Text()
        header.append(f"  Mode: {mode}", style="bold cyan")
        header.append("  |  ")
        header.append(f"Status: {status}", style=f"bold {status_color}")
        header.append("  |  ")
        header.append(f"Uptime: {uptime_str}", style="bold white")

        return Panel(
            Align.center(header),
            title="[bold bright_white]FENRIR Dashboard[/]",
            border_style="bright_blue",
        )

    def _build_portfolio_panel(self) -> Panel:
        """Portfolio summary: positions count, invested, value, P&L."""
        try:
            summary = self.bot.positions.get_portfolio_summary()
        except Exception:
            summary = {
                "num_positions": 0, "total_invested_sol": 0,
                "current_value_sol": 0, "total_pnl_sol": 0, "total_pnl_pct": 0,
            }

        num = summary["num_positions"]
        invested = summary["total_invested_sol"]
        value = summary["current_value_sol"]
        pnl_sol = summary["total_pnl_sol"]
        pnl_pct = summary["total_pnl_pct"]
        pnl_color = "green" if pnl_sol >= 0 else "red"

        text = Text()
        text.append(f"  Positions: {num}", style="white")
        text.append(f"  |  Invested: {invested:.4f} SOL", style="white")
        text.append(f"  |  Value: {value:.4f} SOL", style="white")
        text.append(
            f"  |  P&L: {pnl_sol:+.4f} SOL ({pnl_pct:+.1f}%)",
            style=f"bold {pnl_color}",
        )

        return Panel(text, title="Portfolio Summary", border_style="cyan")

    def _build_positions_panel(self) -> Panel:
        """Table of open positions with live P&L."""
        # Snapshot to avoid dict mutation during iteration
        positions_snapshot = list(self.bot.positions.positions.items())

        if not positions_snapshot:
            return Panel(
                Align.center(Text("No open positions", style="dim italic")),
                title="Open Positions",
                border_style="cyan",
            )

        table = Table(
            expand=True,
            show_header=True,
            header_style="bold bright_cyan",
            border_style="dim",
        )
        table.add_column("Token", style="white", min_width=12)
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L %", justify="right")
        table.add_column("P&L SOL", justify="right")
        table.add_column("Age", justify="right")

        for token_addr, pos in positions_snapshot:
            try:
                pnl_pct = pos.get_pnl_percent()
                pnl_sol = pos.get_pnl_sol()
                pnl_color = "green" if pnl_pct >= 0 else "red"
                age = self._format_duration(datetime.now() - pos.entry_time)
                token_short = f"{token_addr[:4]}...{token_addr[-4:]}"

                table.add_row(
                    token_short,
                    f"{pos.entry_price:.8f}",
                    f"{pos.current_price:.8f}",
                    Text(f"{pnl_pct:+.1f}%", style=f"bold {pnl_color}"),
                    Text(f"{pnl_sol:+.4f}", style=pnl_color),
                    age,
                )
            except Exception:
                continue

        return Panel(table, title="Open Positions", border_style="cyan")

    def _build_trades_panel(self) -> Panel:
        """Recent trades from SQLite database."""
        if not self.db:
            return Panel(
                Align.center(
                    Text("Trade history unavailable (no database)", style="dim italic")
                ),
                title="Recent Trades",
                border_style="yellow",
            )

        try:
            recent_trades = self.db.get_recent_trades(limit=15)
        except Exception:
            recent_trades = []

        if not recent_trades:
            return Panel(
                Align.center(Text("No trades recorded yet", style="dim italic")),
                title="Recent Trades",
                border_style="yellow",
            )

        table = Table(
            expand=True,
            show_header=True,
            header_style="bold bright_yellow",
            border_style="dim",
        )
        table.add_column("Time", min_width=8)
        table.add_column("Type", min_width=4)
        table.add_column("Token", min_width=10)
        table.add_column("Amount SOL", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Notes", max_width=20)

        for trade in recent_trades:
            try:
                time_str = trade.timestamp.strftime("%H:%M:%S")
                type_color = "green" if trade.trade_type == "BUY" else "red"
                symbol = trade.token_symbol or f"{trade.token_mint[:4]}...{trade.token_mint[-4:]}"

                table.add_row(
                    time_str,
                    Text(trade.trade_type, style=f"bold {type_color}"),
                    symbol,
                    f"{trade.amount_sol:.4f}",
                    f"{trade.price_per_token:.8f}",
                    (trade.notes or "")[:20],
                )
            except Exception:
                continue

        return Panel(table, title="Recent Trades", border_style="yellow")

    def _build_performance_panel(self) -> Panel:
        """Performance metrics from database."""
        if not self.db:
            return Panel(
                Align.center(
                    Text("Performance data unavailable", style="dim italic")
                ),
                title="Performance",
                border_style="magenta",
            )

        try:
            perf = self.db.get_performance_summary()
        except Exception:
            perf = None

        if not perf or perf.get("total_trades", 0) == 0:
            return Panel(
                Align.center(
                    Text(
                        "No closed trades yet -- metrics will appear here",
                        style="dim italic",
                    )
                ),
                title="Performance",
                border_style="magenta",
            )

        win_rate = perf.get("win_rate", 0)
        sharpe = perf.get("sharpe_ratio", 0)
        largest_loss = perf.get("largest_loss_pct", 0)
        total_trades = perf.get("total_trades", 0)
        avg_hold = perf.get("avg_hold_time_minutes", 0)
        total_pnl = perf.get("total_pnl_sol", 0)
        profit_factor = perf.get("profit_factor", 0)
        pnl_color = "green" if total_pnl >= 0 else "red"

        text = Text()
        text.append(f"  Win Rate: {win_rate:.0f}%", style="white")
        text.append(f"  |  Sharpe: {sharpe:.2f}", style="white")
        text.append(f"  |  Max DD: {largest_loss:+.1f}%", style="red")
        text.append(f"  |  PF: {profit_factor:.1f}", style="white")
        text.append("\n")
        text.append(f"  Total Trades: {total_trades}", style="white")
        text.append(f"  |  Avg Hold: {avg_hold:.1f}m", style="white")
        text.append(
            f"  |  Total P&L: {total_pnl:+.4f} SOL",
            style=f"bold {pnl_color}",
        )

        return Panel(text, title="Performance", border_style="magenta")

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _format_duration(td: timedelta) -> str:
        """Format timedelta as short human-readable string."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m"
        else:
            hours = total_seconds // 3600
            mins = (total_seconds % 3600) // 60
            return f"{hours}h{mins}m"
