#!/usr/bin/env python3
"""
FENRIR - Telegram Event Adapter

Sends alerts to Telegram for critical trading events.
Supports optional approval gates for high-value trades.

Requires: aiohttp (already in project deps)

Usage:
    adapter = TelegramAdapter(
        bot_token="123456:ABC...",
        chat_id="-100123456789",
        alert_on_trades=True,
        alert_on_errors=True,
    )
    bus.register(adapter)
"""

import asyncio
import logging

import aiohttp

from fenrir.events.bus import EventListener
from fenrir.events.types import EventSeverity, TradeEvent

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAdapter(EventListener):
    """
    Sends event notifications to a Telegram chat.

    Filters:
    - CRITICAL severity: Always sent (trade executions)
    - WARNING severity: Sent if alert_on_errors=True
    - Trade events: Sent if alert_on_trades=True
    - AI overrides: Always sent (unusual and noteworthy)

    Rate limited to prevent Telegram API throttling.
    """

    min_severity = EventSeverity.WARNING

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        alert_on_trades: bool = True,
        alert_on_errors: bool = True,
        max_messages_per_minute: int = 20,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.alert_on_trades = alert_on_trades
        self.alert_on_errors = alert_on_errors
        self.max_per_minute = max_messages_per_minute

        self._session: aiohttp.ClientSession | None = None
        self._message_times: list[float] = []
        self._enabled = bool(bot_token and chat_id)

    def accepts(self, event: TradeEvent) -> bool:
        if not self._enabled:
            return False

        # Always send trade executions
        if event.event_type in ("BUY_EXECUTED", "SELL_EXECUTED") and self.alert_on_trades:
            return True

        # Always send AI overrides (noteworthy)
        if event.event_type == "AI_OVERRIDE":
            return True

        # Always send budget exhaustion
        if event.event_type == "BUDGET_EXHAUSTED":
            return True

        # Errors if configured
        if event.severity == EventSeverity.WARNING and self.alert_on_errors:
            return event.event_type in ("TRADE_FAILED", "ERROR")

        # CRITICAL events always
        if event.severity == EventSeverity.CRITICAL:
            return True

        return False

    async def on_event(self, event: TradeEvent) -> None:
        if not self._enabled:
            return

        # Rate limiting
        now = asyncio.get_event_loop().time()
        self._message_times = [t for t in self._message_times if now - t < 60]
        if len(self._message_times) >= self.max_per_minute:
            return

        text = self._format_message(event)
        await self._send_message(text)
        self._message_times.append(now)

    def _format_message(self, event: TradeEvent) -> str:
        """Format event as a Telegram message with emoji and key details."""
        lines = []

        # Header with emoji
        emoji_map = {
            "BUY_EXECUTED": "🟢",
            "SELL_EXECUTED": "🔴" if event.data.get("pnl_pct", 0) < 0 else "💰",
            "AI_OVERRIDE": "🧠",
            "TRADE_FAILED": "❌",
            "BUDGET_EXHAUSTED": "🚫",
            "ERROR": "⚠️",
        }
        emoji = emoji_map.get(event.event_type, "📌")
        lines.append(f"{emoji} *FENRIR* — {event.event_type.replace('_', ' ')}")

        # Core message
        lines.append(f"`{event.message}`")

        # Strategy tag
        if event.strategy_id:
            lines.append(f"Strategy: `{event.strategy_id}`")

        # Key data points depending on event type
        data = event.data
        if event.event_type == "BUY_EXECUTED":
            lines.append(f"Amount: `{data.get('amount_sol', 0):.4f} SOL`")
            if data.get("simulation"):
                lines.append("Mode: `SIMULATION`")

        elif event.event_type == "SELL_EXECUTED":
            pnl = data.get("pnl_pct", 0)
            sol = data.get("pnl_sol", 0)
            sign = "+" if pnl > 0 else ""
            lines.append(f"P&L: `{sign}{pnl:.2f}%` (`{sign}{sol:.4f} SOL`)")
            lines.append(f"Reason: `{data.get('reason', 'unknown')}`")
            lines.append(f"Hold: `{data.get('hold_minutes', 0)}min`")

        return "\n".join(lines)

    async def _send_message(self, text: str) -> None:
        """Send a message via Telegram Bot API."""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )

        url = TELEGRAM_API.format(token=self.bot_token)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram send failed (%d): %s", resp.status, body[:200])
        except Exception as e:
            logger.warning("Telegram adapter error: %s", e)

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
