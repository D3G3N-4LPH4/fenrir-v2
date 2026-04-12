#!/usr/bin/env python3
"""
FENRIR - Telegram Event Adapters

Two adapters in one file:

TelegramAdapter (v1)
    Simple EventListener-based adapter used by bot.py.
    Sends critical events to Telegram with basic formatting.

TelegramAdapterV2
    Production-grade adapter with tiered alert system (FLASH/PRIORITY/ROUTINE),
    semantic dedup, per-tier rate limits, command registry, and EventBus wiring.
    Port of Crucix's telegram.mjs.
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import aiohttp

from fenrir.events.bus import EventListener
from fenrir.events.types import EventSeverity, TradeEvent
from fenrir.events.alert_evaluator import (
    TIER_CONFIG,
    AlertEvaluation,
    LLMEvaluationPrompt,
    RuleBasedEvaluator,
    Signal,
    llm_response_to_evaluation,
    parse_llm_json,
)

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_MAX_TEXT = 4096

CommandHandler = Callable[[str, int | None], Coroutine[Any, Any, str | None]]

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


# ─── Internal records for v2 ──────────────────────────────────────────────────

@dataclass
class _AlertRecord:
    tier: str
    timestamp: float  # epoch seconds
    headline: str


# ─── TelegramAdapterV2 ────────────────────────────────────────────────────────

class TelegramAdapterV2:
    """
    Production Telegram adapter for FENRIR v2.

    Features vs v1:
      - FLASH/PRIORITY/ROUTINE tier system with per-tier rate limits + cooldowns
      - Dual-layer semantic dedup (signal-level + alert-level, 4h window)
      - Command registry pattern (external handlers via on_command())
      - Safe long-message chunking at newlines (Telegram 4096 char limit)
      - @botname suffix stripping for group chat compatibility
      - Triple-fallback JSON parsing for LLM responses
      - Mute/unmute with configurable duration
      - EventBus wiring: subscribes to ALERT_REQUESTED, SIGNAL_BATCH, POSITION_EVENT

    Usage:
        adapter = TelegramAdapterV2(
            bot_token=config["telegram"]["bot_token"],
            chat_id=config["telegram"]["chat_id"],
            event_bus=bus,
            llm_client=claude_client,   # optional
        )
        adapter.on_command("/status",   status_handler)
        adapter.on_command("/sweep",    sweep_handler)
        await adapter.start()
    """

    BUILTIN_COMMANDS = {
        "/help":   "Show available commands",
        "/mute":   "Mute alerts for 1h (or /mute 2h)",
        "/unmute": "Resume alerts",
        "/alerts": "Show recent alert history",
    }

    KNOWN_COMMANDS = {
        "/status":    "System health, last sweep time, source status",
        "/sweep":     "Trigger a manual scoring sweep",
        "/brief":     "Compact summary of the latest intelligence",
        "/positions": "Current open positions and P&L",
        "/budget":    "Budget tracker status and exposure",
        **BUILTIN_COMMANDS,
    }

    def __init__(
        self,
        bot_token: str,
        chat_id: str | int,
        event_bus: Any,
        llm_client: Any | None = None,
        poll_interval_s: float = 5.0,
        dedup_window_h: float = 4.0,
        dedup_prune_h: float = 24.0,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.event_bus = event_bus
        self.llm_client = llm_client
        self.poll_interval_s = poll_interval_s
        self.dedup_window_s = dedup_window_h * 3600
        self.dedup_prune_s = dedup_prune_h * 3600

        self._alert_history: list[_AlertRecord] = []
        self._content_hashes: dict[str, float] = {}
        self._mute_until: float | None = None
        self._last_update_id: int = 0
        self._bot_username: str | None = None
        self._command_handlers: dict[str, CommandHandler] = {}
        self._rule_evaluator = RuleBasedEvaluator()
        self._session: aiohttp.ClientSession | None = None
        self._polling_task: asyncio.Task | None = None
        self._running = False

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self.is_configured:
            logger.warning("[Telegram v2] Not configured — bot_token or chat_id missing")
            return

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
        )
        self._running = True

        try:
            await self._load_bot_identity()
            await self._register_bot_commands()
        except Exception as exc:
            logger.error("[Telegram v2] Startup error: %s", exc)

        self._wire_event_bus()
        self._polling_task = asyncio.create_task(self._poll_loop(), name="telegram_v2_poll")
        logger.info("[Telegram v2] Started (bot: @%s, chat: %s)", self._bot_username, self.chat_id)

    async def stop(self) -> None:
        self._running = False
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        logger.info("[Telegram v2] Stopped")

    # ─── EventBus Wiring ─────────────────────────────────────────────────────

    def _wire_event_bus(self) -> None:
        bus = self.event_bus
        bus.subscribe("ALERT_REQUESTED",  self._on_alert_requested)
        bus.subscribe("POSITION_OPENED",  self._on_position_opened)
        bus.subscribe("POSITION_CLOSED",  self._on_position_closed)
        bus.subscribe("BUDGET_ALERT",     self._on_budget_alert)
        bus.subscribe("SYSTEM_HEALTH",    self._on_system_health)

    async def _on_alert_requested(self, payload: dict) -> None:
        try:
            signals = [self._dict_to_signal(s) for s in payload.get("signals", [])]
            direction = payload.get("delta_direction", "mixed")
            token = payload.get("token_address")
            await self.evaluate_and_alert(signals, direction, token)
        except Exception as exc:
            logger.error("[Telegram v2] ALERT_REQUESTED error: %s", exc, exc_info=True)

    async def _on_position_opened(self, payload: dict) -> None:
        token = payload.get("token", "unknown")
        entry = payload.get("entry_price", 0)
        size = payload.get("size", 0)
        score = payload.get("score", 0)
        msg = (
            f"🟢 *Position Opened*\n\n"
            f"`{token}`\n"
            f"Entry: `${entry:.6f}` | Size: `${size:.2f}` | Score: `{score:.1f}`"
        )
        await self.send_message(msg)

    async def _on_position_closed(self, payload: dict) -> None:
        token = payload.get("token", "unknown")
        pnl = payload.get("pnl", 0)
        reason = payload.get("reason", "unknown")
        emoji = "✅" if pnl >= 0 else "🔴"
        msg = (
            f"{emoji} *Position Closed*\n\n"
            f"`{token}`\n"
            f"P&L: `{'%+.2f' % pnl}%` | Reason: {reason}"
        )
        await self.send_message(msg)

    async def _on_budget_alert(self, payload: dict) -> None:
        severity = payload.get("severity", "moderate")
        message = payload.get("message", "Budget threshold reached")
        tier = "FLASH" if severity == "critical" else "PRIORITY"
        tc = TIER_CONFIG[tier]
        msg = f"{tc['emoji']} *Budget Alert*\n\n{message}"
        await self.send_message(msg)

    async def _on_system_health(self, payload: dict) -> None:
        source = payload.get("source", "unknown")
        status = payload.get("status", "unknown")
        message = payload.get("message", "")
        if status == "critical":
            msg = f"⚠️ *System Health — {source}*\n\n{message}"
            await self.send_message(msg)

    # ─── Core Alert Evaluation Pipeline ──────────────────────────────────────

    async def evaluate_and_alert(
        self,
        signals: list[Signal],
        delta_direction: str = "mixed",
        token_address: str | None = None,
    ) -> bool:
        if not self.is_configured or not signals:
            return False
        if self._is_muted():
            return False

        fresh_signals = [s for s in signals if not self._is_semantic_duplicate(s)]
        if not fresh_signals:
            await self._publish("TELEGRAM_SUPPRESSED", {"reason": "semantic_duplicate"})
            return False

        evaluation = await self._evaluate(fresh_signals, delta_direction, token_address)

        if not evaluation.should_alert:
            await self._publish("TELEGRAM_SUPPRESSED", {"reason": evaluation.suppress_reason})
            return False

        if not self._check_rate_limit(evaluation.tier):
            await self._publish("TELEGRAM_SUPPRESSED", {"reason": f"rate_limited:{evaluation.tier}"})
            return False

        message = self._format_tiered_alert(evaluation, delta_direction)
        result = await self.send_message(message)

        if result.get("ok"):
            for sig in fresh_signals:
                self._record_content_hash(sig)
            self._record_alert(evaluation.tier, evaluation.headline)
            await self._publish("TELEGRAM_SENT", {
                "tier": evaluation.tier,
                "headline": evaluation.headline,
                "message_id": result.get("message_id"),
                "token_address": token_address,
            })
            logger.info("[Telegram v2] %s alert sent: %s", evaluation.tier, evaluation.headline)
            return True

        return False

    async def _evaluate(
        self,
        signals: list[Signal],
        delta_direction: str,
        token_address: str | None,
    ) -> AlertEvaluation:
        if self.llm_client is not None:
            try:
                evaluation = await self._llm_evaluate(signals, delta_direction, token_address)
                if evaluation is not None:
                    return evaluation
            except Exception as exc:
                logger.warning("[Telegram v2] LLM eval failed, falling back to rules: %s", exc)

        evaluation = self._rule_evaluator.evaluate(signals, delta_direction)
        evaluation.source = "rules"
        return evaluation

    async def _llm_evaluate(
        self,
        signals: list[Signal],
        delta_direction: str,
        token_address: str | None,
    ) -> AlertEvaluation | None:
        user_msg = LLMEvaluationPrompt.build_user_message(signals, delta_direction, token_address)
        response_text = await self.llm_client.complete(
            system=LLMEvaluationPrompt.SYSTEM,
            user=user_msg,
            max_tokens=800,
        )
        parsed = parse_llm_json(response_text)
        if parsed is None or not isinstance(parsed.get("shouldAlert"), bool):
            return None
        evaluation = llm_response_to_evaluation(parsed, source="llm")
        evaluation.token_address = token_address
        return evaluation

    # ─── Message Sending ──────────────────────────────────────────────────────

    async def send_message(
        self,
        text: str,
        chat_id: str | None = None,
        parse_mode: str = "Markdown",
        disable_preview: bool = True,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        if not self.is_configured or not self._session:
            return {"ok": False}

        target_chat = chat_id or self.chat_id
        chunks = self._chunk_text(text, TELEGRAM_MAX_TEXT)
        last_result: dict[str, Any] = {"ok": False}

        for i, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": target_chat,
                "text": chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_preview,
            }
            if reply_to_message_id and i == 0:
                payload["reply_to_message_id"] = reply_to_message_id

            try:
                async with self._session.post(
                    f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage",
                    json=payload,
                ) as resp:
                    if not resp.ok:
                        body = await resp.text()
                        logger.error("[Telegram v2] Send failed (%s): %s", resp.status, body[:200])
                        return last_result
                    data = await resp.json()
                    last_result = {
                        "ok": True,
                        "message_id": data.get("result", {}).get("message_id"),
                    }
            except Exception as exc:
                logger.error("[Telegram v2] Send error: %s", exc)
                return {"ok": False}

        return last_result

    @staticmethod
    def _chunk_text(text: str, max_len: int = TELEGRAM_MAX_TEXT) -> list[str]:
        if not text:
            return []
        if len(text) <= max_len:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_len, len(text))
            if end < len(text):
                last_nl = text.rfind("\n", start, end)
                if last_nl > start:
                    end = last_nl + 1
            chunks.append(text[start:end])
            start = end
        return chunks

    # ─── Command Registry ─────────────────────────────────────────────────────

    def on_command(self, command: str, handler: CommandHandler) -> None:
        self._command_handlers[command.lower()] = handler

    # ─── Polling Loop ─────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[Telegram v2] Poll loop error: %s", exc)
            await asyncio.sleep(self.poll_interval_s)

    async def _poll_once(self) -> None:
        if not self._session:
            return
        params = {
            "offset": self._last_update_id + 1,
            "timeout": 0,
            "limit": 10,
            "allowed_updates": ["message"],
        }
        try:
            async with self._session.get(
                f"{TELEGRAM_API_BASE}/bot{self.bot_token}/getUpdates",
                params=params,
                timeout=aiohttp.ClientTimeout(total=12),
            ) as resp:
                if not resp.ok:
                    return
                data = await resp.json()

            if not data.get("ok") or not isinstance(data.get("result"), list):
                return

            for update in data["result"]:
                self._last_update_id = max(self._last_update_id, update.get("update_id", 0))
                msg = update.get("message")
                if not msg or not msg.get("text"):
                    continue
                if str(msg.get("chat", {}).get("id")) != self.chat_id:
                    continue
                await self._handle_message(msg)

        except asyncio.TimeoutError:
            pass
        except Exception as exc:
            if "aborted" not in str(exc).lower():
                logger.error("[Telegram v2] Poll error: %s", exc)

    async def _handle_message(self, msg: dict) -> None:
        text = msg.get("text", "").strip()
        parts = text.split()
        if not parts:
            return
        raw_command = parts[0].lower()
        command = self._normalize_command(raw_command)
        if not command:
            return

        args = " ".join(parts[1:])
        reply_chat_id = str(msg.get("chat", {}).get("id"))
        message_id = msg.get("message_id")

        if command == "/help":
            all_commands = {**self.KNOWN_COMMANDS, **{
                cmd: "Custom command" for cmd in self._command_handlers
                if cmd not in self.KNOWN_COMMANDS
            }}
            lines = [f"{cmd} — {desc}" for cmd, desc in all_commands.items()]
            await self.send_message(
                f"🐺 *FENRIR COMMANDS*\n\n" + "\n".join(lines) + "\n\n_Commands are case-insensitive_",
                chat_id=reply_chat_id,
                reply_to_message_id=message_id,
            )
            return

        if command == "/mute":
            hours = float(args) if args else 1.0
            self._mute_until = time.time() + hours * 3600
            until_str = time.strftime("%H:%M UTC", time.gmtime(self._mute_until))
            await self.send_message(
                f"🔇 Alerts muted for {hours:.0f}h — until {until_str}\nUse /unmute to resume.",
                chat_id=reply_chat_id,
                reply_to_message_id=message_id,
            )
            return

        if command == "/unmute":
            self._mute_until = None
            await self.send_message("🔔 Alerts resumed.", chat_id=reply_chat_id, reply_to_message_id=message_id)
            return

        if command == "/alerts":
            recent = self._alert_history[-10:]
            if not recent:
                await self.send_message("No recent alerts.", chat_id=reply_chat_id, reply_to_message_id=message_id)
                return
            lines = [
                f"{TIER_CONFIG[r.tier]['emoji']} {r.tier} — "
                f"{time.strftime('%H:%M UTC', time.gmtime(r.timestamp))} — "
                f"{r.headline}"
                for r in recent
            ]
            await self.send_message(
                f"📋 *Recent Alerts (last {len(recent)})*\n\n" + "\n".join(lines),
                chat_id=reply_chat_id,
                reply_to_message_id=message_id,
            )
            return

        handler = self._command_handlers.get(command)
        if handler:
            try:
                response = await handler(args, message_id)
                if response:
                    await self.send_message(response, chat_id=reply_chat_id, reply_to_message_id=message_id)
            except Exception as exc:
                logger.error("[Telegram v2] Command %s error: %s", command, exc)
                await self.send_message(f"❌ Command failed: {exc}", chat_id=reply_chat_id, reply_to_message_id=message_id)

    def _normalize_command(self, raw: str) -> str | None:
        if not raw.startswith("/"):
            return None
        at_idx = raw.find("@")
        if at_idx == -1:
            return raw
        command = raw[:at_idx]
        mentioned_bot = raw[at_idx + 1:].lower()
        if not self._bot_username or mentioned_bot == self._bot_username:
            return command
        return None

    # ─── Semantic Dedup ───────────────────────────────────────────────────────

    def _content_hash(self, signal: Signal) -> str:
        if signal.text:
            normalized = signal.text.lower().replace("\n", " ")
            normalized = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", normalized)
            normalized = re.sub(r"\d+\.\d+%?", "NUM", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()[:120]
        elif signal.label:
            normalized = f"{signal.label}:{signal.direction or 'none'}"
        else:
            normalized = signal.key
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _is_semantic_duplicate(self, signal: Signal) -> bool:
        h = self._content_hash(signal)
        last_seen = self._content_hashes.get(h)
        if last_seen is None:
            return False
        return (time.time() - last_seen) < self.dedup_window_s

    def _record_content_hash(self, signal: Signal) -> None:
        self._content_hashes[self._content_hash(signal)] = time.time()
        cutoff = time.time() - self.dedup_prune_s
        stale = [h for h, ts in self._content_hashes.items() if ts < cutoff]
        for h in stale:
            del self._content_hashes[h]

    # ─── Rate Limiting ────────────────────────────────────────────────────────

    def _check_rate_limit(self, tier: str) -> bool:
        config = TIER_CONFIG.get(tier)
        if not config:
            return True
        now = time.time()
        same_tier = [r for r in self._alert_history if r.tier == tier]
        if same_tier:
            elapsed = now - same_tier[-1].timestamp
            if elapsed < config["cooldown_s"]:
                return False
        recent_count = sum(1 for r in self._alert_history if r.tier == tier and r.timestamp > now - 3600)
        return recent_count < config["max_per_hour"]

    def _record_alert(self, tier: str, headline: str) -> None:
        self._alert_history.append(_AlertRecord(tier=tier, timestamp=time.time(), headline=headline))
        if len(self._alert_history) > 50:
            self._alert_history = self._alert_history[-50:]

    def _is_muted(self) -> bool:
        if self._mute_until is None:
            return False
        if time.time() > self._mute_until:
            self._mute_until = None
            return False
        return True

    # ─── Message Formatting ───────────────────────────────────────────────────

    def _format_tiered_alert(self, evaluation: AlertEvaluation, direction: str) -> str:
        tc = TIER_CONFIG.get(evaluation.tier, TIER_CONFIG["ROUTINE"])
        confidence_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "⚪"}.get(evaluation.confidence, "⚪")

        lines = [
            f"{tc['emoji']} *FENRIR {tc['label']}*",
            "",
            f"*{_escape_md(evaluation.headline)}*",
            "",
            evaluation.reason,
            "",
            f"Confidence: {confidence_emoji} {evaluation.confidence}",
            f"Direction: {direction.upper()}",
        ]

        if evaluation.cross_correlation:
            lines.append(f"Cross-correlation: {evaluation.cross_correlation}")
        if evaluation.token_address:
            lines.append(f"Token: `{evaluation.token_address}`")
        if evaluation.actionable and evaluation.actionable != "Monitor":
            lines += ["", f"💡 *Action:* {_escape_md(evaluation.actionable)}"]
        if evaluation.signals:
            lines += ["", "*Signals:*"]
            for sig in evaluation.signals:
                lines.append(f"• {_escape_md(sig)}")

        lines += ["", f"_{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} · {evaluation.source}_"]
        return "\n".join(lines)

    # ─── Bot Setup ────────────────────────────────────────────────────────────

    async def _load_bot_identity(self) -> None:
        if not self._session:
            return
        async with self._session.get(
            f"{TELEGRAM_API_BASE}/bot{self.bot_token}/getMe",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json()
            if data.get("ok") and data.get("result", {}).get("username"):
                self._bot_username = data["result"]["username"].lower()
            else:
                raise RuntimeError(f"getMe failed: {data}")

    async def _register_bot_commands(self) -> None:
        if not self._session:
            return
        try:
            chat_id = int(self.chat_id)
        except ValueError:
            return

        bot_commands = [
            {"command": cmd.lstrip("/"), "description": desc[:256]}
            for cmd, desc in self.KNOWN_COMMANDS.items()
        ]
        try:
            async with self._session.post(
                f"{TELEGRAM_API_BASE}/bot{self.bot_token}/setMyCommands",
                json={"commands": bot_commands, "scope": {"type": "chat", "chat_id": chat_id}},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.warning("[Telegram v2] setMyCommands failed: %s", data)
        except Exception as exc:
            logger.warning("[Telegram v2] Command registration error: %s", exc)

    # ─── EventBus Helper ──────────────────────────────────────────────────────

    async def _publish(self, event_type: str, payload: dict) -> None:
        try:
            if hasattr(self.event_bus, "publish_async"):
                await self.event_bus.publish_async(event_type, payload)
            elif hasattr(self.event_bus, "publish"):
                self.event_bus.publish(event_type, payload)
        except Exception as exc:
            logger.debug("[Telegram v2] EventBus publish error (%s): %s", event_type, exc)

    @staticmethod
    def _dict_to_signal(d: dict) -> Signal:
        return Signal(
            key=d.get("key", "unknown"),
            label=d.get("label", d.get("key", "unknown")),
            severity=d.get("severity", "moderate"),
            direction=d.get("direction", "new"),
            value=d.get("value") or d.get("to"),
            prev_value=d.get("prev_value") or d.get("from"),
            pct_change=d.get("pct_change"),
            text=d.get("text"),
            token_address=d.get("token_address"),
            source=d.get("source"),
            extra=d.get("extra", {}),
        )


# ─── Utilities ────────────────────────────────────────────────────────────────

def _escape_md(text: str) -> str:
    """Escape legacy Markdown special characters for Telegram."""
    if not text:
        return ""
    return re.sub(r"([_*`\[])", r"\\\1", text)
