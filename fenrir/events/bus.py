#!/usr/bin/env python3
"""
FENRIR - Event Bus

Async pub/sub event bus that decouples the bot's core logic from
its output channels. Components emit events; adapters consume them.

Usage:
    bus = EventBus()
    bus.register(TelegramAdapter(bot_token, chat_id))
    bus.register(LogAdapter(logger))
    bus.register(AuditAdapter(audit_chain))

    # Anywhere in the bot:
    await bus.emit(buy_executed_event(...))
"""

import asyncio
import logging
from abc import ABC, abstractmethod

from fenrir.events.types import EventCategory, EventSeverity, TradeEvent

logger = logging.getLogger(__name__)


class EventListener(ABC):
    """
    Base class for all event bus adapters.

    Subclasses decide which events they care about by overriding
    the filter methods, and what to do with them in on_event().
    """

    # Override these in subclasses to filter events
    min_severity: EventSeverity = EventSeverity.DEBUG
    categories: set[EventCategory] | None = None  # None = all categories
    event_types: set[str] | None = None  # None = all types

    def accepts(self, event: TradeEvent) -> bool:
        """Check if this listener wants to process the event."""
        # Severity filter
        severity_order = [
            EventSeverity.DEBUG,
            EventSeverity.INFO,
            EventSeverity.WARNING,
            EventSeverity.CRITICAL,
        ]
        if severity_order.index(event.severity) < severity_order.index(self.min_severity):
            return False

        # Category filter
        if self.categories is not None and event.category not in self.categories:
            return False

        # Event type filter
        if self.event_types is not None and event.event_type not in self.event_types:
            return False

        return True

    @abstractmethod
    async def on_event(self, event: TradeEvent) -> None:
        """Process an event. Must be non-blocking or use asyncio."""

    async def shutdown(self) -> None:  # noqa: B027
        """Optional cleanup when the bus shuts down."""


class EventBus:
    """
    Central async event bus.

    All bot components emit events here. Registered listeners
    receive events concurrently and independently — a slow or
    failing adapter never blocks the trading loop.
    """

    def __init__(self):
        self._listeners: list[EventListener] = []
        self._event_count: int = 0
        self._error_count: int = 0

    def register(self, listener: EventListener) -> None:
        """Register an event listener/adapter."""
        self._listeners.append(listener)

    def unregister(self, listener: EventListener) -> None:
        """Remove a listener."""
        self._listeners = [lst for lst in self._listeners if lst is not listener]

    async def emit(self, event: TradeEvent) -> None:
        """
        Emit an event to all registered listeners.

        Each listener runs concurrently. Errors in one adapter
        never affect others or the caller.
        """
        self._event_count += 1

        tasks = []
        for listener in self._listeners:
            if listener.accepts(event):
                tasks.append(self._safe_dispatch(listener, event))

        if tasks:
            await asyncio.gather(*tasks)

    async def _safe_dispatch(self, listener: EventListener, event: TradeEvent) -> None:
        """Dispatch to a listener with error isolation."""
        try:
            await listener.on_event(event)
        except Exception as e:
            self._error_count += 1
            listener_name = type(listener).__name__
            logger.error(
                "Event listener %s failed on %s: %s",
                listener_name,
                event.event_type,
                e,
            )

    async def shutdown(self) -> None:
        """Shut down all listeners gracefully."""
        for listener in self._listeners:
            try:
                await listener.shutdown()
            except Exception as e:
                logger.error("Listener shutdown error: %s", e)

    def get_stats(self) -> dict:
        """Bus statistics."""
        return {
            "listeners": len(self._listeners),
            "listener_types": [type(lst).__name__ for lst in self._listeners],
            "events_emitted": self._event_count,
            "dispatch_errors": self._error_count,
        }
