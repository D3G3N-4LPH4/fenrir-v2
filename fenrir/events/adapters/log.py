#!/usr/bin/env python3
"""
FENRIR - Log Event Adapter

Bridges the event bus to the existing FenrirLogger.
Replaces scattered logger.info() calls with centralized event handling.
"""

import logging

from fenrir.events.bus import EventListener
from fenrir.events.types import EventSeverity, TradeEvent

logger = logging.getLogger("FENRIR")


class LogAdapter(EventListener):
    """
    Forwards events to the standard Python logging system.

    Severity mapping:
        DEBUG    -> logger.debug
        INFO     -> logger.info
        WARNING  -> logger.warning
        CRITICAL -> logger.info (with emoji prefix for visibility)
    """

    min_severity = EventSeverity.INFO

    SEVERITY_MAP = {
        EventSeverity.DEBUG: logging.DEBUG,
        EventSeverity.INFO: logging.INFO,
        EventSeverity.WARNING: logging.WARNING,
        EventSeverity.CRITICAL: logging.INFO,  # CRITICAL trades still log at INFO
    }

    async def on_event(self, event: TradeEvent) -> None:
        level = self.SEVERITY_MAP.get(event.severity, logging.INFO)
        prefix = ""

        if event.severity == EventSeverity.CRITICAL:
            prefix = "⚡ "
        elif event.severity == EventSeverity.WARNING:
            prefix = "⚠️  "

        logger.log(level, "%s%s", prefix, event.message)
