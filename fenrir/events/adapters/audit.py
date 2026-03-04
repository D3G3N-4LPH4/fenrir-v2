#!/usr/bin/env python3
"""
FENRIR - Audit Event Adapter

Records every significant event to the Merkle hash-chain audit trail.
Silent, non-blocking, and tamper-evident.
"""

from fenrir.data.audit import AuditChain
from fenrir.events.bus import EventListener
from fenrir.events.types import EventSeverity, TradeEvent


class AuditAdapter(EventListener):
    """
    Writes events to the AuditChain.

    Captures everything at INFO severity and above.
    Skips PRICE_UPDATE at DEBUG to avoid flooding the chain.
    """

    min_severity = EventSeverity.INFO

    def __init__(self, audit_chain: AuditChain):
        self.audit = audit_chain

    async def on_event(self, event: TradeEvent) -> None:
        self.audit.record(
            event_type=event.event_type,
            token_address=event.token_address,
            payload={
                "message": event.message,
                "severity": event.severity.value,
                "category": event.category.value,
                "symbol": event.token_symbol,
                **event.data,
            },
            strategy_id=event.strategy_id,
        )

    async def shutdown(self) -> None:
        self.audit.close()
