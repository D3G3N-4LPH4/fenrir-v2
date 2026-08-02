#!/usr/bin/env python3
"""
FENRIR - Multi-agent decision pipeline (Phase 3, strangler)

Decomposes the single ClaudeBrain decision loop into three coordinating async
agents that communicate over the existing EventBus:

    TOKEN_DETECTED --> ScannerAgent  --> CANDIDATE_FLAGGED
    CANDIDATE_FLAGGED --> SizingAgent --> POSITION_SIZED
    POSITION_SIZED   --> ExecutionAgent --> BUY_EXECUTED (existing lifecycle events)

Each agent is an EventListener that ENQUEUES on on_event and returns immediately,
then a background worker drains its queue. That decoupling is what gives the Phase 3
guarantees: the scanner keeps flagging while the execution agent is mid-trade (its
slow work runs in the executor's worker, not the emit chain), and a failure
processing one item never blocks the next (the worker loop isolates errors).

Strangler: this runs ALONGSIDE the existing loop, behind config.multi_agent_pipeline
_enabled (off by default). The TradingStrategy ABC stays the pluggable unit the
Scanner/Sizing agents consult.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fenrir.events.bus import EventBus, EventListener
from fenrir.events.types import (
    TradeEvent,
    candidate_flagged_event,
    position_sized_event,
)


class PipelineAgent(EventListener):
    """Base: an event-driven worker. on_event enqueues (fast); a background task
    drains the queue and processes items independently, isolating per-item errors.
    """

    name: str = "agent"

    def __init__(self, bus: EventBus, logger: Any = None, max_queue: int = 1000) -> None:
        self._bus = bus
        self._logger = logger
        self._queue: asyncio.Queue[TradeEvent] = asyncio.Queue(maxsize=max_queue)
        self._worker: asyncio.Task | None = None
        self.processed = 0
        self.failed = 0

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def on_event(self, event: TradeEvent) -> None:
        # Enqueue and return immediately so the emit chain is never blocked by this
        # agent's (possibly slow) processing.
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._log("warning", f"{self.name}: queue full, dropping {event.event_type}")

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._process(event)
                self.processed += 1
            except Exception as e:  # noqa: BLE001 - one bad item must not kill the worker
                self.failed += 1
                self._log("error", f"{self.name}: processing {event.event_type} failed: {e}")
            finally:
                self._queue.task_done()

    async def _process(self, event: TradeEvent) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _log(self, level: str, msg: str) -> None:
        if self._logger is not None:
            getattr(self._logger, level, self._logger.info)(msg)


class ScannerAgent(PipelineAgent):
    """Consumes detections; consults strategies to flag tradeable candidates."""

    name = "ScannerAgent"
    event_types = {"TOKEN_DETECTED"}

    def __init__(
        self,
        bus: EventBus,
        claims: Callable[[str, dict], Awaitable[list[str]]],
        logger: Any = None,
    ) -> None:
        """``claims(token_address, token_data) -> [strategy_id, ...]`` returns the
        strategies that want this token (the TradingStrategy ABC decides)."""
        super().__init__(bus, logger)
        self._claims = claims

    async def _process(self, event: TradeEvent) -> None:
        token_data = dict(event.data)
        strategy_ids = await self._claims(event.token_address or "", token_data)
        for sid in strategy_ids:
            await self._bus.emit(
                candidate_flagged_event(
                    token_address=event.token_address or "",
                    symbol=event.token_symbol or "???",
                    strategy_id=sid,
                    token_data=token_data,
                )
            )


class SizingAgent(PipelineAgent):
    """Consumes candidates; applies sizing + risk, emits POSITION_SIZED (or drops)."""

    name = "SizingAgent"
    event_types = {"CANDIDATE_FLAGGED"}

    def __init__(
        self,
        bus: EventBus,
        size: Callable[[str, dict], Awaitable[float | None]],
        logger: Any = None,
    ) -> None:
        """``size(strategy_id, token_data) -> amount_sol | None``. None = reject
        (risk/budget/geometry declined); a per-strategy failure or rejection here
        never blocks other candidates."""
        super().__init__(bus, logger)
        self._size = size

    async def _process(self, event: TradeEvent) -> None:
        token_data = event.data.get("token_data", {})
        amount = await self._size(event.strategy_id or "default", token_data)
        if amount is None or amount <= 0:
            return  # declined — isolated, next candidate continues
        await self._bus.emit(
            position_sized_event(
                token_address=event.token_address or "",
                symbol=event.token_symbol or "???",
                strategy_id=event.strategy_id or "default",
                amount_sol=amount,
                token_data=token_data,
            )
        )


class ExecutionAgent(PipelineAgent):
    """Consumes sized positions; executes the buy (its own worker, so a slow trade
    never blocks the scanner/sizer)."""

    name = "ExecutionAgent"
    event_types = {"POSITION_SIZED"}

    def __init__(
        self,
        bus: EventBus,
        execute: Callable[[dict, float, str], Awaitable[bool]],
        logger: Any = None,
    ) -> None:
        """``execute(token_data, amount_sol, strategy_id) -> success``."""
        super().__init__(bus, logger)
        self._execute = execute

    async def _process(self, event: TradeEvent) -> None:
        token_data = event.data.get("token_data", {})
        amount = float(event.data.get("amount_sol", 0.0))
        await self._execute(token_data, amount, event.strategy_id or "default")


class AgentPipeline:
    """Owns the three agents, registers them on the bus, and manages their workers.

    Register order is irrelevant (they're decoupled by event type); start() spins up
    every worker, stop() tears them down.
    """

    def __init__(
        self,
        bus: EventBus,
        claims: Callable[[str, dict], Awaitable[list[str]]],
        size: Callable[[str, dict], Awaitable[float | None]],
        execute: Callable[[dict, float, str], Awaitable[bool]],
        logger: Any = None,
    ) -> None:
        self.scanner = ScannerAgent(bus, claims, logger)
        self.sizer = SizingAgent(bus, size, logger)
        self.executor = ExecutionAgent(bus, execute, logger)
        self._agents: list[PipelineAgent] = [self.scanner, self.sizer, self.executor]
        for agent in self._agents:
            bus.register(agent)

    async def start(self) -> None:
        for agent in self._agents:
            await agent.start()

    async def stop(self) -> None:
        for agent in self._agents:
            await agent.stop()

    def stats(self) -> dict:
        return {a.name: {"processed": a.processed, "failed": a.failed} for a in self._agents}
