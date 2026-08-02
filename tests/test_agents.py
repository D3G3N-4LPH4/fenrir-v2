#!/usr/bin/env python3
"""
FENRIR - Multi-agent pipeline tests (Phase 3)

Acceptance criteria:
  - The scanner keeps flagging candidates while the execution agent is mid-trade
    (a slow executor does not stall the scanner/sizer).
  - A sizing (or execution) failure on one item never blocks the next.

The queue-backed worker design is what makes on_event non-blocking; these tests
pin that behavior. No network.
"""

from __future__ import annotations

import asyncio

import pytest

from fenrir.core.agents import AgentPipeline, ExecutionAgent, SizingAgent
from fenrir.events.bus import EventBus
from fenrir.events.types import position_sized_event, token_detected_event


def _detect(mint: str, creator: str | None = None):
    return token_detected_event(mint, mint.upper(), "n", 1.0, 5.0, creator=creator)


class TestPipelineWiring:
    @pytest.mark.asyncio
    async def test_detection_flows_through_all_three_stages(self):
        bus = EventBus()
        executed: list[tuple[str, float, str]] = []

        async def claims(addr, td):
            return ["sniper"]

        async def size(sid, td):
            return 0.1

        async def execute(td, amount, sid):
            executed.append((td.get("__mint", "?"), amount, sid))
            return True

        pipe = AgentPipeline(bus, claims, size, execute)
        await pipe.start()
        ev = _detect("MINT1")
        ev.data["__mint"] = "MINT1"
        await bus.emit(ev)
        await asyncio.sleep(0.05)  # let the workers drain
        await pipe.stop()

        assert executed == [("MINT1", 0.1, "sniper")]

    @pytest.mark.asyncio
    async def test_declined_sizing_stops_before_execution(self):
        bus = EventBus()
        executed = []

        async def claims(addr, td):
            return ["sniper"]

        async def size(sid, td):
            return None  # declined

        async def execute(td, amount, sid):
            executed.append(sid)
            return True

        pipe = AgentPipeline(bus, claims, size, execute)
        await pipe.start()
        await bus.emit(_detect("MINT1"))
        await asyncio.sleep(0.05)
        await pipe.stop()

        assert executed == []  # sizer declined -> no POSITION_SIZED -> no execute


class TestConcurrencyAcceptance:
    @pytest.mark.asyncio
    async def test_on_event_returns_without_awaiting_slow_processing(self):
        """on_event must enqueue and return fast, not block on the trade."""
        bus = EventBus()
        started = asyncio.Event()

        async def execute(td, amount, sid):
            started.set()
            await asyncio.sleep(10)  # a very slow trade
            return True

        agent = ExecutionAgent(bus, execute)
        await agent.start()
        ev = position_sized_event("M", "M", "sniper", 0.1, {})

        # on_event should return effectively instantly even though execute sleeps 10s.
        await asyncio.wait_for(agent.on_event(ev), timeout=0.5)
        await asyncio.sleep(0.02)
        assert started.is_set()  # worker picked it up in the background
        await agent.stop()

    @pytest.mark.asyncio
    async def test_scanner_keeps_flagging_while_executor_mid_trade(self):
        bus = EventBus()
        flagged: list[str] = []
        release = asyncio.Event()

        async def claims(addr, td):
            flagged.append(addr)
            return ["sniper"]

        async def size(sid, td):
            return 0.1

        async def execute(td, amount, sid):
            await release.wait()  # executor is stuck mid-trade
            return True

        pipe = AgentPipeline(bus, claims, size, execute)
        await pipe.start()

        # Fire the first detection; it reaches the (blocked) executor.
        await bus.emit(_detect("A"))
        await asyncio.sleep(0.02)
        # While the executor is stuck, more detections still get flagged + sized.
        for m in ("B", "C", "D"):
            await bus.emit(_detect(m))
        await asyncio.sleep(0.05)

        assert set(flagged) == {"A", "B", "C", "D"}  # scanner never stalled
        assert pipe.executor.processed == 0  # executor still mid-trade
        release.set()
        await asyncio.sleep(0.05)
        await pipe.stop()
        assert pipe.executor.processed == 4  # all drained once released

    @pytest.mark.asyncio
    async def test_sizing_failure_is_isolated(self):
        """A raised sizing for one item must not block the next item."""
        bus = EventBus()
        sized: list[str] = []

        async def size(sid, td):
            if td.get("boom"):
                raise RuntimeError("sizing blew up")
            sized.append(sid)
            return None

        agent = SizingAgent(bus, size)
        await agent.start()

        from fenrir.events.types import candidate_flagged_event

        await agent.on_event(candidate_flagged_event("A", "A", "boomstrat", {"boom": True}))
        await agent.on_event(candidate_flagged_event("B", "B", "goodstrat", {}))
        await asyncio.sleep(0.05)
        await agent.stop()

        assert agent.failed == 1  # the boom item
        assert sized == ["goodstrat"]  # the next item still processed


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_report_processed_counts(self):
        bus = EventBus()

        async def claims(a, t):
            return []

        async def size(s, t):
            return None

        async def execute(t, a, s):
            return True

        pipe = AgentPipeline(bus, claims, size, execute)
        await pipe.start()
        await bus.emit(_detect("A"))
        await asyncio.sleep(0.05)
        await pipe.stop()
        assert pipe.stats()["ScannerAgent"]["processed"] == 1
