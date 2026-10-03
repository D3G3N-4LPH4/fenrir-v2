"""Tests for SlotBook and CurveExitWatcher."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from solders.pubkey import Pubkey

from fenrir.trading.curve_exits import CurveExitWatcher
from fenrir.trading.slot_book import SlotBook


def test_slot_book_record_and_summary(tmp_path) -> None:
    book = SlotBook(path=str(tmp_path / "book.jsonl"))
    book.record(
        mint="mint1",
        create_slot=100,
        landed_slot=103,
        filled=True,
        amount_sol=0.05,
        tip_lamports=10_000,
        pnl_sol=0.01,
    )
    book.record(
        mint="mint2",
        create_slot=200,
        landed_slot=None,
        filled=False,
        amount_sol=0.05,
        tip_lamports=10_000,
    )
    rows = [json.loads(line) for line in (tmp_path / "book.jsonl").read_text().splitlines()]
    assert rows[0]["slot_offset"] == 3
    assert rows[1]["slot_offset"] is None

    s = book.summary()
    assert s["attempts"] == 2
    assert s["fill_rate"] == pytest.approx(0.5)
    assert s["median_slot_offset"] == 3
    assert s["expectancy_sol"] == pytest.approx(0.01)


def test_slot_book_empty_summary(tmp_path) -> None:
    book = SlotBook(path=str(tmp_path / "empty.jsonl"))
    s = book.summary()
    assert s["attempts"] == 0
    assert s["fill_rate"] == 0.0
    assert s["median_slot_offset"] is None
    assert s["expectancy_sol"] is None


def test_slot_book_creates_parent_dirs(tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "book.jsonl"
    book = SlotBook(path=str(nested))
    book.record(
        mint="m",
        create_slot=1,
        landed_slot=2,
        filled=True,
        amount_sol=0.05,
        tip_lamports=0,
    )
    assert nested.exists()


class _FakeAccount:
    def __init__(self, price: float):
        self._price = price

    def get_price(self) -> float:
        return self._price


class _FakeClient:
    def __init__(self, prices: list[float]):
        self._prices = prices

    async def account_subscribe(self, pubkey):
        for p in self._prices:
            yield _FakeAccount(p)
            await asyncio.sleep(0)


def _position(pnl_pct: float, stop_at: float):
    pos = SimpleNamespace()
    pos.update_price = lambda p: None
    pos.get_pnl_percent = lambda: pnl_pct
    pos.should_stop_loss = lambda pct: pnl_pct <= -abs(pct)
    return pos


@pytest.mark.asyncio
async def test_curve_exit_watcher_fires_stop() -> None:
    stops: list[tuple[str, str]] = []
    nudges = 0

    async def on_stop(addr: str, reason: str) -> None:
        stops.append((addr, reason))

    def nudge() -> None:
        nonlocal nudges
        nudges += 1

    positions = SimpleNamespace(positions={"tok1": _position(-30.0, 25.0)})
    client = _FakeClient([1.0, 0.9, 0.8])
    watcher = CurveExitWatcher(client, positions, on_stop, nudge)
    watcher.watch("tok1", str(Pubkey.new_unique()), 25.0)
    await asyncio.wait_for(watcher._tasks["tok1"], timeout=5)

    assert len(stops) == 1
    assert stops[0][0] == "tok1"
    assert "Curve stop" in stops[0][1]
    assert nudges >= 1


@pytest.mark.asyncio
async def test_curve_exit_watcher_no_stop_when_healthy() -> None:
    stops: list = []
    positions = SimpleNamespace(positions={"tok1": _position(10.0, 25.0)})
    client = _FakeClient([1.0, 1.1])

    async def on_stop(addr: str, reason: str) -> None:
        stops.append(addr)

    watcher = CurveExitWatcher(client, positions, on_stop, lambda: None)
    watcher.watch("tok1", str(Pubkey.new_unique()), 25.0)
    await asyncio.wait_for(watcher._tasks["tok1"], timeout=5)
    assert stops == []


@pytest.mark.asyncio
async def test_curve_exit_watcher_unwatch_cancels() -> None:
    async def never_ends(pubkey):
        while True:
            await asyncio.sleep(3600)
            yield _FakeAccount(1.0)

    client = SimpleNamespace(account_subscribe=never_ends)
    positions = SimpleNamespace(positions={})

    async def on_stop(addr: str, reason: str) -> None:
        pass

    watcher = CurveExitWatcher(client, positions, on_stop, lambda: None)
    watcher.watch("tok1", str(Pubkey.new_unique()), 25.0)
    await asyncio.sleep(0.05)
    assert "tok1" in watcher._tasks
    watcher.unwatch("tok1")
    assert "tok1" not in watcher._tasks


@pytest.mark.asyncio
async def test_curve_exit_watcher_watch_idempotent() -> None:
    async def on_stop(addr: str, reason: str) -> None:
        pass

    watcher = CurveExitWatcher(None, None, on_stop, lambda: None)
    watcher.watch("tok1", str(Pubkey.new_unique()), 25.0)
    first = watcher._tasks["tok1"]
    watcher.watch("tok1", str(Pubkey.new_unique()), 25.0)
    assert watcher._tasks["tok1"] is first
    watcher.unwatch("tok1")
