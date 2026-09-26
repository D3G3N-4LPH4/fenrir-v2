#!/usr/bin/env python3
"""
FENRIR - EVM collect-and-report CLI tests (Phase 7, read-only)

Covers the scanner's drain_collections (end-of-window: wait, don't cancel) and the CLI's
report helper (backtest whatever was collected; a clear message when nothing was). The
full main() is network + wall-clock bound and not unit-tested. No network here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fenrir.config import BotConfig
from fenrir.evm import EvmEvaluatorScanner
from fenrir.evm.evaluator import EvmEvaluation
from fenrir.filters import MarketData
from fenrir.signals import SignalDirection
from fenrir.signals.models import Signal
from tools.evm_collect import _report_from_samples

TOKEN = "0x1234567890abcdef1234567890abcdef12345678"


def _eval() -> EvmEvaluation:
    return EvmEvaluation(
        chain="ethereum",
        token_address=TOKEN,
        symbol="PEPE",
        market_data=MarketData(token_address=TOKEN),
        signals=[Signal("momentum", TOKEN, SignalDirection.LONG, 0.6)],
    )


class _SlowCollector:
    """A collector whose collect() takes a beat, to exercise draining."""

    def __init__(self) -> None:
        self.done = 0

    async def collect(self, token: str, market_data: Any, symbol: str) -> None:
        await asyncio.sleep(0.05)
        self.done += 1


class _FakeEvaluator:
    async def evaluate(self, token: str, chain: Any = None) -> Any:
        return _eval()

    strategies: list[Any] = []


def _source(tokens: list[str]) -> Any:
    async def _get() -> list[str]:
        return tokens

    return _get


class TestDrainCollections:
    async def test_waits_for_inflight(self) -> None:
        col = _SlowCollector()
        sc = EvmEvaluatorScanner(_FakeEvaluator(), _source([TOKEN]), collector=col)
        await sc.scan_once()  # spawns a slow collection
        assert col.done == 0  # not finished yet
        await sc.drain_collections(timeout=2.0)
        assert col.done == 1  # drained to completion (not cancelled)

    async def test_drain_noop_when_nothing_pending(self) -> None:
        sc = EvmEvaluatorScanner(_FakeEvaluator(), _source([TOKEN]))
        await sc.drain_collections(timeout=1.0)  # no collector → no tasks → returns


class TestReportHelper:
    def _record(self) -> dict[str, Any]:
        return {
            "token_address": TOKEN,
            "symbol": "PEPE",
            "market_data": {
                "age_minutes": 120.0,
                "market_cap_usd": 5_000_000.0,
                "liquidity_usd": 500_000.0,
                "volume_5m_usd": 60_000.0,
                "volume_1h_usd": 400_000.0,
                "txns_5m_buys": 70,
                "txns_5m_sells": 30,
                "price_change_5m_pct": 2.0,
                "price_change_1h_pct": 25.0,
                "price_change_24h_pct": 150.0,
            },
            "forward_prices": [1.0, 1.6, 1.7],
            "frame_seconds": 600.0,
        }

    def test_report_from_samples(self, tmp_path: Path) -> None:
        path = tmp_path / "evm_samples.jsonl"
        path.write_text(json.dumps(self._record()) + "\n", encoding="utf-8")
        report = _report_from_samples(str(path), ["momentum"], BotConfig())
        assert "Collected 1 EVM samples" in report
        assert "FENRIR BACKTEST REPORT" in report
        assert "momentum" in report

    def test_report_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        report = _report_from_samples(str(path), ["momentum"], BotConfig())
        assert "nothing to report" in report
