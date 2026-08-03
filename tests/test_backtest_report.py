#!/usr/bin/env python3
"""
FENRIR - Backtest report + CLI tests (Phase 6.3)

The text report renders per-strategy rows, the combined line, and the confluence
comparison/verdict; the CLI loads a JSON samples file, runs the portfolio backtester,
and prints a report. No network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fenrir.backtest import PortfolioBacktester, format_report
from fenrir.backtest.models import (
    BacktestMetrics,
    BacktestResult,
    BacktestTrade,
)
from fenrir.backtest.portfolio import PortfolioResult
from fenrir.config import BotConfig
from fenrir.strategies.momentum import MomentumStrategy

TOK = "UP111111111111111111111111111111111111111111"


def _trade(token: str, sid: str, pnl: float) -> BacktestTrade:
    return BacktestTrade(token, sid, 1.0, 1.0, "end_of_data", 1, pnl, 0.5)


class TestReport:
    def test_renders_sections(self) -> None:
        result = PortfolioResult(
            per_strategy={
                "momentum": BacktestResult(
                    "momentum",
                    trades=[_trade(TOK, "momentum", 60.0)],
                    metrics=BacktestMetrics(trades=1, wins=1, win_rate=1.0, expectancy_pct=60.0),
                    samples_evaluated=3,
                    samples_entered=1,
                )
            },
            combined_metrics=BacktestMetrics(trades=1, wins=1, win_rate=1.0, expectancy_pct=60.0),
            non_confluent_metrics=BacktestMetrics(trades=1, wins=1, win_rate=1.0),
        )
        report = format_report(result)
        assert "FENRIR BACKTEST REPORT" in report
        assert "momentum" in report
        assert "COMBINED" in report
        assert "Confluence" in report
        assert "entered 1/3 evaluated" in report

    def test_confluence_edge_verdict(self) -> None:
        # Confluent expectancy 20 vs non-confluent 5 → +15% "helped".
        result = PortfolioResult(
            confluent_metrics=BacktestMetrics(trades=2, expectancy_pct=20.0),
            non_confluent_metrics=BacktestMetrics(trades=3, expectancy_pct=5.0),
            confluent_tokens=[TOK],
        )
        report = format_report(result)
        assert "confluence edge (expectancy): +15.00% — helped" in report

    def test_confluence_edge_na(self) -> None:
        result = PortfolioResult(
            combined_metrics=BacktestMetrics(trades=1),
            non_confluent_metrics=BacktestMetrics(trades=1),
        )
        report = format_report(result)
        assert "confluence edge: n/a" in report

    def test_real_result_formats(self) -> None:
        from fenrir.backtest.models import BacktestSample

        sample = BacktestSample(
            token_address=TOK,
            token_data={"token_address": TOK},
            market_data=_momentum_md(),
            forward_prices=[1.0, 1.6, 1.7],
            frame_seconds=600.0,
        )
        result = PortfolioBacktester().run([MomentumStrategy(BotConfig())], [sample])
        report = format_report(result)
        assert "momentum" in report
        assert "COMBINED" in report


def _momentum_md() -> Any:
    from fenrir.filters import MarketData

    return MarketData(
        token_address=TOK,
        pair_address="P",
        dex_id="raydium",
        age_minutes=120.0,
        market_cap_usd=500_000.0,
        price_usd=0.001,
        liquidity_usd=100_000.0,
        volume_5m_usd=30_000.0,
        volume_1h_usd=200_000.0,
        txns_5m_buys=70,
        txns_5m_sells=30,
        price_change_5m_pct=2.0,
        price_change_1h_pct=25.0,
        price_change_24h_pct=150.0,
    )


class TestCli:
    def _samples_file(self, tmp_path: Path) -> Path:
        record = {
            "token_address": TOK,
            "symbol": "UP",
            "market_data": {
                "age_minutes": 120.0,
                "market_cap_usd": 500_000.0,
                "liquidity_usd": 100_000.0,
                "volume_5m_usd": 30_000.0,
                "volume_1h_usd": 200_000.0,
                "txns_5m_buys": 70,
                "txns_5m_sells": 30,
                "price_change_5m_pct": 2.0,
                "price_change_1h_pct": 25.0,
                "price_change_24h_pct": 150.0,
            },
            "forward_prices": [1.0, 1.6, 1.7],
            "frame_seconds": 600.0,
        }
        path = tmp_path / "history.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        return path

    def test_cli_runs(self, tmp_path: Path, capsys: Any) -> None:
        from tools.signal_backtest import main

        rc = main(["--samples", str(self._samples_file(tmp_path)), "--strategies", "momentum"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "FENRIR BACKTEST REPORT" in out
        assert "Loaded 1 samples" in out

    def test_cli_no_samples(self, tmp_path: Path, capsys: Any) -> None:
        from tools.signal_backtest import main

        empty = tmp_path / "empty.json"
        empty.write_text("[]", encoding="utf-8")
        rc = main(["--samples", str(empty), "--strategies", "momentum"])
        assert rc == 1

    def test_cli_unknown_strategy(self, tmp_path: Path, capsys: Any) -> None:
        from tools.signal_backtest import main

        rc = main(["--samples", str(self._samples_file(tmp_path)), "--strategies", "nope"])
        assert rc == 1
        assert "unknown strategy" in capsys.readouterr().out
