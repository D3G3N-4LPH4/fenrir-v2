#!/usr/bin/env python3
"""
FENRIR - Portfolio backtest + loader tests (Phase 6.2)

Multi-strategy backtest over shared history: per-strategy results, combined metrics, and
the confluence split (confluent vs non-confluent trades). Plus the dict/JSON loader
including malformed-record skipping. No network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fenrir.backtest import (
    PortfolioBacktester,
    load_samples,
    sample_from_dict,
    samples_from_dicts,
)
from fenrir.backtest.models import BacktestSample
from fenrir.config import BotConfig
from fenrir.filters import MarketData
from fenrir.strategies.mean_reversion import MeanReversionStrategy
from fenrir.strategies.momentum import MomentumStrategy

TOK_UP = "UP111111111111111111111111111111111111111111"
TOK_DOWN = "DN222222222222222222222222222222222222222222"


def _momentum_md(token: str) -> MarketData:
    return MarketData(
        token_address=token,
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


def _reversion_md(token: str) -> MarketData:
    return MarketData(
        token_address=token,
        pair_address="P",
        dex_id="raydium",
        age_minutes=180.0,
        market_cap_usd=500_000.0,
        price_usd=0.001,
        liquidity_usd=100_000.0,
        volume_1h_usd=200_000.0,
        txns_5m_buys=55,
        txns_5m_sells=45,
        price_change_5m_pct=1.0,
        price_change_1h_pct=-25.0,
        price_change_24h_pct=-10.0,
    )


def _sample(token: str, md: MarketData, prices: list[float]) -> BacktestSample:
    return BacktestSample(
        token_address=token,
        token_data={"token_address": token},
        market_data=md,
        forward_prices=prices,
        frame_seconds=600.0,
    )


class TestPortfolio:
    def test_per_strategy_and_combined(self) -> None:
        cfg = BotConfig()
        strategies = [MomentumStrategy(cfg), MeanReversionStrategy(cfg)]
        # TOK_UP fires momentum only; TOK_DOWN fires mean_reversion only.
        samples = [
            _sample(TOK_UP, _momentum_md(TOK_UP), [1.0, 1.6, 1.7]),
            _sample(TOK_DOWN, _reversion_md(TOK_DOWN), [1.0, 1.3, 1.4]),
        ]
        res = PortfolioBacktester().run(strategies, samples)
        assert set(res.per_strategy) == {"momentum", "mean_reversion"}
        assert res.per_strategy["momentum"].samples_entered == 1
        assert res.per_strategy["mean_reversion"].samples_entered == 1
        assert res.combined_metrics.trades == 2

    def test_confluence_split(self) -> None:
        cfg = BotConfig()
        strategies = [MomentumStrategy(cfg), MeanReversionStrategy(cfg)]
        # These two strategies are mutually exclusive by design, so no token is confluent
        # → all trades land in non_confluent, none in confluent.
        samples = [
            _sample(TOK_UP, _momentum_md(TOK_UP), [1.0, 1.6, 1.7]),
            _sample(TOK_DOWN, _reversion_md(TOK_DOWN), [1.0, 1.3, 1.4]),
        ]
        res = PortfolioBacktester().run(strategies, samples, confluence_min_sources=2)
        assert res.confluent_tokens == []
        assert res.confluent_metrics.trades == 0
        assert res.non_confluent_metrics.trades == 2

    def test_confluent_token_detected(self) -> None:
        # A fake second strategy that also enters TOK_UP makes it confluent.
        cfg = BotConfig()

        class _AlsoMomentum(MomentumStrategy):
            strategy_id = "momentum_2"

        strategies: list[Any] = [MomentumStrategy(cfg), _AlsoMomentum(cfg)]
        samples = [_sample(TOK_UP, _momentum_md(TOK_UP), [1.0, 1.6, 1.7])]
        res = PortfolioBacktester().run(strategies, samples, confluence_min_sources=2)
        assert res.confluent_tokens == [TOK_UP]
        assert res.confluent_metrics.trades == 2  # both strategies' trades on the token
        assert res.non_confluent_metrics.trades == 0

    def test_to_dict(self) -> None:
        res = PortfolioBacktester().run(
            [MomentumStrategy(BotConfig())],
            [_sample(TOK_UP, _momentum_md(TOK_UP), [1.0, 1.6, 1.7])],
        )
        d = res.to_dict()
        assert "momentum" in d["per_strategy"]
        assert d["combined"]["trades"] == 1


class TestLoader:
    def _record(self, token: str = TOK_UP) -> dict[str, Any]:
        return {
            "token_address": token,
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

    def test_sample_from_dict(self) -> None:
        sample = sample_from_dict(self._record())
        assert sample is not None
        assert sample.token_address == TOK_UP
        assert sample.forward_prices == [1.0, 1.6, 1.7]
        assert sample.market_data.price_change_1h_pct == 25.0
        assert sample.frame_seconds == 600.0

    def test_loaded_sample_backtests(self) -> None:
        sample = sample_from_dict(self._record())
        assert sample is not None
        res = PortfolioBacktester().run([MomentumStrategy(BotConfig())], [sample])
        assert res.per_strategy["momentum"].samples_entered == 1

    def test_malformed_records_skipped(self) -> None:
        records = [
            self._record(),
            {"token_address": TOK_DOWN},  # no forward_prices
            {"forward_prices": [1.0]},  # no token
            {"token_address": "X", "forward_prices": "notalist"},
            {"token_address": "Y", "forward_prices": [1.0], "market_data": {"bogus_key": 1}},
        ]
        samples = samples_from_dicts(records)
        assert len(samples) == 1
        assert samples[0].token_address == TOK_UP

    def test_load_from_json_file(self, tmp_path: Path) -> None:
        path = tmp_path / "samples.json"
        path.write_text(json.dumps({"samples": [self._record()]}), encoding="utf-8")
        samples = load_samples(path)
        assert len(samples) == 1

    def test_load_top_level_list(self, tmp_path: Path) -> None:
        path = tmp_path / "samples.json"
        path.write_text(json.dumps([self._record()]), encoding="utf-8")
        assert len(load_samples(path)) == 1
