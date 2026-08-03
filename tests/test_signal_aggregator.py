#!/usr/bin/env python3
"""
FENRIR - Signal aggregator tests (Phase 5.2, confluence + ranking)

Covers confluence (noisy-OR over distinct sources, one-strategy-counts-once), TTL
expiry, direction separation, cross-token ranking, and raw-object ingestion. No network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fenrir.signals import SignalAggregator, SignalDirection
from fenrir.signals.models import Signal

TOK_A = "AAAA1111111111111111111111111111111111111111"
TOK_B = "BBBB2222222222222222222222222222222222222222"

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _sig(
    source: str,
    token: str,
    strength: float,
    direction: SignalDirection = SignalDirection.LONG,
    ts: datetime = _T0,
) -> Signal:
    return Signal(
        source=source, token_address=token, direction=direction, strength=strength, timestamp=ts
    )


class TestConfluence:
    def test_single_source(self) -> None:
        agg = SignalAggregator(now_fn=lambda: _T0)
        agg.add(_sig("momentum", TOK_A, 0.6))
        res = agg.confluence_for(TOK_A, now=_T0)
        assert res is not None
        assert res.sources == ["momentum"]
        assert res.combined_strength == pytest.approx(0.6)
        assert res.is_confluent(min_sources=2) is False

    def test_noisy_or_boosts_independent_agreement(self) -> None:
        agg = SignalAggregator(now_fn=lambda: _T0)
        agg.add(_sig("momentum", TOK_A, 0.5))
        agg.add(_sig("volume_anomaly", TOK_A, 0.5))
        res = agg.confluence_for(TOK_A, now=_T0)
        assert res is not None
        # 1 - (1-0.5)(1-0.5) = 0.75 — two independent 0.5s beat either alone.
        assert res.combined_strength == pytest.approx(0.75)
        assert res.source_count == 2
        assert res.is_confluent() is True
        assert res.max_strength == pytest.approx(0.5)

    def test_one_strategy_counts_once(self) -> None:
        # A chatty strategy emitting twice must not fake confluence; strongest kept.
        agg = SignalAggregator(now_fn=lambda: _T0)
        agg.add(_sig("momentum", TOK_A, 0.4))
        agg.add(_sig("momentum", TOK_A, 0.7))
        res = agg.confluence_for(TOK_A, now=_T0)
        assert res is not None
        assert res.sources == ["momentum"]
        assert res.combined_strength == pytest.approx(0.7)  # not combined with itself

    def test_none_when_no_signal(self) -> None:
        agg = SignalAggregator(now_fn=lambda: _T0)
        assert agg.confluence_for(TOK_A, now=_T0) is None

    def test_zero_strength_ignored(self) -> None:
        agg = SignalAggregator(now_fn=lambda: _T0)
        agg.add(_sig("momentum", TOK_A, 0.0))
        assert agg.confluence_for(TOK_A, now=_T0) is None


class TestDirection:
    def test_directions_do_not_mix(self) -> None:
        agg = SignalAggregator(now_fn=lambda: _T0)
        agg.add(_sig("momentum", TOK_A, 0.6, SignalDirection.LONG))
        agg.add(_sig("arbitrage", TOK_A, 0.6, SignalDirection.NEUTRAL))
        long_res = agg.confluence_for(TOK_A, SignalDirection.LONG, now=_T0)
        neutral_res = agg.confluence_for(TOK_A, SignalDirection.NEUTRAL, now=_T0)
        assert long_res is not None and long_res.sources == ["momentum"]
        assert neutral_res is not None and neutral_res.sources == ["arbitrage"]


class TestTTL:
    def test_stale_signals_expire(self) -> None:
        agg = SignalAggregator(ttl_seconds=60.0, now_fn=lambda: _T0)
        agg.add(_sig("momentum", TOK_A, 0.6, ts=_T0 - timedelta(seconds=120)))  # stale
        agg.add(_sig("volume_anomaly", TOK_A, 0.5, ts=_T0))  # fresh
        res = agg.confluence_for(TOK_A, now=_T0)
        assert res is not None
        assert res.sources == ["volume_anomaly"]  # stale one pruned

    def test_active_count_prunes(self) -> None:
        agg = SignalAggregator(ttl_seconds=60.0, now_fn=lambda: _T0)
        agg.add(_sig("a", TOK_A, 0.5, ts=_T0 - timedelta(seconds=120)))
        agg.add(_sig("b", TOK_A, 0.5, ts=_T0))
        assert agg.active_count == 1


class TestRanking:
    def test_confluent_token_ranks_above_lonely_stronger_one(self) -> None:
        agg = SignalAggregator(now_fn=lambda: _T0)
        # TOK_A: two independent 0.5s → combined 0.75.
        agg.add(_sig("momentum", TOK_A, 0.5))
        agg.add(_sig("volume_anomaly", TOK_A, 0.5))
        # TOK_B: one strong 0.7 → combined 0.7.
        agg.add(_sig("momentum", TOK_B, 0.7))
        ranked = agg.ranked(now=_T0)
        assert [r.token_address for r in ranked] == [TOK_A, TOK_B]
        assert ranked[0].combined_strength == pytest.approx(0.75)

    def test_min_sources_filter(self) -> None:
        agg = SignalAggregator(now_fn=lambda: _T0)
        agg.add(_sig("momentum", TOK_A, 0.5))
        agg.add(_sig("volume_anomaly", TOK_A, 0.5))
        agg.add(_sig("momentum", TOK_B, 0.9))  # single source
        confluent = agg.ranked(min_sources=2, now=_T0)
        assert [r.token_address for r in confluent] == [TOK_A]

    def test_empty(self) -> None:
        assert SignalAggregator(now_fn=lambda: _T0).ranked(now=_T0) == []


class TestIngest:
    def test_ingest_normalizes_raw_object(self) -> None:
        from fenrir.trading.arbitrage import ArbitrageDetector, VenueQuote

        opp = ArbitrageDetector().evaluate(
            TOK_A,
            [
                VenueQuote("raydium", price=1.00, liquidity_sol=100_000.0, fee_bps=10),
                VenueQuote("pumpswap", price=1.05, liquidity_sol=100_000.0, fee_bps=10),
            ],
            size_sol=1.0,
        )
        assert opp is not None
        agg = SignalAggregator(now_fn=lambda: _T0)
        sig = agg.ingest(opp)
        assert sig.source == "arbitrage"
        assert sig.direction is SignalDirection.NEUTRAL
        res = agg.confluence_for(TOK_A, SignalDirection.NEUTRAL, now=_T0)
        assert res is not None
