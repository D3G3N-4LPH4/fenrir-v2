"""Regression tests: a wedged provider call must never stall the scout batch.

tools/scout.py layers hard deadlines over the providers' own per-request
timeouts: per-token, per-source, and a run-level deadline that emits partial
results. These tests pin that behavior with providers that never return.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from fenrir.discovery.filters import FilterEngine  # noqa: E402
from fenrir.discovery.models import Chain  # noqa: E402
from fenrir.discovery.playbooks import PlaybookTagger  # noqa: E402
from fenrir.discovery.scoring import ScoringEngine  # noqa: E402
from tools import scout as scout_mod  # noqa: E402
from tools.scout import fetch_source_addresses, scout_chain, summarize_timings  # noqa: E402


class HangingDexScreener:
    """Source fetch works; per-token snapshot never returns."""

    async def fetch_boosted_addresses(self, chain):
        return ["So11111111111111111111111111111111111111112"]

    async def fetch_snapshot(self, addr, chain=None):
        await asyncio.sleep(3600)
        raise AssertionError("must not reach here")


class HangingSource:
    """The discovery source itself never returns."""

    async def fetch_boosted_addresses(self, chain):
        await asyncio.sleep(3600)
        raise AssertionError("must not reach here")


def _engines():
    return FilterEngine(), ScoringEngine(), PlaybookTagger()


def test_stuck_token_cannot_stall_batch(monkeypatch) -> None:
    """A token whose snapshot hangs is dropped at the token deadline."""
    monkeypatch.setattr(scout_mod, "TOKEN_TIMEOUT_SECONDS", 0.2)
    ds = HangingDexScreener()
    engine, scorer, tagger = _engines()
    timings: list = []
    t0 = time.perf_counter()
    cands, by_source = asyncio.run(
        scout_chain(
            Chain.SOLANA,
            ds,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            engine,
            scorer,
            tagger,
            ["boosted"],
            25,
            12,
            60.0,
            None,
            timings,
            None,
        )
    )
    elapsed = time.perf_counter() - t0
    assert cands == []
    assert by_source.get("boosted") == 1  # counted as scanned, not a candidate
    assert elapsed < 10  # 0.2s deadline honored, not the 3600s hang
    assert any(t["phase"] == "timeout" for t in timings)


def test_stuck_source_cannot_stall_discovery(monkeypatch) -> None:
    """A discovery source that hangs is dropped at the source deadline."""
    monkeypatch.setattr(scout_mod, "SOURCE_TIMEOUT_SECONDS", 0.2)
    t0 = time.perf_counter()
    out = asyncio.run(
        fetch_source_addresses(
            Chain.SOLANA,
            HangingSource(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            ["boosted"],
            25,
            12,
        )
    )
    elapsed = time.perf_counter() - t0
    assert out == [("boosted", [])]  # fail-open: empty, not fatal
    assert elapsed < 10


def test_summarize_timings_flags_timeouts() -> None:
    timings = [
        {"addr": "a1", "phase": "snapshot", "seconds": 1.5},
        {"addr": "a1", "phase": "safety", "seconds": 8.0},
        {"addr": "a2", "phase": "timeout", "seconds": 90.0, "after": "safety"},
        {"addr": "a3", "phase": "timeout", "seconds": 90.0, "after": None},
    ]
    s = summarize_timings(timings)
    assert s["token_timeouts"] == 2
    assert s["timeout_after_phase"] == {"safety": 1, "snapshot": 1}
    assert s["tokens_timed"] == 1
    assert s["phase_seconds"]["timeout"] == 180.0
    assert s["slowest_tokens"][0]["addr"] == "a1"
    assert s["slowest_tokens"][0]["seconds"] == 9.5
