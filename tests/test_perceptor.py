#!/usr/bin/env python3
"""
FENRIR - Perceptor provider tests

Pure mapper tests use a trimmed fixture captured from the live Perceptor
investigation of OXP (0x32dae312abe8f6fdb782907b85edbc90d2e74b02, 2026-09-28).
Provider cache/network tests use fakes — no network.
"""

from __future__ import annotations

import json
import os

import pytest

from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot
from fenrir.discovery.providers.perceptor import (
    PerceptorProvider,
    enrich_robinhood_safety,
    parse_perceptor,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "perceptor_oxp.json")


def load_fixture() -> dict:
    with open(FIXTURE) as f:
        return json.load(f)


def test_parse_oxp_verdict() -> None:
    r = parse_perceptor(load_fixture())
    assert r is not None
    assert r.band == "medium"
    assert r.band_label == "Caution"
    assert "fresh wallets" in (r.headline or "")
    assert r.investigation_id == "d69cff1d27574161b11fef9b28a561a1"


def test_parse_oxp_safety_signals() -> None:
    s = parse_perceptor(load_fixture()).safety
    assert s.lp_locked_or_burned is True
    assert s.lp_locked_pct == 100.0
    assert s.ownership_renounced is True
    assert s.mint_disabled is True
    assert s.blacklist_present is False
    assert s.freeze_disabled is True
    assert s.buy_tax_pct == 4.0
    assert s.sell_tax_pct == 0.0
    assert s.honeypot is False  # "Selling: Works" check
    assert s.risk_score == 50.0  # medium band
    assert any("Early buyers sold 82%" in f for f in s.risk_flags)
    assert not s.is_empty


def test_parse_incomplete_verdict() -> None:
    r = parse_perceptor({"investigation_id": "x", "verdict": {"band": "low"}})
    assert r is not None
    assert r.safety.risk_score == 15.0
    assert r.safety.lp_locked_or_burned is None
    assert r.safety.honeypot is None


def test_parse_garbage() -> None:
    assert parse_perceptor({}) is None
    assert parse_perceptor({"verdict": None}) is None
    assert parse_perceptor("nope") is None


def _snap(chain: Chain = Chain.ROBINHOOD) -> TokenSnapshot:
    return TokenSnapshot(
        token_address="0x32dae312abe8f6fdb782907b85edbc90d2e74b02",
        chain=chain,
        symbol="OXP",
    )


@pytest.mark.asyncio
async def test_enrich_skips_non_robinhood(tmp_path) -> None:
    p = PerceptorProvider(cache_path=str(tmp_path / "c.json"))
    snap = _snap(Chain.SOLANA)
    assert await enrich_robinhood_safety(snap, p) is None
    assert snap.safety.is_empty
    await p.close()


@pytest.mark.asyncio
async def test_enrich_skips_when_goplus_has_data(tmp_path) -> None:
    p = PerceptorProvider(cache_path=str(tmp_path / "c.json"))
    snap = _snap()
    snap.safety = SafetySignals(honeypot=False, buy_tax_pct=1.0)
    assert await enrich_robinhood_safety(snap, p) is None
    assert snap.safety.honeypot is False  # untouched
    await p.close()


@pytest.mark.asyncio
async def test_ensure_investigation_posts_once(tmp_path) -> None:
    posts: list[dict] = []

    class FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"investigation_id": "abc123"}

    class FakeSession:
        closed = False

        def post(self, url, json=None, timeout=None):
            posts.append(json)
            return FakeResp()

        async def close(self):
            pass

    p = PerceptorProvider(cache_path=str(tmp_path / "c.json"))

    async def fake_session():
        return FakeSession()

    p._get_session = fake_session  # type: ignore[method-assign]
    inv1 = await p.ensure_investigation(4663, "0xABC")
    inv2 = await p.ensure_investigation(4663, "0xabc")  # case-insensitive cache
    assert inv1 == inv2 == "abc123"
    assert len(posts) == 1  # second call served from cache
    assert posts[0] == {"chain_id": 4663, "address": "0xABC"}
    await p.close()


@pytest.mark.asyncio
async def test_refresh_report_completes_pending(tmp_path) -> None:
    data = load_fixture()

    class FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return data

    class FakeSession:
        closed = False

        def get(self, url, timeout=None):
            return FakeResp()

        async def close(self):
            pass

    cache = tmp_path / "c.json"
    cache.write_text(json.dumps({"0xabc": {
        "investigation_id": "d69cff1d27574161b11fef9b28a561a1",
        "status": "pending", "report": None, "checked_at": 0,
    }}))
    p = PerceptorProvider(cache_path=str(cache))

    async def fake_session():
        return FakeSession()

    p._get_session = fake_session  # type: ignore[method-assign]
    report = await p.refresh_report("0xABC")
    assert report is not None
    assert report.band == "medium"
    assert report.safety.lp_locked_or_burned is True
    # persisted: a second provider reading the same cache needs no network
    p2 = PerceptorProvider(cache_path=str(cache))
    assert p2.cached_report("0xabc").band == "medium"
    await p.close()
    await p2.close()


@pytest.mark.asyncio
async def test_enrich_merges_completed_report(tmp_path) -> None:
    data = load_fixture()
    cache = tmp_path / "c.json"
    cache.write_text(json.dumps({"0x32dae312abe8f6fdb782907b85edbc90d2e74b02": {
        "investigation_id": "d69cff1d27574161b11fef9b28a561a1",
        "status": "complete", "report": data, "checked_at": 0,
    }}))
    p = PerceptorProvider(cache_path=str(cache))
    snap = _snap()
    assert snap.safety.is_empty
    report = await enrich_robinhood_safety(snap, p)
    assert report is not None and report.band == "medium"
    assert snap.safety.lp_locked_or_burned is True
    assert snap.safety.honeypot is False
    await p.close()
