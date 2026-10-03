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
        data = json.load(f)
        assert isinstance(data, dict)
        return data


def test_parse_oxp_verdict() -> None:
    r = parse_perceptor(load_fixture())
    assert r is not None
    assert r.band == "medium"
    assert r.band_label == "Caution"
    assert "fresh wallets" in (r.headline or "")
    assert r.investigation_id == "d69cff1d27574161b11fef9b28a561a1"


def test_parse_oxp_safety_signals() -> None:
    r = parse_perceptor(load_fixture())
    assert r is not None
    s = r.safety
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
    # Deliberately wrong type: the parser must fail open, not raise.
    assert parse_perceptor("nope") is None  # type: ignore[arg-type]


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
            assert json is not None
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
    cache.write_text(
        json.dumps(
            {
                "0xabc": {
                    "investigation_id": "d69cff1d27574161b11fef9b28a561a1",
                    "status": "pending",
                    "report": None,
                    "checked_at": 0,
                }
            }
        )
    )
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
    r2 = p2.cached_report("0xabc")
    assert r2 is not None
    assert r2.band == "medium"
    await p.close()
    await p2.close()


@pytest.mark.asyncio
async def test_enrich_merges_completed_report(tmp_path) -> None:
    data = load_fixture()
    cache = tmp_path / "c.json"
    cache.write_text(
        json.dumps(
            {
                "0x32dae312abe8f6fdb782907b85edbc90d2e74b02": {
                    "investigation_id": "d69cff1d27574161b11fef9b28a561a1",
                    "status": "complete",
                    "report": data,
                    "checked_at": 0,
                }
            }
        )
    )
    p = PerceptorProvider(cache_path=str(cache))
    snap = _snap()
    assert snap.safety.is_empty
    report = await enrich_robinhood_safety(snap, p)
    assert report is not None and report.band == "medium"
    assert snap.safety.lp_locked_or_burned is True
    assert snap.safety.honeypot is False
    await p.close()


# ---------------------------------------------------------------------------
# Verdict follow-ups
# ---------------------------------------------------------------------------


def _oxp_report():
    return parse_perceptor(load_fixture())


def test_snapshot_context() -> None:
    from fenrir.discovery.providers.perceptor import snapshot_context

    snap = TokenSnapshot(
        chain=Chain.ROBINHOOD,
        token_address="0xabc",
        symbol="TST",
        name="Test Token",
    )
    ctx = snapshot_context(snap)
    assert ctx == {
        "symbol": "TST",
        "name": "Test Token",
        "chain": "robinhood",
        "dexscreener": "https://dexscreener.com/robinhood/0xabc",
    }


@pytest.mark.asyncio
async def test_ensure_investigation_stores_context(tmp_path) -> None:
    class FakeResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return {"investigation_id": "ctx1"}

    class FakeSession:
        def post(self, url, json=None, timeout=None):
            return FakeResp()

        async def close(self):
            pass

    p = PerceptorProvider(cache_path=str(tmp_path / "c.json"))

    async def fake_session():
        return FakeSession()

    p._get_session = fake_session  # type: ignore[method-assign]
    ctx = {"symbol": "TST", "chain": "robinhood"}
    inv = await p.ensure_investigation(4663, "0xDEF", ctx)
    assert inv == "ctx1"
    entry = p._load_cache()["0xdef"]
    assert entry["context"] == ctx
    assert entry["followup_sent"] is False
    # cache hit merges new context without a second POST
    await p.ensure_investigation(4663, "0xdef", {"name": "Test"})
    entry = p._load_cache()["0xdef"]
    assert entry["context"] == {"symbol": "TST", "chain": "robinhood", "name": "Test"}
    await p.close()


def test_format_perceptor_verdict() -> None:
    from fenrir.discovery.alerts import format_perceptor_verdict

    report = _oxp_report()
    assert report is not None
    ctx = {
        "symbol": "OXP",
        "name": "Perceptor",
        "chain": "robinhood",
        "dexscreener": "https://dexscreener.com/robinhood/0x32dae312abe8f6fdb782907b85edbc90d2e74b02",
    }
    msg = format_perceptor_verdict("0x32dae312abe8f6fdb782907b85edbc90d2e74b02", ctx, report)
    assert "*OXP*" in msg
    assert "Perceptor verdict" in msg
    assert "Caution" in msg  # band_label from the OXP fixture
    assert "`0x32dae312abe8f6fdb782907b85edbc90d2e74b02`" in msg
    assert (
        "[DexScreener](https://dexscreener.com/robinhood/0x32dae312abe8f6fdb782907b85edbc90d2e74b02)"
        in msg
    )
    assert "perceptor.info" in msg
    # no raw Markdown-breaking chars from free text
    assert "\n\n\n" not in msg


def test_format_perceptor_verdict_no_context() -> None:
    from fenrir.discovery.alerts import format_perceptor_verdict

    report = _oxp_report()
    assert report is not None
    msg = format_perceptor_verdict("0xabc", None, report)
    assert "*?*" in msg  # symbol fallback
    assert "`0xabc`" in msg
    assert "dexscreener.com/robinhood/0xabc" in msg


@pytest.mark.asyncio
async def test_sweep_notify_sends_once(tmp_path, monkeypatch) -> None:
    import importlib.util
    import sys
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "perceptor_tool",
        os.path.join(os.path.dirname(__file__), "..", "tools", "perceptor.py"),
    )
    assert spec is not None
    tool = importlib.util.module_from_spec(spec)
    sys.modules["perceptor_tool"] = tool
    assert spec.loader is not None
    spec.loader.exec_module(tool)

    sent: list[str] = []

    def _capture(text: str) -> bool:
        sent.append(text)
        return True

    monkeypatch.setattr(tool, "_send_telegram", _capture)

    p = PerceptorProvider(cache_path=str(tmp_path / "c.json"))
    addr = "0xbeef"
    p._load_cache()[addr] = {
        "investigation_id": "sweep1",
        "status": "pending",
        "report": None,
        "checked_at": 0.0,
        "followup_sent": False,
        "context": {"symbol": "SWP", "chain": "robinhood"},
    }
    p._save_cache()

    async def fake_refresh(a):
        # simulate the verdict landing on this sweep
        p._load_cache()[a]["status"] = "complete"
        p._load_cache()[a]["report"] = load_fixture()
        p._save_cache()
        return _oxp_report()

    monkeypatch.setattr(p, "refresh_report", fake_refresh)
    monkeypatch.setattr(tool, "PerceptorProvider", lambda *a, **k: p)

    rc = await tool.cmd_sweep(SimpleNamespace(notify=True))
    assert rc == 0
    assert len(sent) == 1
    assert "*SWP*" in sent[0]
    assert p._load_cache()[addr]["followup_sent"] is True

    # second sweep: already notified -> no duplicate send
    rc = await tool.cmd_sweep(SimpleNamespace(notify=True))
    assert rc == 0
    assert len(sent) == 1
    await p.close()


@pytest.mark.asyncio
async def test_sweep_revisits_completed_unsent_not_sent(tmp_path, monkeypatch, capsys) -> None:
    """Sweep selects pending AND completed-but-unsent entries; skips sent ones.

    Regression test for the delivery bug where completed scans flipped to
    status=="complete" without followup_sent and were never revisited.
    """
    import importlib.util
    import sys
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "perceptor_tool2",
        os.path.join(os.path.dirname(__file__), "..", "tools", "perceptor.py"),
    )
    assert spec is not None
    tool = importlib.util.module_from_spec(spec)
    sys.modules["perceptor_tool2"] = tool
    assert spec.loader is not None
    spec.loader.exec_module(tool)

    def entry(status, sent):
        return {
            "investigation_id": f"id-{status}-{sent}",
            "status": status,
            "report": load_fixture(),
            "checked_at": 0.0,
            "followup_sent": sent,
            "context": {"symbol": "T", "chain": "robinhood"},
        }

    p = PerceptorProvider(cache_path=str(tmp_path / "c.json"))
    p._load_cache()["0xaaa"] = entry("pending", False)
    p._load_cache()["0xbbb"] = entry("complete", False)  # must be revisited
    p._load_cache()["0xccc"] = entry("complete", True)  # must be skipped
    p._save_cache()

    refreshed: list[str] = []

    async def fake_refresh(a):
        refreshed.append(a)
        return _oxp_report()

    monkeypatch.setattr(p, "refresh_report", fake_refresh)
    monkeypatch.setattr(tool, "PerceptorProvider", lambda *a, **k: p)

    rc = await tool.cmd_sweep(SimpleNamespace(notify=False))
    assert rc == 0
    assert set(refreshed) == {"0xaaa", "0xbbb"}
    out = json.loads(capsys.readouterr().out)
    assert out["checked"] == 2
    assert {c["address"] for c in out["completed"]} == {"0xaaa", "0xbbb"}
    assert out["notified"] == []  # no --notify: nothing marked sent
    await p.close()


@pytest.mark.asyncio
async def test_ensure_investigation_403_cools_down_no_double_post(tmp_path) -> None:
    """A throttled POST must not be retried immediately (no double-POST)."""
    posts: list[dict | None] = []

    class FakeResp:
        status = 403

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return "rate limit exceeded"

    class FakeSession:
        closed = False

        def post(self, url, json: dict | None = None, timeout=None):
            posts.append(json)
            return FakeResp()

        async def close(self):
            pass

    p = PerceptorProvider(cache_path=str(tmp_path / "c.json"))

    async def fake_session():
        return FakeSession()

    p._get_session = fake_session  # type: ignore[method-assign]
    assert await p.ensure_investigation(4663, "0xAAA") is None
    # immediate second call (the old scout else-branch double-POST) must not
    # hit the network again: 1 initial + 1 backoff retry, then cooldown.
    assert await p.ensure_investigation(4663, "0xaaa") is None
    assert len(posts) == 2
    await p.close()


@pytest.mark.asyncio
async def test_ensure_investigation_retries_403_then_succeeds(tmp_path) -> None:
    """One retry with backoff: 403 followed by 200 returns the id."""
    calls = {"n": 0}

    class FakeResp:
        def __init__(self, status):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return "slow down"

        async def json(self):
            return {"investigation_id": "retry-ok"}

    class FakeSession:
        closed = False

        def post(self, url, json=None, timeout=None):
            calls["n"] += 1
            return FakeResp(403 if calls["n"] == 1 else 200)

        async def close(self):
            pass

    p = PerceptorProvider(cache_path=str(tmp_path / "c.json"))

    async def fake_session():
        return FakeSession()

    p._get_session = fake_session  # type: ignore[method-assign]
    assert await p.ensure_investigation(4663, "0xBBB") == "retry-ok"
    assert calls["n"] == 2
    await p.close()
