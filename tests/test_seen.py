"""Tests for fenrir.discovery.seen — case-normalized alert dedup."""

import json
from typing import Any

from fenrir.discovery import seen


def test_normalize_lowercases_and_strips():
    assert seen.normalize("0xABCdEf") == "0xabcdef"
    assert seen.normalize("  0xABC  \n") == "0xabc"


def test_should_alert_fresh_address():
    assert seen.should_alert({}, "0xabc", now=100000.0) is True


def test_should_alert_respects_window_across_casing():
    store: dict[str, dict[str, Any]] = {}
    seen.record_alert(store, "0xABCdEf", symbol="T", score=70.0, now=1000.0)
    # checksummed vs lowercase must hit the same entry
    assert seen.should_alert(store, "0xabcdef", now=1000.0 + 3600) is False
    assert seen.should_alert(store, "0xABCDEF", now=1000.0 + seen.REALERT_SECONDS + 1) is True


def test_record_alert_refreshes_score_and_keeps_first_seen():
    store: dict[str, dict[str, Any]] = {}
    seen.record_alert(store, "0xabc", symbol="T", score=60.0, now=1000.0)
    seen.record_alert(store, "0xABC", symbol="T2", score=75.0, now=2000.0)
    entry = store["0xabc"]
    assert entry["first_seen"] == 1000.0
    assert entry["last_alerted"] == 2000.0
    assert entry["score"] == 75.0
    assert entry["symbol"] == "T2"
    assert len(store) == 1  # no duplicate key from casing


def test_load_merges_legacy_mixed_case_keys(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text(
        json.dumps(
            {
                "0xABC": {"symbol": "OLD", "first_seen": 1.0, "last_alerted": 100.0, "score": 60.0},
                "0xabc": {"symbol": "NEW", "first_seen": 2.0, "last_alerted": 200.0, "score": 70.0},
            }
        )
    )
    store = seen.load(str(path))
    assert list(store.keys()) == ["0xabc"]
    assert store["0xabc"]["symbol"] == "NEW"  # freshest wins


def test_load_missing_file_returns_empty(tmp_path):
    assert seen.load(str(tmp_path / "nope.json")) == {}


def test_save_roundtrip(tmp_path):
    path = tmp_path / "seen.json"
    store: dict[str, dict[str, Any]] = {}
    seen.record_alert(store, "0xAbC", symbol="T", score=61.0, now=1234.0)
    seen.save(str(path), store)
    back = seen.load(str(path))
    assert back["0xabc"]["score"] == 61.0
