"""Tests for the PVP set view in tools/coin_watch.py.

Covers the PVP correction: timestamp-bucket alignment (not ordinal index),
cross-venue volume/transaction aggregation without double-counting, gap
handling without interpolation, and the --set-name CLI/docstring agreement.
"""

import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "coin_watch",
    os.path.join(os.path.dirname(__file__), "..", "tools", "coin_watch.py"),
)
assert SPEC is not None and SPEC.loader is not None
cw = importlib.util.module_from_spec(SPEC)
sys.modules["coin_watch"] = cw
SPEC.loader.exec_module(cw)


# ---------------------------------------------------------------------------
# venue aggregation


def _pair(chain="solana", pair="pairA", liq=1000.0, m5=100.0, buys=10, sells=8):
    return {
        "chainId": chain,
        "pairAddress": pair,
        "dexId": "raydium",
        "priceUsd": "0.001",
        "marketCap": 500000,
        "liquidity": {"usd": liq},
        "priceChange": {"m5": 1.0, "h1": 2.0, "h24": 3.0},
        "volume": {"m5": m5, "h1": m5 * 10, "h24": m5 * 100},
        "txns": {
            "m5": {"buys": buys, "sells": sells},
            "h1": {"buys": buys * 5, "sells": sells * 5},
        },
    }


def test_aggregate_pairs_sums_venues():
    pairs = [
        _pair(pair="pairA", liq=1000.0, m5=100.0, buys=10, sells=8),
        _pair(pair="pairB", liq=500.0, m5=250.0, buys=30, sells=12),
    ]
    agg, venues = cw._aggregate_pairs(pairs)
    assert venues == 2
    assert agg["vol_5m"] == 350.0
    assert agg["buys_5m"] == 40
    assert agg["sells_5m"] == 20
    assert agg["vol_1h"] == 3500.0
    assert agg["buys_1h"] == 200


def test_aggregate_pairs_dedupes_same_pool():
    pairs = [
        _pair(pair="pairA", liq=1000.0, m5=100.0, buys=10, sells=8),
        _pair(pair="pairA", liq=999.0, m5=100.0, buys=10, sells=8),  # stale duplicate listing
    ]
    agg, venues = cw._aggregate_pairs(pairs)
    assert venues == 1
    assert agg["vol_5m"] == 100.0
    assert agg["buys_5m"] == 10


def test_aggregate_pairs_tolerates_missing_fields():
    agg, venues = cw._aggregate_pairs([{"chainId": "solana", "pairAddress": "x"}])
    assert venues == 1
    assert agg["vol_5m"] == 0.0
    assert agg["buys_5m"] == 0


def _fake_curl(payload):
    class R:
        stdout = json.dumps(payload)

    def run(cmd, **kw):
        return R()

    return run


def test_fetch_snapshot_aggregates_volume_but_prices_from_deepest(monkeypatch):
    payload = {
        "pairs": [
            _pair(pair="shallow", liq=500.0, m5=900.0, buys=90, sells=80),
            _pair(pair="deep", liq=5000.0, m5=100.0, buys=10, sells=8),
        ]
    }
    monkeypatch.setattr(cw.subprocess, "run", _fake_curl(payload))
    snap = cw.fetch_snapshot("0xabc")
    assert snap is not None
    # volume summed across both venues
    assert snap["vol_5m"] == 1000.0
    assert snap["buys_5m"] == 100
    assert snap["sells_5m"] == 88
    assert snap["venues"] == 2
    assert snap["vol_agg"] is True
    # price/mcap/liq still from the deepest pool
    assert snap["pair"] == "deep"
    assert snap["liq"] == 5000.0


def test_fetch_snapshot_returns_none_without_pairs(monkeypatch):
    monkeypatch.setattr(cw.subprocess, "run", _fake_curl({"pairs": []}))
    assert cw.fetch_snapshot("0xabc") is None


# ---------------------------------------------------------------------------
# timestamp buckets


def test_bucket_key_tolerates_tick_drift():
    # ticks ~40s apart (away from a boundary) land in the same 2-minute bucket
    assert cw._bucket_key(960.0) == cw._bucket_key(1000.0)
    # ticks ~3 minutes apart do not
    assert cw._bucket_key(960.0) != cw._bucket_key(1140.0)


def test_bucketize_latest_point_wins():
    series = [
        {"ts": 960.0, "mcap": 1},
        {"ts": 1000.0, "mcap": 2},  # same bucket, later -> wins
        {"ts": 1120.0, "mcap": 3},  # next bucket
    ]
    b = cw._bucketize(series)
    assert len(b) == 2
    assert b[cw._bucket_key(960.0)]["mcap"] == 2
    assert b[cw._bucket_key(1120.0)]["mcap"] == 3


def test_bucketize_skips_points_without_ts():
    b = cw._bucketize([{"mcap": 1}, {"ts": 1000.0, "mcap": 2}])
    assert len(b) == 1


# ---------------------------------------------------------------------------
# cmd_pvp end to end


def _coin(label, set_name, points):
    return {
        "label": label,
        "set": set_name,
        "series": [
            {
                "ts": ts,
                "mcap": mcap,
                "vol_5m": vol,
                "buys_5m": buys,
                "sells_5m": sells,
                "chg_5m": 1.0,
            }
            for ts, mcap, vol, buys, sells in points
        ],
    }


def test_pvp_aligns_members_by_timestamp_not_index(monkeypatch):
    # member B was added two buckets later than member A; ordinal-index
    # alignment would compare A's old points against B's new ones.
    state = {
        "coins": {
            "a": _coin(
                "AAA",
                "si-pvp",
                [(1000.0, 500_000, 1000, 10, 8), (1120.0, 400_000, 2000, 20, 5)],
            ),
            "b": _coin(
                "BBB",
                "si-pvp",
                [(1240.0, 600_000, 3000, 30, 10), (1360.0, 650_000, 1000, 5, 15)],
            ),
        }
    }
    monkeypatch.setattr(cw, "load_state", lambda: state)
    out = io.StringIO()
    with redirect_stdout(out):
        rc = cw.cmd_pvp(SimpleNamespace(set_name="si-pvp", last=10))
    assert rc == 0
    text = out.getvalue()
    # buckets 1000..1360 -> 4 buckets; the first two have only AAA, last two only BBB
    assert text.count("(no tick:") == 4
    # volume share is computed per bucket over present members only
    assert "share 100.0%" in text


def test_pvp_shows_gap_without_interpolation(monkeypatch):
    # AAA misses the middle bucket; its values must not leak into BBB's row
    # and the missing row must show a dash, not invented data.
    state = {
        "coins": {
            "a": _coin(
                "AAA", "si-pvp", [(1000.0, 500_000, 1000, 10, 8), (1240.0, 400_000, 2000, 20, 5)]
            ),
            "b": _coin(
                "BBB",
                "si-pvp",
                [
                    (1000.0, 100_000, 500, 5, 5),
                    (1120.0, 100_000, 500, 5, 5),
                    (1240.0, 100_000, 500, 5, 5),
                ],
            ),
        }
    }
    monkeypatch.setattr(cw, "load_state", lambda: state)
    out = io.StringIO()
    with redirect_stdout(out):
        rc = cw.cmd_pvp(SimpleNamespace(set_name="si-pvp", last=10))
    assert rc == 0
    text = out.getvalue()
    assert "(no tick: AAA)" in text
    aaa_rows = [ln for ln in text.splitlines() if ln.strip().startswith("AAA")]
    # AAA has 2 real points + 1 gap row (dash) + 1 session row
    assert sum(1 for ln in aaa_rows if ln.strip().endswith("—")) == 1


def test_pvp_empty_set(monkeypatch):
    monkeypatch.setattr(cw, "load_state", lambda: {"coins": {}})
    out = io.StringIO()
    with redirect_stdout(out):
        rc = cw.cmd_pvp(SimpleNamespace(set_name="nope", last=10))
    assert rc == 1
    assert "no coins in set" in out.getvalue()


def test_tag_parser_uses_set_name_and_docstring_agrees():
    # regression: usage comment once said --set while the parser used --set-name
    assert "--set-name" in (cw.__doc__ or "")
    assert "--set si-pvp" not in (cw.__doc__ or "")
