"""Tests for fenrir.discovery.alerts — the Telegram scout alert formatter."""

from fenrir.discovery.alerts import escape_md, fmt_usd, format_scout_alert


def _cand(**kw):
    base = {
        "address": "0x9d45b233408e22cab2eead09782aec3ece263a0f",
        "chain": "robinhood",
        "symbol": "RAPE",
        "name": "Rocket Ape",
        "source": "gecko_new",
        "market_cap_usd": 5962.0,
        "liquidity_usd": 7985.35,
        "volume_24h_usd": 1321.06,
        "age_minutes": 2,
        "buys_1h": 11,
        "sells_1h": 0,
        "price_change_1h_pct": 112.0,
        "passed_filters": ["degen_launch"],
        "score": {"overall": 60.0},
        "safety_unknown": True,
        "playbooks": {
            "playbooks": [
                {
                    "strategy_id": "degen_ignition",
                    "display_name": "Degen Ignition",
                    "strength": 0.992,
                }
            ],
            "confluent": False,
        },
    }
    base.update(kw)
    return base


def test_full_alert_structure():
    msg = format_scout_alert(_cand())
    assert "🎯 *RAPE* — Rocket Ape" in msg
    assert "Robinhood · via gecko\\_new" in msg
    assert "⭐ 60.0/100 · degen\\_launch" in msg
    assert "📖 Degen Ignition (0.99)" in msg
    assert "$6.0k" in msg and "$8.0k" in msg and "$1.3k" in msg
    assert "1h 11 buys / 0 sells · +112% 1h" in msg
    assert "2m old" in msg
    assert "`0x9d45b233408e22cab2eead09782aec3ece263a0f`" in msg
    assert (
        "[DexScreener](https://dexscreener.com/robinhood/0x9d45b233408e22cab2eead09782aec3ece263a0f)"
        in msg
    )
    assert "⚠️ safety not verifiable on this chain" in msg


def test_boosted_source_omits_via_line():
    msg = format_scout_alert(_cand(source="boosted"))
    assert "via boosted" not in msg
    assert "Robinhood\n" in msg


def test_no_playbooks_omits_playbook_line():
    msg = format_scout_alert(_cand(playbooks={"playbooks": [], "confluent": False}))
    assert "📖" not in msg


def test_confluent_flag_appended():
    msg = format_scout_alert(
        _cand(
            playbooks={
                "playbooks": [{"display_name": "Momentum", "strength": 0.5}],
                "confluent": True,
            }
        )
    )
    assert "⚡confluent" in msg


def test_known_safety_omits_warning():
    msg = format_scout_alert(_cand(safety_unknown=False))
    assert "safety not verifiable" not in msg


def test_missing_flow_data():
    msg = format_scout_alert(_cand(buys_1h=None, sells_1h=None, price_change_1h_pct=None))
    assert "1h flow n/a" in msg


def test_markdown_escaping():
    msg = format_scout_alert(_cand(symbol="A_B", name="Evil *coin* [x]"))
    assert "A\\_B" in msg
    assert "Evil \\*coin\\* \\[x\\]" in msg


def test_plain_score_float():
    msg = format_scout_alert(_cand(score=72.34))
    assert "⭐ 72.3/100" in msg


def test_fmt_usd():
    assert fmt_usd(5962) == "$6.0k"
    assert fmt_usd(2_400_000) == "$2.40M"
    assert fmt_usd(850) == "$850"
    assert fmt_usd(None) == "?"


def test_escape_md_backslash_first():
    assert escape_md("a\\b_c") == "a\\\\b\\_c"
