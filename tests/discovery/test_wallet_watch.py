#!/usr/bin/env python3
"""Tests for the wallet watcher's buy detection (tools/wallet_watch)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fenrir.discovery.alerts import format_scout_alert  # noqa: E402
from tools.wallet_watch import (  # noqa: E402
    TRANSFER_TOPIC0,
    _pad_topic,
    detect_robinhood_buys,
    detect_solana_buys,
)

WALLET = "Wallet1111111111111111111111111111111111111"
MINT = "Mint11111111111111111111111111111111111111111"
WSOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def _bal(owner, mint, ui_amount):
    return {"owner": owner, "mint": mint, "uiTokenAmount": {"uiAmount": ui_amount}}


def _tx(pre_balances, post_balances, pre_lamports=(100, 100), post_lamports=(99, 100)):
    return {
        "meta": {
            "preTokenBalances": pre_balances,
            "postTokenBalances": post_balances,
            "preBalances": list(pre_lamports),
            "postBalances": list(post_lamports),
        },
        "transaction": {"message": {"accountKeys": [WALLET, "Other111"]}},
    }


def test_solana_buy_detected():
    tx = _tx([_bal(WALLET, MINT, 0.0)], [_bal(WALLET, MINT, 1000.0)])
    buys = detect_solana_buys(tx, WALLET)
    assert len(buys) == 1
    assert buys[0][0] == MINT
    # 1 lamport-ish unit spent (100 -> 99 in test lamports is huge; use real scale)
    assert buys[0][1] >= 0.0


def test_solana_buy_sol_spent():
    tx = _tx(
        [_bal(WALLET, MINT, 0.0)],
        [_bal(WALLET, MINT, 5.0)],
        pre_lamports=(2_000_000_000, 0),
        post_lamports=(1_500_000_000, 0),
    )
    buys = detect_solana_buys(tx, WALLET)
    assert buys[0][1] == pytest.approx(0.5)


def test_solana_wsol_increase_ignored():
    tx = _tx([_bal(WALLET, WSOL, 1.0)], [_bal(WALLET, WSOL, 5.0)])
    assert detect_solana_buys(tx, WALLET) == []


def test_solana_stable_increase_ignored():
    tx = _tx([_bal(WALLET, USDC, 0.0)], [_bal(WALLET, USDC, 500.0)])
    assert detect_solana_buys(tx, WALLET) == []


def test_solana_no_increase_no_buy():
    tx = _tx([_bal(WALLET, MINT, 100.0)], [_bal(WALLET, MINT, 100.0)])
    assert detect_solana_buys(tx, WALLET) == []


def test_solana_other_owner_ignored():
    tx = _tx([], [_bal("SomeoneElse", MINT, 1000.0)])
    assert detect_solana_buys(tx, WALLET) == []


def test_solana_missing_meta():
    assert detect_solana_buys({}, WALLET) == []
    assert detect_solana_buys({"meta": None}, WALLET) == []


def test_robinhood_buy_detected():
    token = "0x" + "ab" * 20
    logs = [
        {
            "address": token,
            "topics": [
                TRANSFER_TOPIC0,
                _pad_topic("0x" + "11" * 20),
                _pad_topic("0x" + "22" * 20),
            ],
        }
    ]
    assert detect_robinhood_buys(logs) == [token]


def test_robinhood_base_currencies_skipped():
    from tools.wallet_watch import USDG_ROBINHOOD, WETH_ROBINHOOD

    logs = [
        {"address": WETH_ROBINHOOD, "topics": []},
        {"address": USDG_ROBINHOOD, "topics": []},
        {"address": "0x" + "0" * 40, "topics": []},
    ]
    assert detect_robinhood_buys(logs) == []


def test_robinhood_dedup():
    token = "0x" + "cd" * 20
    logs = [
        {"address": token, "topics": []},
        {"address": token, "topics": []},
        {"address": token.upper(), "topics": []},
    ]
    assert detect_robinhood_buys(logs) == [token]


def test_pad_topic():
    assert _pad_topic("0xAbC") == "0x" + "0" * 24 + "abc"
    assert len(_pad_topic("0x" + "ff" * 20)) == 66


def _cand():
    return {
        "symbol": "TEST",
        "name": "Test Token",
        "chain": "solana",
        "source": "wallet:smart-1",
        "address": "SomeAddr",
        "score": {"overall": 75.0},
        "passed_filters": ["degen_launch"],
        "market_cap_usd": 10000,
        "liquidity_usd": 5000,
        "volume_24h_usd": 20000,
        "buys_1h": 10,
        "sells_1h": 2,
        "price_change_1h_pct": 25.0,
        "age_minutes": 12,
        "wallet_buys": [
            {"label": "smart-1", "sol_spent": 1.5},
            {"label": "smart-2", "sol_spent": 0.5},
        ],
    }


def test_alert_wallet_line():
    text = format_scout_alert(_cand())
    assert "wallet buy:" in text
    assert "smart-1 + smart-2" in text
    assert "2.00 SOL" in text
    assert "via wallet:smart-1" in text


def test_alert_no_wallet_line_when_absent():
    cand = _cand()
    del cand["wallet_buys"]
    assert "wallet buy:" not in format_scout_alert(cand)
