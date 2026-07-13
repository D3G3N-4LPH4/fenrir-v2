#!/usr/bin/env python3
"""Tests for the non-tradeable mint filter (stablecoins / WSOL / LSTs)."""

from __future__ import annotations

from fenrir.trading.token_filters import NON_TRADEABLE_MINTS, is_tradeable_mint

USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
WSOL = "So11111111111111111111111111111111111111112"
JITOSOL = "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn"
A_PUMP_TOKEN = "Ew1oPyPx6LL8fiwXpUon6Q8t5Vi1FiMQmfTLketCWigG"


def test_stablecoins_and_wsol_not_tradeable() -> None:
    assert is_tradeable_mint(USDC) is False
    assert is_tradeable_mint(USDT) is False
    assert is_tradeable_mint(WSOL) is False
    assert is_tradeable_mint(JITOSOL) is False


def test_regular_token_is_tradeable() -> None:
    assert is_tradeable_mint(A_PUMP_TOKEN) is True


def test_empty_or_none_not_tradeable() -> None:
    assert is_tradeable_mint(None) is False
    assert is_tradeable_mint("") is False


def test_blocklist_contains_core_stables() -> None:
    assert {USDC, USDT, WSOL} <= NON_TRADEABLE_MINTS
