#!/usr/bin/env python3
"""
FENRIR - Non-tradeable mint filter

Stablecoins, wrapped SOL and major liquid-staking tokens are not degen swing
targets: they don't trend, and a smart-money wallet "buying" one is almost always
a swap leg or a receipt, not a conviction entry. The scanner and smart-money
tracker skip them so they never reach the AI / buy path.

(A stablecoin buy also exposed the Jupiter-buy entry-price scale bug live —
USDC "dropped" 88% in seconds — so filtering them is both a quality and a safety
measure.)
"""

from __future__ import annotations

WSOL_MINT = "So11111111111111111111111111111111111111112"

# Stablecoins (USD + EUR), wrapped SOL, and the major Solana liquid-staking
# tokens. Kept as a frozenset for O(1) membership checks.
NON_TRADEABLE_MINTS: frozenset[str] = frozenset(
    {
        WSOL_MINT,
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
        "USDSwr9ApdHk5bvJKMjzff41FfuX8bSxdKcR81vTwcA",  # USDS
        "2u1tszSeqZ3qBWF3uNGPFc8TzMk2tdiwknnRMWGWjGWH",  # USDG
        "9zNQRsGLjNKwCUU5Gq5LR8beUCPzQMVMqKAi3SSZh54u",  # EURC
        "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",  # mSOL
        "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",  # stSOL
        "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn",  # jitoSOL
        "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1",  # bSOL
        "So11111111111111111111111111111111111111112",  # (WSOL, explicit)
    }
)


def is_tradeable_mint(mint: str | None) -> bool:
    """False for stablecoins / WSOL / major LSTs, which are not swing targets."""
    return bool(mint) and mint not in NON_TRADEABLE_MINTS
