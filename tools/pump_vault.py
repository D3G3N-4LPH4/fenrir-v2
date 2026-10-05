#!/usr/bin/env python3
"""Read a pump.fun coin's creator fee vault and express it as % of mcap.

Usage:
    python tools/pump_vault.py <mint>

Prints the creator, the vault PDA, its SOL/USD balance, whether the creator
has swept the fees, the coin's mcap (DexScreener), and vault balance as % of
mcap.

Note: creators almost always sweep to the rent minimum (~0.00065 SOL), so
expect ~0% — an unswept vault with a real balance is the interesting case.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiohttp  # noqa: E402

from fenrir.discovery.pump_vault import read_creator_vault  # noqa: E402

SOL_MINT = "So11111111111111111111111111111111111111112"


async def fetch_mcap_usd(mint: str) -> float | None:
    url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
    try:
        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=20)
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
    except Exception:
        return None
    pairs = data.get("pairs") or []
    if not pairs:
        return None
    best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd") or 0)
    mcap = best.get("fdv") or best.get("marketCap")
    return float(mcap) if mcap else None


async def fetch_sol_usd() -> float | None:
    url = f"https://api.dexscreener.com/latest/dex/tokens/{SOL_MINT}"
    try:
        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=20)
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
    except Exception:
        return None
    pairs = data.get("pairs") or []
    for p in pairs:
        try:
            # wSOL's own priceUsd when it is the base token ~= SOL/USD.
            base = (p.get("baseToken") or {}).get("address")
            if base == SOL_MINT and p.get("priceUsd"):
                return float(p["priceUsd"])
        except (TypeError, ValueError):
            continue
    return None


async def amain() -> int:
    ap = argparse.ArgumentParser(description="pump.fun creator vault reader")
    ap.add_argument("mint", help="token mint address")
    args = ap.parse_args()

    info = await read_creator_vault(args.mint)
    if info is None:
        print(f"{args.mint}: no pump.fun bonding curve found")
        return 1

    mcap, sol_usd = await asyncio.gather(fetch_mcap_usd(args.mint), fetch_sol_usd())

    print(f"mint:    {info['mint']}")
    print(f"creator: {info['creator']}")
    print(f"vault:   {info['vault']}")
    if sol_usd:
        vault_usd = info["sol"] * sol_usd
        print(f"balance: {info['sol']:.6f} SOL (${vault_usd:,.2f})")
    else:
        vault_usd = None
        print(f"balance: {info['sol']:.6f} SOL (SOL/USD unavailable)")
    print(
        "swept:   "
        + (
            "yes — fees collected, rent residue only"
            if info["swept"]
            else "NO — live uncollected fees"
        )
    )
    if mcap:
        print(f"mcap:    ${mcap:,.0f} (DexScreener)")
    if mcap and vault_usd is not None:
        print(f"vault/mcap: {vault_usd / mcap * 100:.4f}%")
    elif mcap:
        print("vault/mcap: n/a (no SOL price)")
    else:
        print("vault/mcap: n/a (no mcap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
