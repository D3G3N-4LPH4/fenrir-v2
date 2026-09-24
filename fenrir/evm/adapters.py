#!/usr/bin/env python3
"""
FENRIR - EVM snapshot adapters (on-chain EVM, read-only)

Bridge the multi-chain discovery ``TokenSnapshot`` (ETH/BNB/Base, from DexScreener +
GoPlus) onto the exact shapes the existing strategies and AI brain already consume —
``MarketData`` (for the signal strategies' ``evaluate_token``) and a ``token_data`` dict
(for the AI decision context). This lets an EVM token flow through the full strategy /
signal / brain machinery UNCHANGED, with no chain leakage into those components.

Pure and read-only: field mapping only, no network, no execution.
"""

from __future__ import annotations

from typing import Any

from fenrir.filters import MarketData


def _buy_pressure(buys: int, sells: int) -> float:
    total = buys + sells
    return buys / total if total > 0 else 0.5


def snapshot_to_market_data(snapshot: Any) -> MarketData:
    """Map an EVM ``TokenSnapshot`` to the ``MarketData`` the signal strategies expect.

    ``price_sol`` is 0 (EVM tokens are USD/quote-denominated, not SOL), and the fields
    DexScreener does not provide (6h price change, unique 1h buyers) default to 0 — the
    strategies read them defensively.
    """
    return MarketData(
        token_address=snapshot.token_address,
        pair_address=snapshot.pair_address,
        dex_id=snapshot.dex_id,
        price_usd=snapshot.price_usd,
        price_sol=0.0,
        liquidity_usd=snapshot.liquidity_usd,
        market_cap_usd=snapshot.market_cap_usd,
        fdv_usd=snapshot.fdv_usd,
        volume_5m_usd=snapshot.volume_5m_usd,
        volume_1h_usd=snapshot.volume_1h_usd,
        volume_6h_usd=snapshot.volume_6h_usd,
        volume_24h_usd=snapshot.volume_24h_usd,
        txns_5m_buys=snapshot.txns_5m_buys,
        txns_5m_sells=snapshot.txns_5m_sells,
        txns_1h_buys=snapshot.txns_1h_buys,
        txns_1h_sells=snapshot.txns_1h_sells,
        unique_buyers_1h=0,
        price_change_5m_pct=snapshot.price_change_5m_pct,
        price_change_1h_pct=snapshot.price_change_1h_pct,
        price_change_6h_pct=0.0,
        price_change_24h_pct=snapshot.price_change_24h_pct,
        age_minutes=snapshot.age_minutes,
        raw=snapshot.raw,
    )


def snapshot_to_token_data(snapshot: Any) -> dict[str, Any]:
    """Map an EVM ``TokenSnapshot`` to the ``token_data`` dict the AI brain + decision
    context read, mirroring the momentum enrichment the Solana ``_scan_and_route`` adds,
    plus the EVM safety signals (honeypot / taxes) and holder distribution.
    """
    safety = getattr(snapshot, "safety", None)
    return {
        "token_address": snapshot.token_address,
        "symbol": snapshot.symbol,
        "name": snapshot.name,
        "chain": snapshot.chain.value,
        "price_usd": snapshot.price_usd,
        "market_cap_usd": snapshot.market_cap_usd,
        "liquidity_usd": snapshot.liquidity_usd,
        # DexScreener momentum, surfaced for the AI decision context (Solana parity).
        "dex_volume_5m_usd": snapshot.volume_5m_usd,
        "dex_txns_5m_buys": snapshot.txns_5m_buys,
        "dex_txns_5m_sells": snapshot.txns_5m_sells,
        "dex_buy_pressure_5m": _buy_pressure(snapshot.txns_5m_buys, snapshot.txns_5m_sells),
        "dex_price_change_1h_pct": snapshot.price_change_1h_pct,
        "dex_liquidity_usd": snapshot.liquidity_usd,
        # EVM safety (the analogue of Solana RugCheck) — read for the AI's risk context.
        "honeypot": getattr(safety, "honeypot", None),
        "buy_tax_pct": getattr(safety, "buy_tax_pct", None),
        "sell_tax_pct": getattr(safety, "sell_tax_pct", None),
        "holder_count": snapshot.holder_count,
        "top_holder_pct": snapshot.top_holder_pct,
    }
