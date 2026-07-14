#!/usr/bin/env python3
"""
FENRIR - EVM discovery adapter (Ethereum / BNB / Base)

One adapter parameterized by chain — the EVM chains differ only by which
DexScreener ``chainId`` and GoPlus ``chain_id`` they use, so they share this
implementation:
  - discover(): DexScreener boosted tokens on the chain → market snapshots.
  - enrich():   GoPlus token-security → SafetySignals + holder distribution.

Base additionally surfaces Aerodrome liquidity naturally via the DexScreener
``dexId`` on the snapshot (no special-casing needed).
"""

from __future__ import annotations

import logging

from fenrir.discovery.models import Chain, TokenSnapshot
from fenrir.discovery.providers.dexscreener import DexScreenerProvider
from fenrir.discovery.providers.goplus import GoPlusProvider

logger = logging.getLogger("FENRIR.EvmAdapter")


class EvmAdapter:
    """Discovery adapter for an EVM chain (Ethereum, BNB or Base)."""

    def __init__(
        self,
        chain: Chain,
        dexscreener: DexScreenerProvider,
        goplus: GoPlusProvider,
    ) -> None:
        self.chain = chain
        self.dexscreener = dexscreener
        self.goplus = goplus

    async def discover(self) -> list[TokenSnapshot]:
        addresses = await self.dexscreener.fetch_boosted_addresses(self.chain)
        out: list[TokenSnapshot] = []
        for addr in addresses:
            snap = await self.dexscreener.fetch_snapshot(addr, self.chain)
            if snap is not None:
                out.append(snap)
        return out

    async def enrich(self, snap: TokenSnapshot) -> None:
        sec = await self.goplus.token_security(self.chain, snap.token_address)
        if sec is None:
            return
        snap.safety = sec.safety
        if sec.holder_count is not None:
            snap.holder_count = sec.holder_count
        if sec.top_holder_pct is not None:
            snap.top_holder_pct = sec.top_holder_pct
        if sec.dev_wallet_pct is not None:
            snap.dev_wallet_pct = sec.dev_wallet_pct

    async def close(self) -> None:
        # GoPlus is shared across EVM adapters; close is idempotent (guarded session).
        await self.goplus.close()
