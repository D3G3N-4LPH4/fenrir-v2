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
from fenrir.discovery.providers.goplus import GoPlusProvider, distribution_metrics

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
        # Wallet concentration: distribution_metrics drops AMM infrastructure
        # (contracts — v2/v3 pools, the v4 PoolManager — plus locked supply),
        # leaving EOA-held supply, the actual dump risk. top10 doubles as the
        # concentration/bundle proxy. pair_address is passed as a fallback for
        # chains where GoPlus omits the is_contract flag.
        top, top10 = distribution_metrics(sec.holders, {snap.pair_address or ""})
        snap.top_holder_pct = top if top is not None else sec.top_holder_pct
        snap.top10_holder_pct = top10
        if sec.dev_wallet_pct is not None:
            snap.dev_wallet_pct = sec.dev_wallet_pct

    async def close(self) -> None:
        # GoPlus is shared across EVM adapters; close is idempotent (guarded session).
        await self.goplus.close()
