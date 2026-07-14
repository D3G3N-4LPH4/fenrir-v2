#!/usr/bin/env python3
"""
FENRIR - Chain adapter protocol

Every chain implements this small surface. ``discover`` returns candidate
snapshots (from the chain's trending/new-pool feed); ``enrich`` fills in market +
safety fields the discovery feed didn't provide. All chain-specific logic (which
provider, which endpoints, how safety maps) lives inside the adapter — the filter
and scoring engines only ever see the resulting :class:`TokenSnapshot`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fenrir.discovery.models import Chain, TokenSnapshot


@runtime_checkable
class ChainAdapter(Protocol):
    """Discovery adapter for one chain."""

    chain: Chain

    async def discover(self) -> list[TokenSnapshot]:
        """Return candidate snapshots from the chain's trending / new-pool feed."""
        ...

    async def enrich(self, snap: TokenSnapshot) -> None:
        """Fill missing market + safety fields on ``snap`` in place (best-effort)."""
        ...

    async def close(self) -> None:
        """Release any provider sessions held by the adapter."""
        ...
