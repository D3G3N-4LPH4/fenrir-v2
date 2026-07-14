#!/usr/bin/env python3
"""Per-chain discovery adapters. Chain specifics live ONLY here."""

from __future__ import annotations

from fenrir.discovery.chains.base import ChainAdapter
from fenrir.discovery.chains.solana import SolanaAdapter

__all__ = ["ChainAdapter", "SolanaAdapter"]
