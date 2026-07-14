#!/usr/bin/env python3
"""
FENRIR - Multi-chain discovery package.

A chain-agnostic discovery layer that scans, filters and scores meme coins across
Solana, Ethereum, BNB and Base. Chain specifics live behind adapters
(``discovery/chains``) and providers (``discovery/providers``); the filter engine
(``discovery/filters``) and scoring engine (``discovery/scoring``) operate ONLY on
the normalized ``TokenSnapshot`` (``discovery/models``), so no chain-specific logic
leaks into the shared components.

Discovery is separate from execution: it surfaces + scores + alerts on candidates
across all four chains, while trade execution stays Solana-only (the existing
``fenrir/trading`` engine). Everything here is opt-in and off by default.
"""

from __future__ import annotations

from fenrir.discovery.models import (
    Chain,
    FilterResult,
    SafetySignals,
    ScoreBreakdown,
    TokenSnapshot,
)

# NOTE: the scanner/chains stack (DiscoveryScanner, build_adapters) is intentionally
# NOT re-exported here — importing a discovery submodule (e.g. the DexScreener
# provider, used by fenrir.filters.market) should stay lightweight. Import the
# service directly: ``from fenrir.discovery.scanner import DiscoveryScanner``.

__all__ = [
    "Chain",
    "FilterResult",
    "SafetySignals",
    "ScoreBreakdown",
    "TokenSnapshot",
]
