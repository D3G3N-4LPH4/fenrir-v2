#!/usr/bin/env python3
"""
FENRIR - Migration Detector (pump.fun → Raydium)

Helpers for the *experimental* migration feed: recognising a pump.fun →
Raydium graduation from transaction logs and turning it into the ``token_data``
dict the pipeline consumes (so the ``migration_snipe`` strategy can act on it).

Scope / caveats:
  - ``has_migration_hint`` and ``build_token_data`` are pure and unit-tested.
  - Extracting the exact token mint from a real migration transaction depends
    on live account layouts and is handled in the monitor's live loop; that
    path is NOT verifiable offline and is gated behind
    ``BotConfig.migration_feed_enabled`` (off by default).

Downstream note: a migrated token trades on Raydium, so the market-data
snapshot (DexScreener via ``fenrir.filters``) supplies ``dex_id="raydium"``,
age, liquidity, etc. This builder therefore only needs to surface the token
address (and light metadata) quickly; the MarketFilter + the strategy's
``evaluate_token`` do the real gating.
"""

from __future__ import annotations

from typing import Any


class MigrationDetector:
    """Recognise pump.fun → Raydium migrations from logs and build token_data."""

    # pump.fun migration authority — the targeted logsSubscribe address.
    PUMP_MIGRATION_PROGRAM = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
    # Raydium AMM v4 — the destination pool program (reference only).
    RAYDIUM_AMM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

    # Log fragments that indicate a graduation/pool-open. Kept broad on purpose;
    # the monitor confirms by fetching and parsing the transaction.
    _HINTS: tuple[str, ...] = (
        "Instruction: Migrate",
        "Program log: Migrate",
        "Instruction: Withdraw",  # pump migration withdraws curve liquidity
        "initialize2",  # Raydium AMM pool initialization
    )

    def has_migration_hint(self, logs: list[str]) -> bool:
        """True if any log line looks like a pump→Raydium migration/pool-open."""
        return any(any(hint in log for hint in self._HINTS) for log in logs)

    def build_token_data(
        self,
        token_mint: str,
        *,
        symbol: str = "???",
        name: str = "Migrated (pump→Raydium)",
        pair_address: str | None = None,
    ) -> dict[str, Any]:
        """
        Build the token_data dict for a migrated token.

        No bonding_curve_state is attached (the curve is complete post-migration);
        pricing/market context comes from the DexScreener MarketData snapshot.
        """
        return {
            "token_address": token_mint,
            "symbol": symbol,
            "name": name,
            "dex_id": "raydium",
            "pair_address": pair_address,
            "migrated": True,
        }
