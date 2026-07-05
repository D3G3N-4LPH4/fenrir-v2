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

# Wrapped SOL — filtered out when identifying the migrated token mint.
WSOL_MINT = "So11111111111111111111111111111111111111112"


class MigrationDetector:
    """Recognise pump.fun → Raydium migrations from logs and build token_data."""

    # pump.fun migration authority — the targeted logsSubscribe address.
    PUMP_MIGRATION_PROGRAM = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
    # Raydium AMM v4 — the destination pool program (reference only).
    RAYDIUM_AMM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
    # pump.fun mint vanity suffix — used to disambiguate the token mint from an
    # LP mint when a migration tx touches more than one non-WSOL mint.
    PUMP_MINT_SUFFIX = "pump"

    # Log fragments that identify a migration. Validated against live txs on the
    # migration authority: real graduations emit "Instruction: MigrateV2" (older
    # ones "Migrate") — "Instruction: Migrate" matches both as a substring.
    # Deliberately narrow: broader guesses (Withdraw / initialize2 / mint init)
    # never appeared on real migrations and would false-positive on unrelated
    # txs that merely mention the authority.
    _HINTS: tuple[str, ...] = (
        "Instruction: Migrate",
        "Program log: Migrate",
    )

    def has_migration_hint(self, logs: list[str]) -> bool:
        """True if any log line looks like a pump→Raydium migration."""
        return any(any(hint in log for hint in self._HINTS) for log in logs)

    def pick_token_mint(self, mints: list[str]) -> str | None:
        """
        Choose the migrated token mint from the mints a tx touches.

        Verified on real MigrateV2 txs: exactly one non-WSOL mint, which is the
        token. Defensive on the multi-mint edge case (e.g. an LP mint also
        present): pick the unique pump-suffixed mint, else give up (None) rather
        than guess.
        """
        candidates = [m for m in dict.fromkeys(mints) if m and m != WSOL_MINT]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        pump = [m for m in candidates if m.endswith(self.PUMP_MINT_SUFFIX)]
        return pump[0] if len(pump) == 1 else None

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
