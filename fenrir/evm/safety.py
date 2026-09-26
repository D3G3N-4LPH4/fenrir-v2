#!/usr/bin/env python3
"""
FENRIR - EVM safety hard-gate (Phase 7.3, read-only)

The EVM analogue of the Solana security filter. EVM is dense with honeypots (tokens you
can buy but not sell) and punitive transfer taxes, so before an EVM token is ever flagged
by a strategy it must clear a safety gate over the provider (GoPlus) ``SafetySignals``
already carried on the snapshot: honeypot, buy/sell tax, blacklist, and (optionally) LP
lock.

Pure policy — no network. Unknown signals (``None``) are treated per ``fail_open``: with
fail-open (the default) an unknown does NOT reject (we only reject on an EXPLICIT bad
signal), matching the RugCheck fail-open philosophy; fail-closed rejects on unknown for
the checks that are enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvmSafetyConfig:
    """Policy for the EVM safety gate. Enabled by default (it only ever removes unsafe
    tokens from a read-only surface)."""

    enabled: bool = True
    reject_honeypot: bool = True
    reject_blacklist: bool = True
    max_buy_tax_pct: float = 10.0
    max_sell_tax_pct: float = 10.0
    require_lp_locked: bool = False  # many legit EVM tokens vary; off by default
    # Unknown (None) signals: fail-open = don't reject on missing data (default);
    # fail-closed = reject when an enabled check has no data.
    fail_open: bool = True


@dataclass
class SafetyVerdict:
    passed: bool
    reason: str = ""


class EvmSafetyGate:
    """Read-only pass/fail policy over a snapshot's ``SafetySignals``."""

    def __init__(self, config: EvmSafetyConfig | None = None) -> None:
        self.config = config or EvmSafetyConfig()

    def _unknown_fails(self) -> bool:
        return not self.config.fail_open

    def check(self, safety: Any) -> SafetyVerdict:
        """Return whether the token clears the gate. ``safety`` is a SafetySignals (or
        anything exposing the same attributes); None → treated as all-unknown."""
        cfg = self.config
        if not cfg.enabled:
            return SafetyVerdict(True, "gate disabled")

        honeypot = getattr(safety, "honeypot", None)
        blacklist = getattr(safety, "blacklist_present", None)
        buy_tax = getattr(safety, "buy_tax_pct", None)
        sell_tax = getattr(safety, "sell_tax_pct", None)
        lp_locked = getattr(safety, "lp_locked_or_burned", None)

        # Honeypot — the cardinal EVM risk (can't sell).
        if cfg.reject_honeypot:
            if honeypot is True:
                return SafetyVerdict(False, "honeypot")
            if honeypot is None and self._unknown_fails():
                return SafetyVerdict(False, "honeypot unknown (fail-closed)")

        # Blacklist capability (contract can freeze your wallet).
        if cfg.reject_blacklist:
            if blacklist is True:
                return SafetyVerdict(False, "blacklist capability")
            if blacklist is None and self._unknown_fails():
                return SafetyVerdict(False, "blacklist unknown (fail-closed)")

        # Transfer taxes.
        if buy_tax is not None and buy_tax > cfg.max_buy_tax_pct:
            return SafetyVerdict(False, f"buy tax {buy_tax:.1f}% > {cfg.max_buy_tax_pct:.0f}%")
        if buy_tax is None and self._unknown_fails():
            return SafetyVerdict(False, "buy tax unknown (fail-closed)")
        if sell_tax is not None and sell_tax > cfg.max_sell_tax_pct:
            return SafetyVerdict(False, f"sell tax {sell_tax:.1f}% > {cfg.max_sell_tax_pct:.0f}%")
        if sell_tax is None and self._unknown_fails():
            return SafetyVerdict(False, "sell tax unknown (fail-closed)")

        # LP lock (optional).
        if cfg.require_lp_locked:
            if lp_locked is False:
                return SafetyVerdict(False, "LP not locked/burned")
            if lp_locked is None and self._unknown_fails():
                return SafetyVerdict(False, "LP lock unknown (fail-closed)")

        return SafetyVerdict(True, "ok")
