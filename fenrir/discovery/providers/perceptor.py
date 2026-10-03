#!/usr/bin/env python3
"""
FENRIR - Perceptor safety provider (Robinhood Chain forensics)

0xPerceptor (perceptor.info) is a read-only token-forensics service for
Robinhood Chain (chain id 4663): liquidity control (burned/locked/pullable),
dev-wallet behavior, holder concentration, launch snipers/bundles, buyer
outcomes, and contract powers (mint/blacklist/pause) plus measured buy/sell
tax. It closes the "safety not verifiable on this chain" gap where GoPlus
has no result for a brand-new token.

Backend is an open FastAPI with no auth:
  POST /api/investigations  {chain_id, address} -> investigation id
  GET  /api/investigations/{id}                -> full result (status + verdict)

Investigations are heavy server-side (full chain scan), so be polite:
  - only investigate tokens that cleared FENRIR's entry filters (a few/day),
  - POST once per address ever (investigation id cached on disk),
  - the POST endpoint rate-limits (observed as intermittent HTTP 403s): the
    provider backs off with jitter, cools down throttled addresses, and
    always fails open,
  - rely on server-side caching (fresh=false default); verdicts are pinned to
    a block, so a completed report never goes stale.

The pure ``parse_perceptor`` mapper is unit-testable without network.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from fenrir.discovery.models import Chain, SafetySignals, TokenSnapshot

logger = logging.getLogger("FENRIR.Perceptor")

PERCEPTOR_API = "https://www.perceptor.info/api"
ROBINHOOD_CHAIN_ID = 4663

DEFAULT_CACHE_PATH = os.path.expanduser("~/.cache/fenrir/perceptor.json")

_BAND_RISK = {"low": 15.0, "medium": 50.0, "high": 85.0}


@dataclass
class PerceptorReport:
    """Parsed Perceptor verdict: safety signals + human summary."""

    safety: SafetySignals = field(default_factory=SafetySignals)
    band: str | None = None  # low | medium | high
    band_label: str | None = None  # e.g. "Caution"
    headline: str | None = None
    checks: list[tuple[str, str]] = field(default_factory=list)
    signals: list[tuple[str, str]] = field(default_factory=list)  # (label, tone)
    investigation_id: str | None = None


def _f(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def snapshot_context(snap: TokenSnapshot) -> dict[str, Any]:
    """Small token dict stored with an investigation for later follow-ups."""
    chain = snap.chain.value if isinstance(snap.chain, Chain) else None
    return {
        "symbol": snap.symbol,
        "name": snap.name,
        "chain": chain,
        "dexscreener": (f"https://dexscreener.com/{chain}/{snap.token_address}" if chain else None),
    }


def parse_perceptor(data: Any) -> PerceptorReport | None:
    """Map a full Perceptor investigation result to safety signals (pure).

    Accepts arbitrary input and returns None for anything that isn't a
    well-formed investigation dict (the API is external and untrusted).
    """
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    if not isinstance(verdict, dict):
        return None

    safety = SafetySignals()

    lc = verdict.get("liquidity_control") or {}
    if lc.get("measured"):
        pct = _f(lc.get("safe"))
        if pct is None:
            pct = _f(lc.get("permanent"))
        if pct is not None:
            safety.lp_locked_pct = round(pct, 2)
            safety.lp_locked_or_burned = pct >= 90.0

    contract = verdict.get("contract") or {}
    if contract.get("measured"):
        powers = {str(p).lower() for p in (contract.get("powers") or [])}
        owner_status = str(contract.get("owner_status") or "").lower()
        if owner_status in ("none", "renounced", ""):
            safety.ownership_renounced = True
        elif owner_status:
            safety.ownership_renounced = False
        if powers:
            safety.mint_disabled = "mint" not in powers
            safety.blacklist_present = "blacklist" in powers
            safety.freeze_disabled = not (powers & {"pause", "pausable", "freeze"})
        else:
            safety.mint_disabled = True
            safety.blacklist_present = False
            safety.freeze_disabled = True

    tax = verdict.get("tax") or {}
    if tax.get("measured"):
        safety.buy_tax_pct = _f(tax.get("buy"))
        safety.sell_tax_pct = _f(tax.get("sell"))

    # "Selling works" check or observed non-dev sellers => not a honeypot.
    checks = [
        (c.get("label"), c.get("value"))
        for c in (verdict.get("checks") or [])
        if isinstance(c, dict)
    ]
    trading = verdict.get("trading_now") or {}
    sellers = _f(trading.get("sellers_not_dev")) or _f(trading.get("sellers"))
    if any(label == "Selling" and str(value).lower() == "works" for label, value in checks):
        safety.honeypot = False
    elif sellers is not None and sellers > 0:
        safety.honeypot = False

    band = str(verdict.get("band") or "").lower() or None
    if band in _BAND_RISK:
        safety.risk_score = _BAND_RISK[band]

    signals = [
        (s.get("label"), s.get("tone"))
        for s in (verdict.get("signals") or [])
        if isinstance(s, dict) and s.get("label")
    ]
    for label, tone in signals:
        if str(tone).lower() in ("medium", "high"):
            safety.risk_flags.append(str(label))
    headline = verdict.get("headline")
    if headline and band not in (None, "low"):
        safety.risk_flags.append(str(headline))

    return PerceptorReport(
        safety=safety,
        band=band,
        band_label=verdict.get("band_label"),
        headline=headline,
        checks=[(str(a or ""), str(b or "")) for a, b in checks],
        signals=[(str(a or ""), str(b or "")) for a, b in signals],
        investigation_id=data.get("investigation_id"),
    )


class PerceptorProvider:
    """Perceptor client with a disk cache. Keyless, fail-open."""

    def __init__(self, cache_path: str | None = None, timeout_seconds: float = 15.0):
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.timeout = timeout_seconds
        self._session: Any = None
        self._cache: dict[str, Any] | None = None
        # Throttle memory: the POST endpoint is heavy server-side and rate
        # limits (observed as intermittent HTTP 403s). A throttled/failed
        # address cools down per-address so retries don't hammer the server;
        # a global backoff calms bursts across addresses. Fail-open always.
        self._post_cooldown: dict[str, float] = {}
        self._post_throttled_until: float = 0.0

    async def _get_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(
                trust_env=True, headers={"User-Agent": "FENRIR/2.0 discovery"}
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -- cache -----------------------------------------------------------
    def _load_cache(self) -> dict[str, Any]:
        cache = self._cache
        if cache is None:
            try:
                with open(self.cache_path) as f:
                    loaded = json.load(f)
                cache = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError):
                cache = {}
            self._cache = cache
        return cache

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._load_cache(), f)
        except OSError as e:
            logger.warning("Perceptor cache write failed: %s", e)

    def cached_report(self, address: str) -> PerceptorReport | None:
        """Return the completed report from cache only (no network)."""
        entry = self._load_cache().get(address.lower())
        if entry and entry.get("status") == "complete" and entry.get("report"):
            return parse_perceptor(entry["report"])
        return None

    # -- API --------------------------------------------------------------
    async def ensure_investigation(
        self, chain_id: int, address: str, context: dict[str, Any] | None = None
    ) -> str | None:
        """POST an investigation once per address; return its id (fail-open).

        Cache hit => no network. Only call for tokens worth the server-side
        cost (i.e. candidates that cleared FENRIR's filters). ``context``
        (symbol/name/chain/dexscreener) is stored with the entry so a later
        verdict follow-up can name the token.
        """
        addr = address.lower()
        entry = self._load_cache().get(addr)
        if entry and entry.get("investigation_id"):
            if context:
                entry.setdefault("context", {}).update(
                    {k: v for k, v in context.items() if v is not None}
                )
                self._save_cache()
            return str(entry["investigation_id"])
        now = time.time()
        if now < self._post_cooldown.get(addr, 0.0):
            logger.debug("Perceptor POST cooling down for %s…", addr[:10])
            return None
        if now < self._post_throttled_until:
            logger.debug("Perceptor POST globally throttled, skipping %s…", addr[:10])
            return None
        try:
            session = await self._get_session()
            for attempt in range(2):  # one retry with backoff
                async with session.post(
                    f"{PERCEPTOR_API}/investigations",
                    json={"chain_id": chain_id, "address": address},
                    timeout=self.timeout,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        inv_id = (
                            data.get("investigation_id") or data.get("public_id") or data.get("id")
                        )
                        if not inv_id:
                            return None
                        self._load_cache()[addr] = {
                            "investigation_id": inv_id,
                            "status": "pending",
                            "report": None,
                            "checked_at": time.time(),
                            "followup_sent": False,
                            "context": {k: v for k, v in (context or {}).items() if v is not None},
                        }
                        self._save_cache()
                        return str(inv_id)
                    if resp.status in (403, 429):
                        body = (await resp.text())[:200]
                        logger.warning(
                            "Perceptor POST throttled HTTP %d for %s…: %s",
                            resp.status,
                            addr[:10],
                            body,
                        )
                        backoff = 30.0 if attempt == 0 else 120.0
                        self._post_throttled_until = time.time() + backoff
                        self._post_cooldown[addr] = time.time() + 600.0
                        await asyncio.sleep(2**attempt + random.uniform(0, 1))
                        continue
                    logger.warning("Perceptor POST HTTP %d for %s…", resp.status, addr[:10])
                    self._post_cooldown[addr] = time.time() + 600.0
                    return None
            return None
        except Exception as e:  # noqa: BLE001 - fail-open
            logger.warning("Perceptor POST failed for %s…: %s", addr[:10], e)
            self._post_cooldown[addr] = time.time() + 600.0
            return None

    async def refresh_report(self, address: str) -> PerceptorReport | None:
        """If a cached investigation completed, fetch + cache + return it."""
        addr = address.lower()
        entry = self._load_cache().get(addr)
        if not entry or not entry.get("investigation_id"):
            return None
        if entry.get("status") == "complete" and entry.get("report"):
            return parse_perceptor(entry["report"])
        try:
            session = await self._get_session()
            async with session.get(
                f"{PERCEPTOR_API}/investigations/{entry['investigation_id']}",
                timeout=self.timeout,
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            if data.get("status") == "complete" and isinstance(data.get("verdict"), dict):
                entry["status"] = "complete"
                entry["report"] = data
                entry["checked_at"] = time.time()
                # Save only on a real state change: the cache carries full
                # verdict payloads and rewriting it on every no-op refresh
                # blocks the event loop for nothing.
                self._save_cache()
                return parse_perceptor(data)
            return None
        except Exception as e:  # noqa: BLE001 - fail-open
            logger.debug("Perceptor refresh failed for %s…: %s", addr[:10], e)
            return None

    async def investigate(
        self,
        chain_id: int,
        address: str,
        timeout_seconds: float = 300.0,
        poll_interval: float = 8.0,
        context: dict[str, Any] | None = None,
    ) -> PerceptorReport | None:
        """Start (or reuse) an investigation and block until it completes."""
        inv_id = await self.ensure_investigation(chain_id, address, context)
        if not inv_id:
            return None
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            report = await self.refresh_report(address)
            if report is not None:
                return report
            import asyncio

            await asyncio.sleep(poll_interval)
        logger.warning("Perceptor investigation %s timed out", inv_id[:8])
        return None


async def enrich_robinhood_safety(
    snap: TokenSnapshot, provider: PerceptorProvider | None
) -> PerceptorReport | None:
    """Fill empty Robinhood-chain safety from Perceptor (best-effort).

    Kicks off a cached investigation when none exists, merges a completed
    verdict into ``snap.safety``. Returns the report, or None when nothing
    usable came back. Never raises.
    """
    if provider is None or snap.chain is not Chain.ROBINHOOD:
        return None
    if not snap.safety.is_empty:
        return provider.cached_report(snap.token_address)
    ctx = snapshot_context(snap)
    try:
        await provider.ensure_investigation(ROBINHOOD_CHAIN_ID, snap.token_address, ctx)
        report = await provider.refresh_report(snap.token_address)
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.debug("Perceptor enrich failed for %s…: %s", snap.token_address[:10], e)
        return None
    if report is not None:
        snap.safety = report.safety
    return report
