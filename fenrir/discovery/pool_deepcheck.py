#!/usr/bin/env python3
"""Deep pool-topology check for Robinhood v4 tokens.

A Perceptor verdict that carries main-pool risk flags ("main pool empty",
"dev can pull liquidity", ...) is a *qualification* for this check, not a
final answer. Perceptor reads the aggregator's "main pair", which we proved
(MOONLET, 2026-10-01) can be a synthetic pair id with zero on-chain
Initialize events while real trading lives in newer pools. This module
establishes ground truth before the verdict kills (or clears) a candidate:

1. Enumerate every real v4 pool containing the token (Initialize scan).
2. Read recent Swap events per pool -> live liquidity, swap count, LP-yank
   detection (liquidity added then pulled back to zero).
3. Bounded LP-owner check on pools that still hold liquidity
   (burned / known-locker / EOA via :mod:`fenrir.discovery.lp_lock_v4`).
4. Qualify the verdict: CONFIRMED | MISATTRIBUTED | PULLABLE | UNKNOWN.

Fail-open throughout: any RPC failure yields qualification UNKNOWN and the
caller's verdict is left untouched.

New HTTP code uses aiohttp (never httpx) — the sandbox proxy env crashes
httpx client construction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import aiohttp

from fenrir.discovery.lp_lock_v4 import (
    ROBINHOOD_RPC_URL_DEFAULT,
    inspect_v4_lp_lock,
)
from fenrir.discovery.providers.rh_onchain import (
    INITIALIZE_TOPIC0,
    LOG_CHUNK_BLOCKS,
    V4_POOL_MANAGER,
)

log = logging.getLogger(__name__)

try:
    from Crypto.Hash import keccak as _keccak

    def _keccak256(data: bytes) -> bytes:
        k = _keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()

    _HAS_KECCAK = True
except ImportError:  # pragma: no cover - requirements pin pycryptodome
    _HAS_KECCAK = False

# ── Trigger: which Perceptor verdict language qualifies for a deep check ──
_MAIN_POOL_PATTERNS = [
    re.compile(r"main\s*pool", re.I),
    re.compile(r"pull.{0,12}liquid", re.I),
    re.compile(r"liquid.{0,12}(empty|drain|gone|pull)", re.I),
    re.compile(r"\blp\b.{0,12}(empty|drain|unlock|pull)", re.I),
]


def verdict_needs_deepcheck(report: Any) -> bool:
    """True when a Perceptor report's flags/headline mention main-pool risk."""
    texts: list[str] = []
    try:
        signals = getattr(report, "signals", None) or []
        texts.extend(str(label or "") for label, _tone in signals)
        flags = getattr(getattr(report, "safety", None), "risk_flags", None) or []
        texts.extend(str(f or "") for f in flags)
        headline = getattr(report, "headline", None)
        if headline:
            texts.append(str(headline))
    except Exception:  # noqa: BLE001 - fail-open
        return False
    blob = " | ".join(texts)
    return any(p.search(blob) for p in _MAIN_POOL_PATTERNS)


# ── Results ───────────────────────────────────────────────────────────────
@dataclass
class LivePool:
    pool_id: str
    block_number: int
    base_address: str
    swaps: int = 0
    last_liquidity: int = 0
    max_liquidity: int = 0
    yanked: bool = False  # liquidity was added, then pulled back to zero
    lp_locked: bool | None = None
    lp_detail: str = ""


@dataclass
class PoolDeepCheck:
    token_address: str
    qualification: str = "UNKNOWN"  # CONFIRMED | MISATTRIBUTED | PULLABLE | UNKNOWN
    pools: list[LivePool] = field(default_factory=list)
    canonical_pool_id: str | None = None  # first-seen pool (chain + registry)
    registry_pool_id: str | None = None  # rh_pairs.json first-seen, if any
    yanked_pool_ids: list[str] = field(default_factory=list)
    pullable_pool_ids: list[str] = field(default_factory=list)
    detail: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _sanitize_proxy_env() -> None:
    """Strip bracketed IPv6 literals that crash httpx-style proxy parsing."""
    for key in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(key, "")
        if "[" in val:
            parts = [p for p in val.split(",") if "[" not in p]
            os.environ[key] = ",".join(parts)


async def _rpc(session: aiohttp.ClientSession, url: str, method: str, params: list) -> Any:
    try:
        async with session.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data.get("result")
    except Exception:  # noqa: BLE001 - fail-open
        return None


def _decode_initialize_for_token(log: dict, token_lc: str) -> dict[str, Any] | None:
    """Parse an Initialize log; keep the pool if the token is on either side."""
    try:
        topics = log.get("topics") or []
        if len(topics) < 4 or topics[0].lower() != INITIALIZE_TOPIC0.lower():
            return None
        c0 = "0x" + topics[2][-40:]
        c1 = "0x" + topics[3][-40:]
        if c0.lower() != token_lc and c1.lower() != token_lc:
            return None
        base = c1 if c0.lower() == token_lc else c0
        return {
            "pool_id": topics[1],
            "block_number": int(log.get("blockNumber", "0x0"), 16),
            "base_address": base,
        }
    except (ValueError, TypeError, IndexError, AttributeError):
        return None


def _swap_topic0() -> str | None:
    if not _HAS_KECCAK:
        return None
    # v4 Swap has a TRAILING uint24 fee param (unlike v3) — omitting it
    # yields a topic0 that matches nothing (caught live 2026-10-01).
    return (
        "0x" + _keccak256(b"Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)").hex()
    )


def _liquidity_from_swap_data(data: str | None) -> int | None:
    """4th abi word of Swap data = uint128 liquidity."""
    try:
        if not data or len(data) < 2 + 64 * 4:
            return None
        raw = bytes.fromhex(data[2:] if data.startswith("0x") else data)
        return int.from_bytes(raw[3 * 32 : 4 * 32], "big")
    except (ValueError, IndexError):
        return None


def _load_registry_pool_id(token_lc: str) -> str | None:
    try:
        path = os.path.expanduser("~/workspace/goals/token-scout-watch/hidden_files/rh_pairs.json")
        with open(path) as f:
            reg = json.load(f)
        seen = reg.get("first_seen") or {}
        for addr, rec in seen.items():
            if str(addr).lower() == token_lc and isinstance(rec, dict):
                pid = rec.get("pool_id")
                return str(pid).lower() if pid else None
    except (OSError, ValueError, AttributeError):
        pass
    return None


async def deepcheck_pools(
    token_address: str,
    *,
    rpc_url: str | None = None,
    scan_blocks: int = 500_000,
    budget_seconds: float = 120.0,
) -> PoolDeepCheck:
    """Run the full deep check, best-effort inside ``budget_seconds``."""
    t0 = time.monotonic()
    token_lc = (token_address or "").lower()
    dc = PoolDeepCheck(token_address=token_address)
    if not token_lc.startswith("0x") or len(token_lc) != 42:
        dc.detail = "not an EVM address"
        return dc

    _sanitize_proxy_env()
    url = rpc_url or os.getenv("ROBINHOOD_RPC_URL", "") or ROBINHOOD_RPC_URL_DEFAULT
    swap_t0 = _swap_topic0()

    def _remaining() -> float:
        return budget_seconds - (time.monotonic() - t0)

    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            latest_raw = await _rpc(session, url, "eth_blockNumber", [])
            if not isinstance(latest_raw, str):
                dc.detail = "RPC unreachable (no block number)"
                return dc
            latest = int(latest_raw, 16)
            from_block = max(0, latest - scan_blocks)

            # Phase 1 — enumerate pools via Initialize scan.
            seen: dict[str, dict[str, Any]] = {}
            try:
                async with asyncio.timeout(max(1.0, min(60.0, _remaining()))):
                    for start in range(from_block, latest + 1, LOG_CHUNK_BLOCKS):
                        if _remaining() <= 2:
                            break
                        end = min(start + LOG_CHUNK_BLOCKS - 1, latest)
                        logs = await _rpc(
                            session,
                            url,
                            "eth_getLogs",
                            [
                                {
                                    "address": V4_POOL_MANAGER,
                                    "topics": [INITIALIZE_TOPIC0],
                                    "fromBlock": hex(start),
                                    "toBlock": hex(end),
                                }
                            ],
                        )
                        if not isinstance(logs, list):
                            continue
                        for lg in logs:
                            parsed = _decode_initialize_for_token(lg, token_lc)
                            if parsed and parsed["pool_id"].lower() not in seen:
                                seen[parsed["pool_id"].lower()] = parsed
            except (TimeoutError, asyncio.CancelledError):
                pass
            if not seen:
                dc.detail = "no v4 pools found for token in scan window"
                dc.qualification = "UNKNOWN"
                return dc

            ordered = sorted(seen.values(), key=lambda p: p["block_number"])
            dc.canonical_pool_id = ordered[0]["pool_id"]
            dc.registry_pool_id = _load_registry_pool_id(token_lc)

            # Phase 2 — swap activity + liquidity per pool.
            if swap_t0:
                try:
                    async with asyncio.timeout(max(1.0, min(60.0, _remaining()))):
                        for pd in ordered:
                            if _remaining() <= 2:
                                break
                            lp = LivePool(
                                pool_id=pd["pool_id"],
                                block_number=pd["block_number"],
                                base_address=pd["base_address"],
                            )
                            # Only the pool's own lifetime needs scanning.
                            s_from = max(from_block, pd["block_number"])
                            liqs: list[int] = []
                            for start in range(s_from, latest + 1, LOG_CHUNK_BLOCKS):
                                if _remaining() <= 2:
                                    break
                                end = min(start + LOG_CHUNK_BLOCKS - 1, latest)
                                logs = await _rpc(
                                    session,
                                    url,
                                    "eth_getLogs",
                                    [
                                        {
                                            "address": V4_POOL_MANAGER,
                                            "topics": [swap_t0, pd["pool_id"]],
                                            "fromBlock": hex(start),
                                            "toBlock": hex(end),
                                        }
                                    ],
                                )
                                if not isinstance(logs, list):
                                    continue
                                for lg in logs:
                                    lq = _liquidity_from_swap_data(lg.get("data"))
                                    if lq is not None:
                                        liqs.append(lq)
                            lp.swaps = len(liqs)
                            if liqs:
                                lp.last_liquidity = liqs[-1]
                                lp.max_liquidity = max(liqs)
                                lp.yanked = lp.max_liquidity > 0 and lp.last_liquidity == 0
                            dc.pools.append(lp)
                except (TimeoutError, asyncio.CancelledError):
                    pass
            else:
                dc.pools = [
                    LivePool(
                        pool_id=p["pool_id"],
                        block_number=p["block_number"],
                        base_address=p["base_address"],
                    )
                    for p in ordered
                ]

            # Phase 3 — LP-owner check on pools that still hold liquidity.
            live = [p for p in dc.pools if p.last_liquidity > 0]
            if live and _remaining() > 10:
                from fenrir.discovery.lp_lock_v4 import _http_transport

                transport = await _http_transport(url, 20.0)
                try:
                    for pool in live:
                        if _remaining() <= 5:
                            break
                        age_min = max(1.0, (latest - pool.block_number) / 300.0)
                        # Tight cap: this is a qualification pass, not an audit.
                        res = await inspect_v4_lp_lock(
                            pool.pool_id,
                            age_minutes=min(age_min, 600.0),
                            max_positions=150,
                            transport=transport,
                        )
                        pool.lp_locked = res.locked
                        pool.lp_detail = res.detail or ""
                        if res.locked is False:
                            dc.pullable_pool_ids.append(pool.pool_id)
                finally:
                    try:
                        sess = getattr(transport, "_session", None)
                        if sess is not None:
                            await sess.close()
                    except Exception as e:  # noqa: BLE001
                        log.debug("deepcheck transport close failed: %s", e)

            for pool in dc.pools:
                if pool.yanked:
                    dc.yanked_pool_ids.append(pool.pool_id)

            dc.qualification = _qualify(dc)
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("deepcheck failed for %s: %s", token_lc[:10], e)
        if dc.qualification == "UNKNOWN" and not dc.detail:
            dc.detail = f"check failed: {type(e).__name__}"
    finally:
        dc.elapsed_seconds = round(time.monotonic() - t0, 1)
    return dc


def _qualify(dc: PoolDeepCheck) -> str:
    """Decide the verdict qualification from observed pool topology."""
    if not dc.pools:
        return "UNKNOWN"
    live = [p for p in dc.pools if p.last_liquidity > 0]
    if dc.pullable_pool_ids or dc.yanked_pool_ids:
        # Someone CAN pull (or already did) — "dev can pull liquidity"
        # confirmed behaviorally, wherever the aggregator looked.
        return "PULLABLE"
    if live:
        canon = next((p for p in dc.pools if p.pool_id == dc.canonical_pool_id), None)
        canon_live = canon is not None and canon.last_liquidity > 0
        if not canon_live:
            # Canonical/first pool empty but trading lives elsewhere:
            # the verdict looked at the wrong pool.
            return "MISATTRIBUTED"
        return "CONFIRMED"
    # No pool holds liquidity anywhere — "main pool empty" holds on-chain.
    return "CONFIRMED"


def format_deepcheck(dc: PoolDeepCheck) -> str:
    """One compact section for the verdict card (bot-facing copy)."""
    q = dc.qualification
    n = len(dc.pools)
    live = sum(1 for p in dc.pools if p.last_liquidity > 0)
    if q == "UNKNOWN":
        return f"🔍 deep check: inconclusive ({dc.detail or 'no data'})."
    if q == "CONFIRMED":
        return (
            f"🔍 deep check: verdict holds — {n} pool{'s' if n != 1 else ''} "
            f"scanned on-chain, none hold liquidity."
        )
    if q == "MISATTRIBUTED":
        return (
            f"🔍 deep check: verdict looked at the wrong pool — first pool empty "
            f"but {live}/{n} newer pool{'s' if live != 1 else ''} hold liquidity. "
            f"Treat the flag as disputed, not dead."
        )
    if q == "PULLABLE":
        bits = []
        if dc.pullable_pool_ids:
            bits.append(f"{len(dc.pullable_pool_ids)} pool(s) with EOA-held LP")
        if dc.yanked_pool_ids:
            bits.append(f"{len(dc.yanked_pool_ids)} pool(s) already yanked to zero")
        return "🔍 deep check: pull risk CONFIRMED on-chain — " + ", ".join(bits) + "."
    return f"🔍 deep check: {q}."


async def main_cli(address: str, budget: float = 120.0) -> int:
    dc = await deepcheck_pools(address, budget_seconds=budget)
    print(json.dumps(dc.to_dict(), indent=1))
    print(format_deepcheck(dc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    addr = sys.argv[1] if len(sys.argv) > 1 else ""
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    sys.exit(asyncio.run(main_cli(addr, budget)))
