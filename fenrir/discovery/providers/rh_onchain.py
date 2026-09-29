#!/usr/bin/env python3
"""
FENRIR - Robinhood-chain on-chain new-pair provider.

Polls ``eth_getLogs`` for Uniswap v4 pool ``Initialize`` events on Robinhood
Chain (chain id 4663) so the scout sees tokens at block zero — minutes before
DexScreener indexes the pair. This is the earliest possible sighting on the
chain: OFY-style runners were already +300% by the time aggregators listed
them, and no threshold tweak can recover a lead the pipeline never had.

Design
------
- :meth:`RobinhoodPairMonitor.sync` advances a persisted ``last_block``
  cursor and records ``first_seen`` (block + wall-clock timestamp) per new
  token. A dedicated 2-minute cron (``rh-pair-watch``, disabled until the
  machine migration) keeps the registry warm; the 10-minute scout consumes
  :meth:`fresh_addresses` and runs them through the normal pipeline.
- Only pools with a known base currency on one side are kept (native ETH,
  WETH, USDG). Meme/meme pairs are skipped as noise.
- Tokens DexScreener hasn't indexed yet evaluate to ``None`` and are simply
  retried next cycle while still fresh — the registry, not the alert, is the
  product of this source.
- Fail-open throughout: RPC errors yield no addresses, never an exception.

Verified live 2026-09-29
------------------------
- RPC ``https://rpc.mainnet.chain.robinhood.com`` → chain id 4663 (0x1237).
- PoolManager ``0x8366a39CC670B4001A1121B8F6A443A643e40951`` (NOT the
  canonical 0x0000…4444 deployment — Robinhood runs its own v4 deployment;
  confirmed via ``eth_getCode`` and two independent deployment records).
- ``Initialize`` topic0 ``0xdd466e67…8d6438`` computed from the canonical
  v4-core event
  ``Initialize(bytes32,address,address,uint24,int24,address,uint160,int24)``.
  (The 5-arg signature floating around in some docs is wrong and returns
  zero logs.)
- 280 pool initializations observed in a 20k-block window; base currencies
  seen: native ETH (address(0)), WETH, USDG.

State file (JSON)::
    {"last_block": 75902183,
     "first_seen": {"<token>": {"block": 75902100, "ts": 1759..., "base": "<addr>",
                                "pool_id": "0x...", "tx": "0x..."}}}
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import cast

import aiohttp

logger = logging.getLogger(__name__)

try:  # EIP-55 checksums for cross-source dedup consistency
    from Crypto.Hash import keccak as _keccak

    def _keccak256(data: bytes) -> bytes:
        k = _keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()

    _HAS_KECCAK = True
except ImportError:  # pragma: no cover - requirements pin pycryptodome
    _HAS_KECCAK = False

# ── Chain constants (verified live 2026-09-29) ────────────────────────────
ROBINHOOD_RPC_URL_DEFAULT = "https://rpc.mainnet.chain.robinhood.com"
ROBINHOOD_CHAIN_ID = 4663

#: Uniswap v4 PoolManager on Robinhood Chain (own deployment, not canonical).
V4_POOL_MANAGER = "0x8366a39CC670B4001A1121B8F6A443A643e40951"

#: keccak256("Initialize(bytes32,address,address,uint24,int24,address,uint160,int24)")
INITIALIZE_TOPIC0 = "0xdd466e674ea557f56295e2d0218a125ea4b4f0f6f3307b95f85e6110838d6438"

NATIVE_ETH = "0x0000000000000000000000000000000000000000"
WETH_ROBINHOOD = "0x0Bd7D308f8E1639FAb988df18A8011f41EAcAD73"
USDG_ROBINHOOD = "0x5fc5360D0400a0Fd4f2af552ADD042D716F1d168"

#: Base currencies whose pairings are worth tracking. Lowercased for compare.
BASE_CURRENCIES = frozenset({NATIVE_ETH.lower(), WETH_ROBINHOOD.lower(), USDG_ROBINHOOD.lower()})

DEFAULT_STATE_PATH = os.path.expanduser(
    os.getenv(
        "RH_PAIRS_STATE_PATH",
        "~/workspace/goals/token-scout-watch/hidden_files/rh_pairs.json",
    )
)

#: First-ever sync backfills this many blocks (~2h at ~250ms/block).
BACKFILL_BLOCKS = 30_000
#: eth_getLogs chunk size (public RPC is rate-limited; stay well under caps).
LOG_CHUNK_BLOCKS = 10_000
#: Registry entries older than this are pruned.
REGISTRY_TTL_HOURS = 24.0


def to_checksum(address: str) -> str:
    """EIP-55 checksum (lowercase fallback when keccak is unavailable)."""
    addr = address.lower().removeprefix("0x")
    if not _HAS_KECCAK or len(addr) != 40:
        return "0x" + addr
    digest = _keccak256(addr.encode("ascii")).hex()
    return "0x" + "".join(c.upper() if int(digest[i], 16) >= 8 else c for i, c in enumerate(addr))


class NewPool:
    """One freshly-initialized v4 pool with a known base currency."""

    __slots__ = ("token_address", "base_address", "pool_id", "block_number", "tx_hash")

    def __init__(
        self,
        token_address: str,
        base_address: str,
        pool_id: str,
        block_number: int,
        tx_hash: str,
    ) -> None:
        self.token_address = token_address
        self.base_address = base_address
        self.pool_id = pool_id
        self.block_number = block_number
        self.tx_hash = tx_hash

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"NewPool(token={self.token_address} base={self.base_address} "
            f"block={self.block_number})"
        )


def parse_initialize_log(log: dict) -> NewPool | None:
    """Parse an eth_getLogs entry into a NewPool, or None when unusable.

    Keeps only pools where exactly one side is a known base currency; the
    other side is the new token. Returns None for meme/meme pairs and
    malformed logs.
    """
    try:
        topics = log.get("topics") or []
        if len(topics) < 4:
            return None
        if topics[0].lower() != INITIALIZE_TOPIC0.lower():
            return None
        currency0 = "0x" + topics[2][-40:]
        currency1 = "0x" + topics[3][-40:]
        c0_base = currency0.lower() in BASE_CURRENCIES
        c1_base = currency1.lower() in BASE_CURRENCIES
        if c0_base == c1_base:  # both base (rebase pair) or neither (meme/meme)
            return None
        token = currency1 if c0_base else currency0
        base = currency0 if c0_base else currency1
        block_number = int(log.get("blockNumber", "0x0"), 16)
        return NewPool(
            token_address=to_checksum(token),
            base_address=to_checksum(base),
            pool_id=topics[1],
            block_number=block_number,
            tx_hash=log.get("transactionHash", ""),
        )
    except (ValueError, TypeError, IndexError, AttributeError):
        return None


class RobinhoodPairMonitor:
    """Polls v4 Initialize events and keeps the first-seen registry."""

    def __init__(
        self,
        rpc_url: str | None = None,
        state_path: str = DEFAULT_STATE_PATH,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.rpc_url = rpc_url or os.getenv("ROBINHOOD_RPC_URL", "") or ROBINHOOD_RPC_URL_DEFAULT
        self.state_path = state_path
        self.timeout = timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # aiohttp, not httpx: httpx crashes on the bracketed IPv6
            # literals in this sandbox's NO_PROXY (see AGENTS.md).
            self._session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "FENRIR/2.0 discovery"},
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _rpc(self, method: str, params: list) -> object | None:
        """One raw JSON-RPC call. Fail-open: None on any error."""
        try:
            session = await self._get_session()
            async with session.post(
                self.rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            ) as resp:
                if resp.status != 200:
                    logger.debug("rh_onchain RPC %s -> HTTP %s", method, resp.status)
                    return None
                payload = await resp.json()
            if "error" in payload:
                logger.debug("rh_onchain RPC %s error: %s", method, payload["error"])
                return None
            return cast("object | None", payload.get("result"))
        except Exception as e:  # noqa: BLE001 - discovery is fail-open
            logger.debug("rh_onchain RPC %s failed: %s", method, e)
            return None

    async def latest_block(self) -> int | None:
        result = await self._rpc("eth_blockNumber", [])
        if not isinstance(result, str):
            return None
        try:
            return int(result, 16)
        except ValueError:
            return None

    async def fetch_initializes(self, from_block: int, to_block: int) -> list[NewPool]:
        """All newly-initialized base-paired v4 pools in [from_block, to_block]."""
        out: list[NewPool] = []
        if to_block < from_block:
            return out
        for start in range(from_block, to_block + 1, LOG_CHUNK_BLOCKS):
            end = min(start + LOG_CHUNK_BLOCKS - 1, to_block)
            result = await self._rpc(
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
            if not isinstance(result, list):
                continue
            for log in result:
                pool = parse_initialize_log(log)
                if pool is not None:
                    out.append(pool)
        return out

    # ── Registry state file ──────────────────────────────────────────

    @staticmethod
    def _load_state(path: str) -> dict:
        try:
            with open(path) as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _save_state(state: dict, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, path)
        except OSError as e:  # noqa: BLE001 - state is best-effort
            logger.debug("rh_onchain state save failed: %s", e)

    def fresh_addresses(self, max_age_hours: float = 6.0, path: str | None = None) -> list[str]:
        """Token addresses first seen within the window (oldest first)."""
        store = self._load_state(path or self.state_path)
        first_seen = store.get("first_seen", {})
        cutoff = time.time() - max_age_hours * 3600
        fresh = [
            token
            for token, entry in first_seen.items()
            if isinstance(entry, dict) and entry.get("ts", 0) >= cutoff
        ]
        fresh.sort(key=lambda t: first_seen[t].get("ts", 0))
        return fresh

    @staticmethod
    def prune_state(
        max_age_hours: float = REGISTRY_TTL_HOURS,
        now: float | None = None,
        path: str = DEFAULT_STATE_PATH,
    ) -> int:
        """Drop registry entries older than the TTL. Returns number pruned."""
        now = now if now is not None else time.time()
        store = RobinhoodPairMonitor._load_state(path)
        first_seen = store.get("first_seen", {})
        cutoff = now - max_age_hours * 3600
        pruned = [
            t for t, e in first_seen.items() if not isinstance(e, dict) or e.get("ts", 0) < cutoff
        ]
        for t in pruned:
            del first_seen[t]
        if pruned:
            RobinhoodPairMonitor._save_state(store, path)
        return len(pruned)

    async def sync(self, path: str | None = None) -> dict:
        """Advance the cursor, record new pools. Returns {'new': [...], ...}.

        Fail-open: RPC trouble yields an empty ``new`` list and leaves the
        cursor untouched so nothing is skipped.
        """
        path = path or self.state_path
        store = self._load_state(path)
        first_seen: dict = store.get("first_seen", {})
        latest = await self.latest_block()
        if latest is None:
            return {"new": [], "latest_block": store.get("last_block"), "error": "rpc_unreachable"}
        last = store.get("last_block")
        if last is None:
            start = max(0, latest - BACKFILL_BLOCKS)
        else:
            start = int(last) + 1
        if start > latest:
            return {"new": [], "latest_block": latest, "up_to_date": True}
        pools = await self.fetch_initializes(start, latest)
        now = time.time()
        new: list[str] = []
        for pool in pools:
            if pool.token_address in first_seen:
                continue
            first_seen[pool.token_address] = {
                "block": pool.block_number,
                "ts": now,
                "base": pool.base_address,
                "pool_id": pool.pool_id,
                "tx": pool.tx_hash,
            }
            new.append(pool.token_address)
        store["first_seen"] = first_seen
        store["last_block"] = latest
        self._save_state(store, path)
        self.prune_state(path=path)
        return {"new": new, "latest_block": latest, "scanned_blocks": latest - start + 1}
