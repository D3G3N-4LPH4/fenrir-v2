#!/usr/bin/env python3
"""On-chain Uniswap v4 LP-lock verification for Robinhood chain.

Uniswap v4 LP positions are ERC721 NFTs minted by the PositionManager contract.
A pool's LP is LOCKED when every position NFT for that pool is burned (zero /
dead address) or held by a known locker contract. A position held by a plain
EOA can be pulled at any time, so any EOA holder means UNLOCKED.

Fail-open throughout: any RPC failure returns ``locked=None`` and the filter
gate decides what "unknown" means (young-coin filters fail closed on it — see
the ATM lesson in :mod:`fenrir.discovery.filters`).

Chain constants verified 2026-09-30 against Robinhood's v4 deployment records:
- PositionManager ``0x58daec3116aae6D93017bAAea7749052E8a04fA7``
- PoolManager     ``0x8366a39CC670B4001A1121B8F6A443A643e40951``
- RPC ``https://rpc.mainnet.chain.robinhood.com`` (chain id 4663)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp

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

# ── Chain constants ───────────────────────────────────────────────────
ROBINHOOD_RPC_URL_DEFAULT = "https://rpc.mainnet.chain.robinhood.com"

#: Uniswap v4 PositionManager on Robinhood Chain (own deployment, not canonical).
V4_POSITION_MANAGER = "0x58daec3116aae6D93017bAAea7749052E8a04fA7"

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
DEAD_ADDRESS = "0x000000000000000000000000000000000000dEaD"
#: Owners that prove the LP cannot be pulled.
BURN_ADDRESSES = frozenset({ZERO_ADDRESS, DEAD_ADDRESS.lower()})

#: Contracts known to custody locked v4 LP on Robinhood chain (lowercased).
#: PairPad launchers graduate into permanently locked v4 pools; their locker
#: contracts hold the position NFTs instead of the dev.
#: Source: pardotfamily/par deployment records (Robinhood chain, 4663).
KNOWN_LOCKERS = frozenset(
    {
        "0x8a6d37b2e6a2ac7970ef69d2932757f04be0a231",  # PairPadLaunchLocker
        "0x5826fbb6201daacd924a3d292841da9142952d59",  # PairPadMultiLaunchLocker
    }
)

#: Blocks per eth_getLogs call (public RPC is rate-limited; stay well under caps).
LOG_CHUNK_BLOCKS = 10_000
#: Assumed chain speed for sizing the mint scan window (conservative).
BLOCKS_PER_MINUTE = 300
#: Extra minutes scanned before the token's first sighting (launch-tx variance).
SCAN_BUFFER_MINUTES = 60
#: Hard cap on the mint-scan window.
MAX_SCAN_BLOCKS = 1_000_000
#: Cap on position tokenIds resolved per check (launch LPs are few; squats many).
#: Young-coin windows are small in practice (a 20m-old coin scans ~800 mints);
#: the cap is a safety valve for pathological windows, and truncation is noted.
MAX_POSITIONS = 1500
#: Parallel RPC call fan-out.
RPC_FANOUT = 10
#: Retries for flaky eth_call responses on the rate-limited public RPC.
ETH_CALL_RETRIES = 1


def _selector(signature: str) -> bytes:
    """First 4 bytes of keccak256(signature) — the ABI function selector."""
    if not _HAS_KECCAK:
        raise RuntimeError("pycryptodome is required for v4 LP-lock checks")
    return _keccak256(signature.encode("ascii"))[:4]


def _transfer_topic0() -> str:
    return "0x" + _keccak256(b"Transfer(address,address,uint256)").hex()


_ZERO_TOPIC = "0x" + "00" * 32


def _addr_word(address: str) -> bytes:
    """20-byte address as a 32-byte ABI word."""
    return bytes.fromhex(address.lower().removeprefix("0x").rjust(64, "0"))


def pool_id_from_key(
    currency0: str, currency1: str, fee: int, tick_spacing: int, hooks: str
) -> str:
    """v4 ``PoolKey.toId()``: keccak256(abi.encode(c0, c1, fee, tickSpacing, hooks))."""
    blob = (
        _addr_word(currency0)
        + _addr_word(currency1)
        + fee.to_bytes(32, "big")
        + tick_spacing.to_bytes(32, "big", signed=True)
        + _addr_word(hooks)
    )
    return "0x" + _keccak256(blob).hex()


def _decode_pool_key(data: bytes) -> tuple[str, str, int, int, str] | None:
    """Decode getPoolAndPositionInfo's PoolKey (first 5 words of the return)."""
    if len(data) < 160:
        return None
    currency0 = "0x" + data[12:32].hex()
    currency1 = "0x" + data[44:64].hex()
    fee = int.from_bytes(data[64:96], "big")
    tick_spacing = int.from_bytes(data[96:128], "big", signed=True)
    hooks = "0x" + data[140:160].hex()
    return currency0, currency1, fee, tick_spacing, hooks


def _decode_address_word(data: bytes) -> str | None:
    if len(data) < 32:
        return None
    return "0x" + data[12:32].hex()


def _parse_mint_token_ids(logs: list[dict[str, Any]]) -> list[int]:
    """TokenIds from PositionManager mints (Transfer with from == zero address)."""
    ids: list[int] = []
    seen: set[int] = set()
    for entry in logs:
        topics = entry.get("topics") or []
        if len(topics) < 4:
            continue
        if str(topics[1]).lower() != _ZERO_TOPIC:
            continue  # not a mint
        try:
            token_id = int(str(topics[3]), 16)
        except ValueError:
            continue
        if token_id not in seen:
            seen.add(token_id)
            ids.append(token_id)
    return ids


def _sanitize_proxy_env() -> None:
    """Strip bracketed IPv6 literals from NO_PROXY (they crash proxy parsing)."""
    for var in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(var)
        if not val:
            continue
        cleaned = ",".join(part for part in val.split(",") if "[" not in part)
        os.environ[var] = cleaned


@dataclass(frozen=True)
class PositionHolding:
    token_id: int
    owner: str


@dataclass
class V4LpLock:
    """Outcome of the on-chain LP-lock check.

    ``locked`` is True (all positions burned/locker-held), False (an EOA holds
    a position — pullable), or None (could not determine — fail-open).
    """

    locked: bool | None
    positions_checked: int = 0
    holdings: list[PositionHolding] = field(default_factory=list)
    detail: str = ""


# transport: async (method, params) -> result | None. Injected in tests.
Transport = Callable[[str, list], Awaitable[Any]]


async def _http_transport(rpc_url: str, timeout_seconds: float) -> Transport:
    _sanitize_proxy_env()
    session = aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        headers={"User-Agent": "FENRIR/2.0 lp-lock-v4"},
    )

    async def _rpc(method: str, params: list) -> Any:
        try:
            async with session.post(
                rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            ) as resp:
                if resp.status != 200:
                    log.debug("lp_lock_v4 RPC %s -> HTTP %s", method, resp.status)
                    return None
                payload = await resp.json()
            if "error" in payload:
                log.debug("lp_lock_v4 RPC %s error: %s", method, payload["error"])
                return None
            return payload.get("result")
        except Exception as e:  # noqa: BLE001 - fail-open
            log.debug("lp_lock_v4 RPC %s failed: %s", method, e)
            return None

    # Attach the session so the caller can close it.
    _rpc._session = session  # type: ignore[attr-defined]
    return _rpc


async def _eth_call(rpc: Transport, to: str, calldata: str, block: str = "latest") -> bytes | None:
    for attempt in range(ETH_CALL_RETRIES + 1):
        result = await rpc("eth_call", [{"to": to, "data": calldata}, block])
        if isinstance(result, str) and result.startswith("0x"):
            try:
                return bytes.fromhex(result.removeprefix("0x"))
            except ValueError:
                return None
        if attempt < ETH_CALL_RETRIES:
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


async def _fetch_mint_token_ids(rpc: Transport, from_block: int, to_block: int) -> list[int]:
    """All PositionManager position tokenIds minted in [from_block, to_block]."""
    topic0 = _transfer_topic0()
    token_ids: list[int] = []
    for start in range(from_block, to_block + 1, LOG_CHUNK_BLOCKS):
        end = min(start + LOG_CHUNK_BLOCKS - 1, to_block)
        logs = await rpc(
            "eth_getLogs",
            [
                {
                    "address": V4_POSITION_MANAGER,
                    "topics": [topic0, _ZERO_TOPIC],
                    "fromBlock": hex(start),
                    "toBlock": hex(end),
                }
            ],
        )
        if isinstance(logs, list):
            token_ids.extend(_parse_mint_token_ids(logs))
    # De-dupe preserving mint order.
    seen: set[int] = set()
    ordered: list[int] = []
    for token_id in token_ids:
        if token_id not in seen:
            seen.add(token_id)
            ordered.append(token_id)
    return ordered


async def _positions_for_pool(rpc: Transport, pool_id: str, token_ids: list[int]) -> list[int]:
    """Which of these position tokenIds belong to ``pool_id``."""
    pool_id = pool_id.lower()
    selector = _selector("getPoolAndPositionInfo(uint256)")
    sem = asyncio.Semaphore(RPC_FANOUT)

    async def _check(token_id: int) -> int | None:
        async with sem:
            data = await _eth_call(
                rpc,
                V4_POSITION_MANAGER,
                "0x" + (selector + token_id.to_bytes(32, "big")).hex(),
            )
        if data is None:
            return None
        key = _decode_pool_key(data)
        if key is None:
            return None
        currency0, currency1, fee, tick_spacing, hooks = key
        if pool_id_from_key(currency0, currency1, fee, tick_spacing, hooks).lower() == pool_id:
            return token_id
        return None

    matches = await asyncio.gather(*(_check(t) for t in token_ids))
    return [t for t in matches if t is not None]


async def _classify_holders(rpc: Transport, token_ids: list[int]) -> list[PositionHolding] | None:
    """Resolve position owners. None if any owner lookup failed."""
    selector = _selector("ownerOf(uint256)")
    sem = asyncio.Semaphore(RPC_FANOUT)

    async def _owner(token_id: int) -> PositionHolding | None:
        async with sem:
            data = await _eth_call(
                rpc,
                V4_POSITION_MANAGER,
                "0x" + (selector + token_id.to_bytes(32, "big")).hex(),
            )
        if data is None:
            return None  # position destroyed (liquidity already removed) — skip
        owner = _decode_address_word(data)
        if owner is None:
            return None
        return PositionHolding(token_id=token_id, owner=owner)

    results = await asyncio.gather(*(_owner(t) for t in token_ids))
    return [r for r in results if r is not None]


async def _is_eoa(rpc: Transport, address: str) -> bool | None:
    code = await rpc("eth_getCode", [address, "latest"])
    if not isinstance(code, str):
        return None
    return code in ("0x", "0x0", "")


async def inspect_v4_lp_lock(
    pool_id: str,
    *,
    rpc_url: str | None = None,
    age_minutes: float | None = None,
    timeout_seconds: float = 20.0,
    max_positions: int = MAX_POSITIONS,
    transport: Transport | None = None,
) -> V4LpLock:
    """Check whether a Uniswap v4 pool's LP is locked, on-chain.

    ``pool_id`` is the v4 pool id (DexScreener's ``pairAddress`` for v4 pools:
    ``0x`` + 64 hex chars). The mint scan window is sized from ``age_minutes``
    when given (the launch LP was minted at launch), so young coins scan only
    tens of thousands of blocks.
    """
    pool_id = (pool_id or "").lower()
    if len(pool_id) != 66 or not pool_id.startswith("0x"):
        return V4LpLock(locked=None, detail="not a v4 pool id")
    if not _HAS_KECCAK:
        return V4LpLock(locked=None, detail="pycryptodome unavailable")

    own_transport = transport is None
    rpc = transport
    try:
        if own_transport:
            url = rpc_url or os.getenv("ROBINHOOD_RPC_URL", "") or ROBINHOOD_RPC_URL_DEFAULT
            rpc = await _http_transport(url, timeout_seconds)
        assert rpc is not None

        latest_raw = await rpc("eth_blockNumber", [])
        if not isinstance(latest_raw, str):
            return V4LpLock(locked=None, detail="RPC unreachable (no block number)")
        latest = int(latest_raw, 16)

        if age_minutes is not None:
            window = int((age_minutes + SCAN_BUFFER_MINUTES) * BLOCKS_PER_MINUTE)
        else:
            window = 200_000
        window = min(window, MAX_SCAN_BLOCKS)
        from_block = max(0, latest - window)

        token_ids = await _fetch_mint_token_ids(rpc, from_block, latest)
        if not token_ids:
            return V4LpLock(locked=None, detail="no v4 positions minted in scan window")
        truncated = len(token_ids) > max_positions
        token_ids = token_ids[:max_positions]

        pool_tokens = await _positions_for_pool(rpc, pool_id, token_ids)
        if not pool_tokens:
            detail = "no positions found for this pool"
            if truncated:
                detail += f" (scan truncated at {max_positions} mints)"
            return V4LpLock(locked=None, detail=detail)

        holdings = await _classify_holders(rpc, pool_tokens)
        if not holdings:
            return V4LpLock(locked=None, detail="position owners unresolvable")

        # Decide over ALL positions: any EOA holder can pull its position, so
        # one EOA anywhere dominates an unclassifiable contract elsewhere.
        eoa_holding: PositionHolding | None = None
        unknown_holding: PositionHolding | None = None
        for holding in holdings:
            owner = holding.owner.lower()
            if owner in BURN_ADDRESSES or owner in KNOWN_LOCKERS:
                continue
            eoa = await _is_eoa(rpc, holding.owner)
            if eoa is True:
                eoa_holding = holding
                break
            if unknown_holding is None:
                unknown_holding = holding
        if eoa_holding is not None:
            return V4LpLock(
                locked=False,
                positions_checked=len(holdings),
                holdings=holdings,
                detail=f"position #{eoa_holding.token_id} held by EOA "
                f"{eoa_holding.owner[:10]}… — LP pullable",
            )
        if unknown_holding is not None:
            return V4LpLock(
                locked=None,
                positions_checked=len(holdings),
                holdings=holdings,
                detail=f"position #{unknown_holding.token_id} held by unlisted contract "
                f"{unknown_holding.owner[:10]}… — LP lock unknown",
            )

        short = ",".join(f"#{h.token_id}" for h in holdings[:5])
        return V4LpLock(
            locked=True,
            positions_checked=len(holdings),
            holdings=holdings,
            detail=f"{len(holdings)} v4 LP position(s) ({short}) burned/locker-held — locked",
        )
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("inspect_v4_lp_lock failed for %s: %s", pool_id[:14], e)
        return V4LpLock(locked=None, detail=f"check failed: {type(e).__name__}")
    finally:
        if own_transport and rpc is not None:
            session = getattr(rpc, "_session", None)
            if session is not None:
                await session.close()
