#!/usr/bin/env python3
"""Bundle + deployer-cluster check for Robinhood chain (4663).

Detects coordinated insider accumulation on a token:

- **Bundle detection**: wallets that bought in the launch window (first
  ~10 min after the pool's ``Initialize``) and are linked by tight buy
  timing or a shared native-ETH funder are one entity. Their combined buys
  as a share of total supply is ``bundled_supply_pct``.
- **Deployer tracing**: the token's ``deployer()`` getter identifies the
  launcher; its current holding, direct token distributions, and its own
  funder (cross-checked against a serial-launcher registry) become
  ``deployer_holding_pct`` / ``deployer_distributed_wallets`` /
  ``deployer_funder_is_serial_launcher``. ``deployer_cluster_pct`` is the
  supply share that flowed through wallets one hop from the deployer.

Block-time scaling (deviation from spec §2.2, measured 2026-09-30):
Robinhood chain runs ~0.1s/block (600/min), not the ~2s the spec assumed.
All block-denominated constants are scaled ×20 to preserve the spec's
*time* intent: 6,000-block launch window (~10 min), 30,000-block fallback
(~50 min), timing runs of ≤60 blocks with ≤20-block gaps (~6s/~2s, the
spec's "≤3-block run" at 2s/block). Same-block buys remain the strongest
coordinated signal at any chain speed.

Funder-tracing deviation (spec §2.3(b)): native ETH transfers emit no logs,
so funder tracing uses the Blockscout txlist API
(``https://robinhoodchain.blockscout.com/api``, env override
``ROBINHOOD_EXPLORER_API_URL``; external then internal txs) instead of
``eth_getLogs``.

Thresholds follow the Bubblemaps severity bands d3g3n adopted (2026-09-30):
<5% clean, 5-15% warn, 15-30% score penalty, >30% hard fail on risk-on
filters. Bubblemaps itself does not support chain 4663, so this is built on
raw RPC + the public Blockscout API.

Fail-open throughout: any RPC/HTTP failure returns ``None`` and the filter
gate treats unknown as "inconclusive" (warn, never fail).

Follows the patterns of :mod:`fenrir.discovery.lp_lock_v4` (commit 2a1590b):
aiohttp only (never httpx), bracketed-IPv6 NO_PROXY sanitization, bounded
RPC budget, shared transport plumbing imported from that module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import aiohttp

from fenrir.discovery.lp_lock_v4 import (
    ROBINHOOD_RPC_URL_DEFAULT,
    Transport,
    _addr_word,
    _eth_call,
    _http_transport,
    _selector,
)

_keccak: Callable[[bytes], bytes] | None
try:
    from fenrir.discovery.lp_lock_v4 import _keccak256 as _keccak_impl

    _keccak = _keccak_impl
except ImportError:  # pragma: no cover - requirements pin pycryptodome
    _keccak = None

log = logging.getLogger(__name__)

# ── Chain constants (Robinhood chain, 4663) ─────────────────────────────
#: Uniswap v4 PoolManager (Robin's own deployment).
V4_POOL_MANAGER = "0x8366a39CC670B4001A1121B8F6A443A643e40951"
#: v4 PositionManager (LP position NFTs).
V4_POSITION_MANAGER = "0x58daec3116aae6D93017bAAea7749052E8a04fA7"
#: Initialize(bytes32,address,address,uint24,int24,address,uint160,int24).
INITIALIZE_TOPIC0 = "0xdd466e674ea557f56295e2d0218a125ea4b4f0f6f3307b95f85e6110838d6438"
#: Base quote assets excluded from buyer lists (plumbing, not buyers).
WETH_ADDRESS = "0x0Bd7D308f8E1639FAb988df18A8011f41EAcAD73"
USDG_ADDRESS = "0x5fc5360D0400a0Fd4f2af552ADD042D716F1d168"
#: Blockscout API for native-ETH funding traces (native transfers emit no
#: logs, so eth_getLogs cannot see them — the API's txlist is the only
#: practical source; keyless, public).
BLOCKSCOUT_API_DEFAULT = "https://robinhoodchain.blockscout.com/api"

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
DEAD_ADDRESS = "0x000000000000000000000000000000000000dEaD"

#: Addresses that are pool/router plumbing, never buyers or distributees.
_PLUMBING = frozenset(
    {
        V4_POOL_MANAGER.lower(),
        V4_POSITION_MANAGER.lower(),
        WETH_ADDRESS.lower(),
        USDG_ADDRESS.lower(),
        ZERO_ADDRESS,
        DEAD_ADDRESS.lower(),
    }
)

#: Funders that must NOT link wallets (CEX hot wallets, public faucets).
#: Extend as they are identified; empty seed is honest — no invented addresses.
FUNDER_DENYLIST: frozenset[str] = frozenset()

#: Known serial launchers: deployer/funder address -> note. Behavior signal
#: (launch→extract→abandon), not a holdings signal — see the FARTDOG dev.
SERIAL_LAUNCHERS: dict[str, str] = {
    "0xea1a19ed166097b3574a68f363292ef006700c90": (
        "FARTDOG dev: 12 launches, 4 in one day, 11 with no liquidity left"
    ),
}

#: Launch window after pool Initialize. Spec §2.2 assumed ~2s blocks (300
#: blocks ≈ 10 min); Robinhood chain measures ~0.1s/block, so 6,000 blocks
#: keeps the intended ~10-minute window.
LAUNCH_WINDOW_BLOCKS = 6_000
#: Fallback window for tokens whose Initialize predates the scan range
#: (spec: 1500 blocks at ~2s = ~50 min; scaled to the measured chain speed).
FALLBACK_WINDOW_BLOCKS = 30_000
#: Native-funding lookback before first buy (blocks). The Blockscout
#: implementation ignores this (it takes the wallet's oldest tx); kept for
#: the seam signature.
FUNDER_LOOKBACK_BLOCKS = 2000
#: Cap on wallets traced for shared funders (timing-cluster members first,
#: then top buyers by amount).
MAX_FUNDER_TRACES = 25
#: Blocks per eth_getLogs call for deployer-flow scans.
LOG_CHUNK_BLOCKS = 10_000
#: Cap on deployer-flow scan chunks (500k blocks ≈ 14h at 0.1s/block).
MAX_FLOW_CHUNKS = 50
#: Measured Robinhood-chain speed (~0.1s/block, 2026-09-30) used to size the
#: Initialize scan window. Spec §2.2 assumed 300/min (~2s blocks).
BLOCKS_PER_MINUTE = 600
#: Extra minutes scanned before the token's first sighting.
SCAN_BUFFER_MINUTES = 60
#: Hard cap on the Initialize scan window.
MAX_SCAN_BLOCKS = 1_000_000
#: Minimum distinct buyers in a timing run to count as a cluster.
TIMING_CLUSTER_MIN_WALLETS = 4
#: Max gap (blocks) between consecutive buys in one timing run. Tight at
#: 0.1s/block: same-block (or adjacent-block) synchronization is the signal;
#: wider runs are noise on hot launches (measured on FARTDOG: 60-block runs
#: flagged 289 unrelated wallets).
TIMING_RUN_GAP_BLOCKS = 1
#: Max span (blocks) of a timing run.
TIMING_RUN_MAX_BLOCKS = 2
#: Parallel RPC/HTTP fan-out.
FANOUT = 5

#: Disk cache: one investigation per token, 24h TTL (distribution only
#: concentrates over time; cached re-evals skip the RPC cost).
_CACHE_PATH = os.path.expanduser("~/.cache/fenrir/bundle_check.json")
_CACHE_TTL_S = 24 * 3600


def _transfer_topic0() -> str:
    assert _keccak is not None  # guarded by check_bundle_and_deployer
    return "0x" + _keccak(b"Transfer(address,address,uint256)").hex()


def _topic_word(address: str) -> str:
    return "0x" + _addr_word(address).hex()


def _topic_address(topic: str) -> str:
    return "0x" + topic[-40:].lower()


# ── Report ──────────────────────────────────────────────────────────────
@dataclass
class BundleDeployerReport:
    """Outcome of the bundle + deployer-cluster check.

    Percentage fields are ``None`` when the check could not determine them
    (fail-open); the filter gate treats ``None`` as "inconclusive".
    """

    bundled_supply_pct: float | None = None
    cluster_count: int = 0
    largest_cluster_pct: float | None = None
    cluster_wallets: list[list[str]] = field(default_factory=list)
    deployer_address: str | None = None
    deployer_holding_pct: float | None = None
    deployer_distributed_wallets: int | None = None
    deployer_distributed_pct: float | None = None
    deployer_funder: str | None = None
    deployer_funder_is_serial_launcher: bool | None = None
    deployer_cluster_pct: float | None = None
    launch_window_partial: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleDeployerReport:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ── Disk cache ──────────────────────────────────────────────────────────
def _load_cache() -> dict[str, Any]:
    try:
        with open(_CACHE_PATH) as f:
            loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def get_cached_bundle_report(token_address: str) -> dict[str, Any] | None:
    """Return the cached report dict when fresh, else None."""
    entry = _load_cache().get(token_address.lower())
    if not isinstance(entry, dict):
        return None
    ttl = float(entry.get("ttl", _CACHE_TTL_S))
    if time.time() - float(entry.get("cached_at", 0)) > ttl:
        return None
    report = entry.get("report")
    return report if isinstance(report, dict) else None


def save_cached_bundle_report(
    token_address: str, report: dict[str, Any], ttl_seconds: float = _CACHE_TTL_S
) -> None:
    cache = _load_cache()
    cache[token_address.lower()] = {
        "cached_at": time.time(),
        "ttl": ttl_seconds,
        "report": report,
    }
    _save_cache(cache)


#: Backoff for inconclusive bundle checks (timed out / no chain data yet).
#: A fraction of the 24h conclusive TTL — the token may become checkable as
#: chain data settles, but we don't want to burn a 60s RPC walk every tick.
INCONCLUSIVE_TTL_S = 3600


# ── Clustering ──────────────────────────────────────────────────────────
def _timing_clusters(first_buys: dict[str, tuple[int, int]]) -> list[set[str]]:
    """Group wallets whose first buys form a tight run: consecutive blocks
    with gaps ≤ ``TIMING_RUN_GAP_BLOCKS`` spanning at most
    ``TIMING_RUN_MAX_BLOCKS``, containing ≥4 distinct buyers."""
    by_block: dict[int, list[str]] = {}
    for wallet, (block, _amount) in first_buys.items():
        by_block.setdefault(block, []).append(wallet)
    blocks = sorted(by_block)
    clusters: list[set[str]] = []
    i = 0
    while i < len(blocks):
        run = [blocks[i]]
        j = i + 1
        while (
            j < len(blocks)
            and blocks[j] - run[-1] <= TIMING_RUN_GAP_BLOCKS
            and blocks[j] - run[0] <= TIMING_RUN_MAX_BLOCKS
        ):
            run.append(blocks[j])
            j += 1
        wallets = {w for b in run for w in by_block[b]}
        if len(wallets) >= TIMING_CLUSTER_MIN_WALLETS:
            clusters.append(wallets)
        i = j
    return clusters


def _merge_clusters(cluster_list: list[set[str]]) -> list[set[str]]:
    """Union clusters that share any wallet (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cluster in cluster_list:
        members = list(cluster)
        for w in members:
            parent.setdefault(w, w)
        for w in members[1:]:
            parent[find(w)] = find(members[0])
    groups: dict[str, set[str]] = {}
    for w in parent:
        groups.setdefault(find(w), set()).add(w)
    return list(groups.values())


# ── Native-ETH funder tracing (Blockscout) ───────────────────────────────
async def _blockscout_get(
    session: aiohttp.ClientSession, params: dict[str, str]
) -> list[dict[str, Any]] | None:
    base = os.getenv("ROBINHOOD_EXPLORER_API_URL", "") or BLOCKSCOUT_API_DEFAULT
    try:
        async with session.get(
            base, params=params, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status != 200:
                return None
            payload = await resp.json()
    except Exception:  # noqa: BLE001 - fail-open
        return None
    if not isinstance(payload, dict) or payload.get("status") != "1":
        return None
    result = payload.get("result")
    return result if isinstance(result, list) else None


async def blockscout_funder(
    session: aiohttp.ClientSession, wallet: str, before_block: int | None = None
) -> str | None:
    """Earliest native-ETH funder of ``wallet`` (external txs, then internal).

    Native transfers emit no logs, so the explorer's txlist is the practical
    source. Returns the funder address or None.
    """
    wallet = wallet.lower()
    for action in ("txlist", "txlistinternal"):
        txs = await _blockscout_get(
            session,
            {
                "module": "account",
                "action": action,
                "address": wallet,
                "sort": "asc",
                "page": "1",
                "offset": "100",
            },
        )
        if not txs:
            continue
        for tx in txs:
            try:
                block = int(tx.get("blockNumber", "0"))
            except (TypeError, ValueError):
                continue
            if before_block is not None and block > before_block:
                break  # ascending: nothing later can qualify
            to = str(tx.get("to") or "").lower()
            frm = str(tx.get("from") or "").lower()
            try:
                value = int(tx.get("value", "0"))
            except (TypeError, ValueError):
                value = 0
            if to == wallet and value > 0 and frm and frm != wallet:
                return frm
    return None


# Funder lookup seam: (session, wallet, before_block) -> funder | None.
# Injected in tests; the default below is the Blockscout implementation.
# ``session`` is None when a test injects its own lookup.
FunderLookup = Callable[[Any, str, int | None], Awaitable[str | None]]


async def _default_funder_lookup(
    session: aiohttp.ClientSession, wallet: str, before_block: int | None
) -> str | None:
    return await blockscout_funder(session, wallet, before_block)


# ── RPC helpers ─────────────────────────────────────────────────────────
async def _get_logs(
    rpc: Transport,
    address: str,
    topics: list[str | None],
    from_block: int,
    to_block: int,
    chunk: int = LOG_CHUNK_BLOCKS,
    max_chunks: int = MAX_FLOW_CHUNKS,
) -> list[list[dict[str, Any]] | None]:
    """Chunked eth_getLogs. Returns per-chunk results; None marks a failed chunk."""
    out: list[list[dict[str, Any]] | None] = []
    chunks = 0
    start = from_block
    while start <= to_block and chunks < max_chunks:
        end = min(start + chunk - 1, to_block)
        logs = await rpc(
            "eth_getLogs",
            [
                {
                    "address": address,
                    "topics": topics,
                    "fromBlock": hex(start),
                    "toBlock": hex(end),
                }
            ],
        )
        out.append(cast("list[dict[str, Any]] | None", logs) if isinstance(logs, list) else None)
        start = end + 1
        chunks += 1
    return out


async def _find_initialize_block(
    rpc: Transport, pool_id: str, latest: int, age_minutes: float | None
) -> int | None:
    """Pool Initialize block via the indexed pool id (topics[1])."""
    if age_minutes is not None:
        window = int((age_minutes + SCAN_BUFFER_MINUTES) * BLOCKS_PER_MINUTE)
    else:
        window = 200_000
    window = min(window, MAX_SCAN_BLOCKS)
    from_block = max(0, latest - window)
    logs = await rpc(
        "eth_getLogs",
        [
            {
                "address": V4_POOL_MANAGER,
                "topics": [INITIALIZE_TOPIC0, _topic_word(pool_id)],
                "fromBlock": hex(from_block),
                "toBlock": hex(latest),
            }
        ],
    )
    if not isinstance(logs, list) or not logs:
        return None
    try:
        return int(logs[0]["blockNumber"], 16)
    except (KeyError, TypeError, ValueError):
        return None


# ── Main check ──────────────────────────────────────────────────────────
async def check_bundle_and_deployer(
    token_address: str,
    pool_id: str,
    *,
    age_minutes: float | None = None,
    rpc_url: str | None = None,
    timeout_seconds: float = 20.0,
    transport: Transport | None = None,
    funder_lookup: FunderLookup | None = None,
    funder_denylist: frozenset[str] | None = None,
) -> BundleDeployerReport | None:
    """Run the bundle + deployer-cluster check on a Robinhood-chain token.

    ``pool_id`` is the v4 pool id (66-char hex, DexScreener's ``pairAddress``).
    Returns a report, or ``None`` when the check could not run (fail-open).
    """
    pool_id = (pool_id or "").lower()
    token_address = (token_address or "").lower()
    if len(pool_id) != 66 or not pool_id.startswith("0x"):
        return None
    if len(token_address) != 42 or not token_address.startswith("0x"):
        return None
    if _keccak is None:  # pragma: no cover - requirements pin pycryptodome
        return None

    denylist = {
        a.lower() for a in (funder_denylist if funder_denylist is not None else FUNDER_DENYLIST)
    }
    lookup: FunderLookup
    if funder_lookup is not None:
        # Injected lookup (tests) — no HTTP session needed.
        lookup = funder_lookup
        need_session = False
    else:
        lookup = _default_funder_lookup
        need_session = True
    transfer_t0 = _transfer_topic0()
    pm_word = _topic_word(V4_POOL_MANAGER)

    own_transport = transport is None
    rpc = transport
    session: aiohttp.ClientSession | None = None
    try:
        if own_transport:
            url = rpc_url or os.getenv("ROBINHOOD_RPC_URL", "") or ROBINHOOD_RPC_URL_DEFAULT
            rpc = await _http_transport(url, timeout_seconds)
        if need_session and session is None:
            session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "FENRIR/2.0 bundle-check"},
            )
        assert rpc is not None
        # The default lookup needs a real session (guaranteed above); an
        # injected lookup ignores the session entirely (None in tests).
        lookup_session: Any = session

        latest_raw = await rpc("eth_blockNumber", [])
        if not isinstance(latest_raw, str):
            return None
        latest = int(latest_raw, 16)

        # 1. Anchor the launch window on the pool's Initialize event.
        init_block = await _find_initialize_block(rpc, pool_id, latest, age_minutes)
        partial = False
        if init_block is None:
            from_block = max(0, latest - FALLBACK_WINDOW_BLOCKS)
            to_block = latest
            partial = True
        else:
            from_block = init_block
            to_block = init_block + LAUNCH_WINDOW_BLOCKS

        # 2. Buyers: transfers OUT of the PoolManager (topic1 == PM).
        # 2. Buyer flows: transfers OUT of the PoolManager (topic1 == PM) are
        # buys; transfers back INTO the PM (topic2 == PM) are sells/liquidity
        # adds. Net per wallet over the launch window — NOT gross first buys:
        # on a 0.1s-block chain the same tokens recycle PM→A→PM→B in seconds,
        # so gross accounting double-counts and can exceed 100% of supply
        # (seen live on FARTDOG: 118%). Net also zeroes out routers and
        # high-churn bots, which is what we want.
        chunks = await _get_logs(
            rpc,
            token_address,
            [transfer_t0, pm_word],
            from_block,
            to_block,
            chunk=to_block - from_block + 1,
            max_chunks=1,
        )
        sell_chunks = await _get_logs(
            rpc,
            token_address,
            [transfer_t0, None, pm_word],
            from_block,
            to_block,
            chunk=to_block - from_block + 1,
            max_chunks=1,
        )
        if chunks[0] is None or sell_chunks[0] is None:
            return None
        bought: dict[str, int] = {}
        sold: dict[str, int] = {}
        first_buys: dict[str, tuple[int, int]] = {}  # wallet -> (first buy block, _)
        for entry in chunks[0]:
            topics = entry.get("topics") or []
            if len(topics) < 3:
                continue
            buyer = _topic_address(str(topics[2]))
            if buyer in _PLUMBING or buyer == token_address:
                continue
            try:
                block = int(entry["blockNumber"], 16)
                amount = int(str(entry.get("data", "0x0")), 16)
            except (KeyError, TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            bought[buyer] = bought.get(buyer, 0) + amount
            if buyer not in first_buys or block < first_buys[buyer][0]:
                first_buys[buyer] = (block, 0)
        for entry in sell_chunks[0]:
            topics = entry.get("topics") or []
            if len(topics) < 3:
                continue
            seller = _topic_address(str(topics[1]))
            if seller in _PLUMBING or seller == token_address:
                continue
            try:
                amount = int(str(entry.get("data", "0x0")), 16)
            except (KeyError, TypeError, ValueError):
                continue
            if amount > 0:
                sold[seller] = sold.get(seller, 0) + amount
        # Net accumulation per wallet; floored at 0 (a net seller is not a
        # holder). Sum of nets can never exceed the pool's net outflow.
        net_accum: dict[str, int] = {w: max(0, bought.get(w, 0) - sold.get(w, 0)) for w in bought}

        # 3. Total supply for percentage math.
        supply_raw = await _eth_call(rpc, token_address, "0x" + _selector("totalSupply()").hex())
        if not supply_raw or len(supply_raw) < 32:
            return None
        total_supply = int.from_bytes(supply_raw[:32], "big")
        if total_supply <= 0:
            return None

        # 4. Timing clusters.
        timing = _timing_clusters(first_buys)

        # 5. Funder clusters: timing members first, then top buyers by amount.
        candidates: list[str] = []
        seen: set[str] = set()
        for cluster in timing:
            for w in cluster:
                if w not in seen:
                    seen.add(w)
                    candidates.append(w)
        for w in sorted(net_accum, key=lambda w: net_accum[w], reverse=True):
            if w not in seen:
                seen.add(w)
                candidates.append(w)
            if len(candidates) >= MAX_FUNDER_TRACES:
                break
        candidates = candidates[:MAX_FUNDER_TRACES]

        sem = asyncio.Semaphore(FANOUT)

        async def _trace(wallet: str) -> tuple[str, str | None]:
            block, _amt = first_buys[wallet]
            async with sem:
                try:
                    funder = await lookup(lookup_session, wallet, block)
                except Exception:  # noqa: BLE001 - fail-open per wallet
                    funder = None
            return wallet, funder.lower() if funder else None

        funders = dict(await asyncio.gather(*(_trace(w) for w in candidates)))
        by_funder: dict[str, set[str]] = {}
        for wallet, funder in funders.items():
            if funder and funder not in denylist and funder not in _PLUMBING:
                by_funder.setdefault(funder, set()).add(wallet)
        funder_clusters = [ws for ws in by_funder.values() if len(ws) >= 2]

        # 6. Union timing + funder clusters; compute supply shares.
        clusters = _merge_clusters(timing + funder_clusters)
        cluster_amounts: list[float] = []
        cluster_wallets: list[list[str]] = []
        for cluster in clusters:
            total = sum(net_accum.get(w, 0) for w in cluster)
            cluster_amounts.append(total / total_supply * 100.0)
            cluster_wallets.append(sorted(cluster))
        bundled_pct = sum(cluster_amounts) if cluster_amounts else 0.0
        largest_pct = max(cluster_amounts) if cluster_amounts else 0.0

        report = BundleDeployerReport(
            bundled_supply_pct=round(bundled_pct, 2),
            cluster_count=len(clusters),
            largest_cluster_pct=round(largest_pct, 2),
            cluster_wallets=cluster_wallets,
            launch_window_partial=partial,
        )

        # 7. Deployer tracing.
        deployer_raw = await _eth_call(rpc, token_address, "0x" + _selector("deployer()").hex())
        deployer = None
        if deployer_raw and len(deployer_raw) >= 32:
            addr = "0x" + deployer_raw[12:32].hex()
            if addr != ZERO_ADDRESS:
                deployer = addr
        if deployer:
            report.deployer_address = deployer
            bal_raw = await _eth_call(
                rpc,
                token_address,
                "0x" + (_selector("balanceOf(address)") + _addr_word(deployer)).hex(),
            )
            if bal_raw and len(bal_raw) >= 32:
                report.deployer_holding_pct = round(
                    int.from_bytes(bal_raw[:32], "big") / total_supply * 100.0, 2
                )
            flow_chunks = await _get_logs(
                rpc,
                token_address,
                [transfer_t0, _topic_word(deployer)],
                init_block or from_block,
                latest,
            )
            recipients: dict[str, int] = {}
            for chunk_logs in flow_chunks:
                if chunk_logs is None:
                    continue
                for entry in chunk_logs:
                    topics = entry.get("topics") or []
                    if len(topics) < 3:
                        continue
                    to = _topic_address(str(topics[2]))
                    if to in _PLUMBING or to == token_address or to == deployer:
                        continue
                    try:
                        amount = int(str(entry.get("data", "0x0")), 16)
                    except (TypeError, ValueError):
                        continue
                    if amount > 0:
                        recipients[to] = recipients.get(to, 0) + amount
            report.deployer_distributed_wallets = len(recipients)
            report.deployer_distributed_pct = round(
                sum(recipients.values()) / total_supply * 100.0, 2
            )
            # Supply that flowed through wallets one hop from the deployer.
            report.deployer_cluster_pct = report.deployer_distributed_pct
            try:
                async with sem:
                    dfunder = await lookup(lookup_session, deployer, init_block)
            except Exception:  # noqa: BLE001 - fail-open
                dfunder = None
            if dfunder:
                report.deployer_funder = dfunder.lower()
            # Serial-launcher check: the deployer's funder OR the deployer
            # itself (covers devs who fund their own launches — the common
            # case, e.g. FARTDOG's dev).
            serial = SERIAL_LAUNCHERS.get((report.deployer_funder or "").lower())
            serial_self = SERIAL_LAUNCHERS.get(deployer.lower())
            if serial or serial_self:
                report.deployer_funder_is_serial_launcher = True
            elif dfunder:
                report.deployer_funder_is_serial_launcher = False
            # else: None — Blockscout found no funder, stay inconclusive.

        # 8. Human-readable detail.
        bits = [
            f"bundled {report.bundled_supply_pct:.1f}% of supply across "
            f"{report.cluster_count} cluster(s) (largest {report.largest_cluster_pct:.1f}%)"
        ]
        if report.deployer_address:
            bits.append(
                f"deployer {report.deployer_address[:10]}… holds "
                f"{report.deployer_holding_pct if report.deployer_holding_pct is not None else '?'}%, "
                f"distributed to {report.deployer_distributed_wallets} wallets"
            )
            if report.deployer_funder_is_serial_launcher:
                bits.append("deployer/funder is a known serial launcher")
        if partial:
            bits.append("launch window partial (Initialize predates scan range)")
        report.detail = "; ".join(bits)
        return report
    except Exception as e:  # noqa: BLE001 - fail-open
        log.debug("check_bundle_and_deployer failed for %s: %s", token_address[:14], e)
        return None
    finally:
        if own_transport:
            if session is not None:
                await session.close()
            rpc_session = getattr(rpc, "_session", None) if rpc is not None else None
            if rpc_session is not None:
                await rpc_session.close()
