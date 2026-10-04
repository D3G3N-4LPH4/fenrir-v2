"""On-chain Robinhood-chain safety reader — the Perceptor replacement.

Perceptor gated its API behind wallet sign-in on 2026-10-02 (POST ->
403 SIGN_IN_REQUIRED; cached GETs return superseded/null verdicts), so new
safety investigations are impossible. This provider replaces it with direct
on-chain reads, filling the same ``SafetySignals`` fields so nothing
downstream changes.

Readers (all fail-open; ``None`` = unknown, never a negative):
  contract powers   ``owner()`` + bytecode selector scan (EIP-1967 aware)
                    -> ownership_renounced, mint_disabled, blacklist_present,
                    freeze_disabled
  selling works     recent v4 Swap events showing the token flowing INTO the
                    pool (real sells settled on-chain) -> honeypot=False.
                    No sells observed leaves honeypot unknown, NOT True.
  LP lock           initial-LP pullability per pool (PositionManager mints
                    near pool creation), aggregated -> lp_locked_pct
  pool liveness     main-pool reserves ~zero -> "main pool is empty" flag;
                    deep-check qualification -> "dev can pull liquidity"

Not covered in v1 (left as unknown): buy/sell tax %. DEX-trade taxes need
a full swap simulation; a transfer-only probe would give false confidence,
so the fields stay ``None`` and the filters' unknown-tax policy applies.

Budget: the scout wraps enrichment in a 45s wait_for. Every leg here is
bounded (small block windows, single pool when snap.pair_address is set)
to land comfortably inside it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

import aiohttp

from fenrir.discovery.lp_lock_v4 import ROBINHOOD_RPC_URL_DEFAULT
from fenrir.discovery.models import SafetySignals
from fenrir.discovery.pool_deepcheck import deepcheck_pools

logger = logging.getLogger(__name__)

CACHE_PATH = os.path.expanduser("~/.cache/fenrir/robinhood_safety.json")
CACHE_TTL_SECONDS = 24 * 3600


@dataclass
class SafetyReport:
    """On-chain safety verdict: safety signals + human summary."""

    safety: SafetySignals = field(default_factory=SafetySignals)
    band: str | None = None  # low | medium | high
    band_label: str | None = None  # e.g. "Caution"
    headline: str | None = None
    checks: list[tuple[str, str]] = field(default_factory=list)
    signals: list[tuple[str, str]] = field(default_factory=list)  # (label, tone)
    investigation_id: str | None = None


# Backwards-compatible alias (was the Perceptor verdict class).
PerceptorReport = SafetyReport

# ── Contract-power selectors ─────────────────────────────────────────
_SEL_OWNER = "8da5cb5b"  # owner()
_SEL_MINT = "40c10f19"  # mint(address,uint256) — the common mint
_SEL_BLACKLIST = "f9f92be4"  # blacklist(address) — common variant
_SEL_PAUSE = "8456cb59"  # pause()
# EIP-1967 implementation slot (proxies hide selectors in the impl).
_EIP1967_SLOT = "0x360894a13ba1a3210667c828832db98dca3e2076cc3735a920a3ca505d382bbc"
_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


# v4 Swap event topic0 (with trailing uint24 fee param — see pool_deepcheck).
# Reused from the canonical computation there; None when keccak unavailable.
def _swap_topic0() -> str | None:
    try:
        from fenrir.discovery.pool_deepcheck import _swap_topic0 as _canonical

        return _canonical()
    except Exception:  # noqa: BLE001
        return None


# Swap-event scan window for the behavioral sell check.
_SELL_SCAN_BLOCKS = 50_000
# Deep-check budget inside this provider (scout gives us 45s total).
_DEEPCHECK_BUDGET_S = 15.0


def _sanitize_proxy_env() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(key, "")
        if "[" in val:
            cleaned = ",".join(p for p in val.split(",") if "[" not in p)
            os.environ[key] = cleaned


async def _http_transport(rpc_url: str, timeout_seconds: float = 15.0):
    _sanitize_proxy_env()
    session = aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        headers={"User-Agent": "FENRIR/2.0 robinhood-safety"},
    )

    async def _rpc(method: str, params: list):
        try:
            async with session.post(
                rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            ) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
            if "error" in payload:
                return None
            return payload.get("result")
        except Exception as e:  # noqa: BLE001 - fail-open
            logger.debug("robinhood-safety RPC %s failed: %s", method, e)
            return None

    _rpc._session = session  # type: ignore[attr-defined]
    return _rpc


async def _eth_call(rpc, to: str, calldata: str) -> bytes | None:
    result = await rpc("eth_call", [{"to": to, "data": calldata}, "latest"])
    if isinstance(result, str) and result.startswith("0x"):
        try:
            return bytes.fromhex(result.removeprefix("0x"))
        except ValueError:
            return None
    return None


def _decode_address_word(raw: bytes) -> str | None:
    if len(raw) != 32:
        return None
    return "0x" + raw[12:].hex()


# ── Reader 1: contract powers ────────────────────────────────────────


async def read_contract_powers(rpc, token: str) -> dict:
    """owner() + bytecode selector scan. Returns partial dict, missing = unknown."""
    out: dict = {}
    raw = await _eth_call(rpc, token, "0x" + _SEL_OWNER)
    owner = _decode_address_word(raw) if raw else None
    if owner is not None:
        renounced = owner.lower() == _ZERO_ADDRESS
        out["ownership_renounced"] = renounced
        out["owner_live"] = not renounced
    code = await rpc("eth_getCode", [token, "latest"])
    blob = ""
    if isinstance(code, str) and code.startswith("0x"):
        blob = code[2:].lower()
        # Proxy? Also scan the implementation's bytecode.
        impl_raw = await rpc("eth_getStorageAt", [token, _EIP1967_SLOT, "latest"])
        if isinstance(impl_raw, str) and impl_raw.startswith("0x"):
            impl = "0x" + impl_raw[-40:]
            if impl.lower() != _ZERO_ADDRESS:
                impl_code = await rpc("eth_getCode", [impl, "latest"])
                if isinstance(impl_code, str) and impl_code.startswith("0x"):
                    blob += impl_code[2:].lower()
    if blob:
        has_mint = _SEL_MINT in blob
        # Mint confidence: renounced => nobody can mint. Live owner + mint
        # selector => can mint. Live owner without a recognizable selector =>
        # UNKNOWN (never assert safe) — custom mint fns slip past the scan.
        if out.get("ownership_renounced") is True:
            out["mint_disabled"] = True
        elif out.get("owner_live"):
            out["mint_disabled"] = False if has_mint else None
        elif not has_mint:
            out["mint_disabled"] = True
        out["blacklist_present"] = _SEL_BLACKLIST in blob
        out["freeze_disabled"] = _SEL_PAUSE not in blob
    return out


# ── Reader 2: behavioral sell check ──────────────────────────────────


# v4 PoolManager (emits Initialize + Swap; address filter lifts the RPC's
# 30k-block log range limit).
_V4_POOL_MANAGER = "0x8366a39CC670B4001A1121B8F6A443A643e40951"


async def _pool_init_info(rpc, pool_id: str, token_lc: str) -> dict:
    """Initialize block + currency order for a v4 pool. {} when unresolvable."""
    from fenrir.discovery.bundle_check import INITIALIZE_TOPIC0

    try:
        latest_raw = await rpc("eth_blockNumber", [])
        latest = int(latest_raw, 16)
    except (TypeError, ValueError):
        return {}
    # RPC caps log queries at 10M blocks even with an address filter;
    # 10M blocks ≈ 23 days — far older than any scout candidate pool.
    logs = await rpc(
        "eth_getLogs",
        [
            {
                "address": _V4_POOL_MANAGER,
                "fromBlock": hex(max(0, latest - 10_000_000)),
                "toBlock": "latest",
                "topics": [INITIALIZE_TOPIC0, pool_id],
            }
        ],
    )
    if not isinstance(logs, list):
        return {}
    for log in logs:
        try:
            topics = log.get("topics") or []
            c0 = ("0x" + topics[2][-40:]).lower()
            c1 = ("0x" + topics[3][-40:]).lower()
            if c0 != token_lc and c1 != token_lc:
                continue
            return {
                "pool_id": pool_id,
                "init_block": int(log.get("blockNumber", "0x0"), 16),
                "token_is_c0": c0 == token_lc,
            }
        except (IndexError, TypeError, AttributeError, ValueError):
            continue
    return {}


async def _swap_sells_observed(
    rpc, pool_id: str, token_lc: str, token_is_c0: bool | None = None
) -> bool | None:
    """True when recent v4 Swap events show the token flowing INTO the pool.

    A sell is the token paid into the pool: amount0 > 0 when the token is
    currency0, amount1 > 0 when it is currency1 (v4 Swap deltas are signed;
    positive = into the pool). Returns None when the scan fails; False only
    when swaps were seen but none were sells of this token.
    """
    topic0 = _swap_topic0()
    if not topic0:
        return None
    if token_is_c0 is None:
        info = await _pool_init_info(rpc, pool_id, token_lc)
        token_is_c0 = info.get("token_is_c0")
    if token_is_c0 is None:
        return None
    try:
        latest_raw = await rpc("eth_blockNumber", [])
        latest = int(latest_raw, 16)
    except (TypeError, ValueError):
        return None
    logs = await rpc(
        "eth_getLogs",
        [
            {
                "address": _V4_POOL_MANAGER,
                "fromBlock": hex(max(0, latest - _SELL_SCAN_BLOCKS)),
                "toBlock": "latest",
                "topics": [topic0, pool_id],
            }
        ],
    )
    if not isinstance(logs, list):
        return None
    seen_swap = False
    for log in logs:
        try:
            data = bytes.fromhex((log.get("data") or "0x").removeprefix("0x"))
            if len(data) < 64:
                continue
            seen_swap = True
            a0 = int.from_bytes(data[0:32], "big", signed=True)
            a1 = int.from_bytes(data[32:64], "big", signed=True)
            sold = (token_is_c0 and a0 > 0) or (not token_is_c0 and a1 > 0)
            if sold:
                return True
        except (ValueError, TypeError, IndexError):
            continue
    return False if seen_swap else None


# ── Reader 3: LP lock (initial-LP focus) ──────────────────────────────


async def _initial_lp_pullable(rpc, pool_id: str, init_block: int) -> bool | None:
    """Is the pool's INITIAL LP (minted near creation) EOA-held?

    The launch liquidity is the rug-relevant LP. Scans PositionManager mints
    in a tight window after pool creation — a handful of RPC calls instead
    of the full position census. True = pullable, False = locked/burned,
    None = unknown.
    """
    from fenrir.discovery.lp_lock_v4 import (
        BURN_ADDRESSES,
        KNOWN_LOCKERS,
        _classify_holders,
        _fetch_mint_token_ids,
        _is_eoa,
        _positions_for_pool,
    )

    try:
        token_ids = await _fetch_mint_token_ids(rpc, init_block, init_block + 20_000)
        if not token_ids:
            return None
        pool_tokens = await _positions_for_pool(rpc, pool_id, token_ids[:200])
        if not pool_tokens:
            return None
        holdings = await _classify_holders(rpc, pool_tokens)
        if not holdings:
            return None
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.debug("initial LP scan failed: %s", e)
        return None
    for h in holdings:
        o = h.owner.lower()
        if o in BURN_ADDRESSES or o in KNOWN_LOCKERS:
            continue
        try:
            eoa = await _is_eoa(rpc, h.owner)
        except Exception:  # noqa: BLE001
            return None
        if eoa is True:
            return True
        if eoa is None:
            return None
    return False


async def read_lp_lock(rpc, pool_infos: list[dict]) -> dict:
    """Aggregate initial-LP checks -> {locked_pct, locked_or_burned}."""
    if not pool_infos:
        return {}
    locked_flags = []
    for info in pool_infos[:3]:  # bounded: top pools only
        pid = info.get("pool_id")
        init_block = info.get("init_block")
        if not pid or not init_block:
            continue
        pullable = await _initial_lp_pullable(rpc, pid, init_block)
        if pullable is not None:
            locked_flags.append(not pullable)
    if not locked_flags:
        return {}
    pct = round(100.0 * sum(locked_flags) / len(locked_flags), 1)
    return {"lp_locked_pct": pct, "lp_locked_or_burned": pct >= 90.0}


# ── Assembly ─────────────────────────────────────────────────────────


def _band_for(safety: SafetySignals) -> tuple[str, str]:
    """Our band: high/medium/low + label. Filters never gated on Perceptor's
    band either, so this is informational."""
    flags = safety.risk_flags
    if safety.honeypot:
        return "high", "Honeypot"
    if any("pull" in f for f in flags) or safety.blacklist_present:
        return "high", "High risk"
    if flags or safety.mint_disabled is False:
        return "medium", "Caution"
    return "low", "Clean"


async def read_robinhood_safety(
    token_address: str,
    *,
    rpc_url: str | None = None,
    pair_address: str | None = None,
    timeout_seconds: float = 40.0,
) -> SafetyReport | None:
    """Full on-chain safety read for a Robinhood-chain token.

    Returns a SafetyReport verdict or None when nothing usable came back.
    Never raises.
    """
    token = (token_address or "").lower()
    if not token.startswith("0x") or len(token) != 42:
        return None
    url = rpc_url or os.getenv("ROBINHOOD_RPC_URL", "") or ROBINHOOD_RPC_URL_DEFAULT
    t0 = time.monotonic()
    safety = SafetySignals()
    rpc = await _http_transport(url, timeout_seconds=12.0)
    try:
        # 1. contract powers (cheap: 2-4 RPC calls)
        powers = await read_contract_powers(rpc, token)
        if powers.get("ownership_renounced") is not None:
            safety.ownership_renounced = powers["ownership_renounced"]
        if powers.get("mint_disabled") is not None:
            safety.mint_disabled = powers["mint_disabled"]
        if powers.get("blacklist_present") is not None:
            safety.blacklist_present = powers["blacklist_present"]
            if powers["blacklist_present"]:
                safety.risk_flags.append("blacklist function present")
        if powers.get("owner_live"):
            safety.risk_flags.append("owner live")
        if powers.get("freeze_disabled") is not None:
            safety.freeze_disabled = powers["freeze_disabled"]
            if not powers["freeze_disabled"]:
                safety.risk_flags.append("pausable")
        if safety.mint_disabled is False:
            safety.risk_flags.append("mint authority live")

        # 2-4. pool set, sell check, LP lock — one Initialize read per pool
        # serves the currency order (sells) and the init block (LP window).
        pool_infos: list[dict] = []
        if pair_address and len(pair_address) == 66 and pair_address.startswith("0x"):
            pool_infos = [{"pool_id": pair_address.lower()}]
        else:
            try:
                dc = await asyncio.wait_for(
                    deepcheck_pools(token, rpc_url=url, budget_seconds=_DEEPCHECK_BUDGET_S),
                    timeout=_DEEPCHECK_BUDGET_S + 5,
                )
                pool_infos = [{"pool_id": p.pool_id} for p in (dc.pools or [])[:5]]
                if dc.qualification == "PULLABLE":
                    safety.risk_flags.append("dev can pull liquidity")
                elif dc.qualification == "MISATTRIBUTED":
                    safety.risk_flags.append("main pool misattributed")
            except Exception as e:  # noqa: BLE001 - fail-open
                logger.debug("deepcheck enumeration failed: %s", e)
        for info in pool_infos:
            init = await _pool_init_info(rpc, info["pool_id"], token)
            info.update(init)
        pool_infos = [i for i in pool_infos if i.get("init_block")]

        if pool_infos:
            main = pool_infos[0]
            sells = await _swap_sells_observed(
                rpc, main["pool_id"], token, token_is_c0=main.get("token_is_c0")
            )
            if sells is True:
                safety.honeypot = False
            # sells is False/None -> honeypot stays unknown (never assert True)

        lp = await read_lp_lock(rpc, pool_infos)
        lp_pct = lp.get("lp_locked_pct")
        if lp_pct is not None:
            safety.lp_locked_pct = lp_pct
            safety.lp_locked_or_burned = lp["lp_locked_or_burned"]
            if not lp["lp_locked_or_burned"] and lp_pct < 50:
                safety.risk_flags.append("dev can pull liquidity")

        if safety.is_empty:
            return None
        band, label = _band_for(safety)
        headline = "; ".join(safety.risk_flags) if safety.risk_flags else None
        return SafetyReport(
            safety=safety,
            band=band,
            band_label=label,
            headline=headline,
            checks=[],
            signals=[(f, "high") for f in safety.risk_flags],
            investigation_id=f"local:{token[:10]}",
        )
    except Exception as e:  # noqa: BLE001 - never raise out of enrichment
        logger.debug("robinhood-safety read failed for %s: %s", token[:10], e)
        return None
    finally:
        try:
            await rpc._session.close()  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            logger.debug("session close failed: %s", e)
        logger.debug("robinhood-safety read for %s took %.1fs", token[:10], time.monotonic() - t0)


# ── Orchestration ────────────────────────────────────────────────────


async def enrich_robinhood_safety(snap, local_provider=None):
    """Fill empty Robinhood-chain safety, best-effort. Never raises.

    Uses the local on-chain reader only (fresh, authoritative). Perceptor
    was retired 2026-10-03 (API auth-walled since 2026-10-02; its sweep
    returned zero completions) — the on-chain reader is the sole safety
    source now. Merges a usable verdict into ``snap.safety``.
    Returns the report, or None.
    """
    from fenrir.discovery.models import Chain

    if snap.chain is not Chain.ROBINHOOD:
        return None
    if not snap.safety.is_empty:
        if local_provider is not None:
            hit = local_provider.cached_report(snap.token_address)
            if hit is not None:
                return hit
        return None
    if local_provider is not None:
        try:
            report = await local_provider.ensure_report(
                snap.token_address,
                pair_address=getattr(snap, "pair_address", None),
            )
        except Exception as e:  # noqa: BLE001 - fail-open
            logger.debug("local safety read failed: %s", e)
            report = None
        if report is not None:
            snap.safety = report.safety
            return report
    return None


# ── Provider (Perceptor-compatible interface) ────────────────────────


class RobinhoodSafetyProvider:
    """On-chain Robinhood safety reader: disk cache + live chain reads."""

    def __init__(self, cache_path: str | None = None, rpc_url: str | None = None) -> None:
        self.cache_path = cache_path or CACHE_PATH
        self.rpc_url = rpc_url

    def _load_cache(self) -> dict:
        try:
            with open(self.cache_path) as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self, data: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.cache_path)
        except OSError:
            pass

    def cached_report(self, address: str):
        entry = self._load_cache().get(address.lower())
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > CACHE_TTL_SECONDS:
            return None
        s = SafetySignals(
            **{
                k: v
                for k, v in (entry.get("safety") or {}).items()
                if k in SafetySignals.__dataclass_fields__
            }
        )
        return SafetyReport(
            safety=s,
            band=entry.get("band"),
            band_label=entry.get("band_label"),
            headline=entry.get("headline"),
            checks=[],
            signals=entry.get("signals") or [],
            investigation_id=entry.get("investigation_id"),
        )

    async def ensure_report(
        self,
        address: str,
        pair_address: str | None = None,
    ):
        """Fresh on-chain read (or 24h cache); never raises."""
        hit = self.cached_report(address)
        if hit is not None:
            return hit
        report = await read_robinhood_safety(
            address,
            rpc_url=self.rpc_url,
            pair_address=pair_address,
        )
        if report is None:
            return None
        cache = self._load_cache()
        cache[address.lower()] = {
            "ts": time.time(),
            "safety": {k: getattr(report.safety, k) for k in SafetySignals.__dataclass_fields__},
            "band": report.band,
            "band_label": report.band_label,
            "headline": report.headline,
            "signals": report.signals,
            "investigation_id": report.investigation_id,
        }
        self._save_cache(cache)
        return report
