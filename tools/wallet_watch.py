#!/usr/bin/env python3
"""FENRIR - wallet watch: turn tracked-wallet buys into scout candidates.

Polls a curated wallet list and, when a tracked wallet acquires a NEW token,
runs that token through the same FENRIR pipeline as the scout (snapshot +
safety + entry filters + 0-100 score). Emits candidate dicts in the scout
schema (source "wallet:<label>") so the existing alert / dedup / gate-tracker
flow consumes them unchanged.

Buy detection is venue-agnostic per chain:
- Solana: pre/post SPL-token balance deltas per transaction (catches pump.fun
  bonding-curve buys, AMM swaps, Jupiter routes without decoding any
  instruction). WSOL / stablecoins / LSTs are excluded via is_tradeable_mint.
- Robinhood chain: ERC20 Transfer events with `to` == wallet. DEX fills land
  as transfers to the buyer, so this catches swaps too. Base currencies
  (WETH, USDG) are excluded.

Wallet list: JSON file {"solana": [{"address": ..., "label": ...}],
"robinhood": [...]}. Labels are free-form (e.g. "alice", "smart-1").

State: {"solana": {addr: {"last_sigs": [...]}}, "robinhood": {addr:
{"last_block": N}}}. The first-ever run seeds state WITHOUT alerting, so we
only fire on buys that happen after tracking starts (no backlog blast).

Usage:
    python tools/wallet_watch.py --min-score 60 > /tmp/wallet_out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.scout import evaluate_address  # noqa: E402
from fenrir.discovery.acceleration import AccelTracker  # noqa: E402
from fenrir.discovery.filters import FilterEngine  # noqa: E402
from fenrir.discovery.playbooks import PlaybookTagger  # noqa: E402
from fenrir.discovery.providers.dexscreener import DexScreenerProvider  # noqa: E402
from fenrir.discovery.providers.goplus import GoPlusProvider  # noqa: E402
from fenrir.discovery.providers.robinhood_safety import RobinhoodSafetyProvider  # noqa: E402
from fenrir.discovery.scoring import ScoringEngine  # noqa: E402
from fenrir.discovery.models import Chain  # noqa: E402
from fenrir.trading.token_filters import is_tradeable_mint  # noqa: E402

# ERC20 Transfer(address,address,uint256)
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3e6"
WETH_ROBINHOOD = "0x0bd7d308f8e1639faab988df18a8011f41eacad73"
USDG_ROBINHOOD = "0x5fc5360d0400a0fd4f2af552add042d716f1d168"
ZERO_ADDRESS = "0x" + "0" * 40
BASE_CURRENCIES = {WETH_ROBINHOOD, USDG_ROBINHOOD, ZERO_ADDRESS}

DEFAULT_WALLETS_PATH = os.path.expanduser(
    "~/workspace/goals/token-scout-watch/hidden_files/wallets.json"
)
DEFAULT_STATE_PATH = os.path.expanduser(
    "~/workspace/goals/token-scout-watch/hidden_files/wallet_watch.json"
)
PUBLIC_RH_RPC = "https://rpc.mainnet.chain.robinhood.com"


def _pad_topic(addr: str) -> str:
    return "0x" + "0" * 24 + addr.lower().removeprefix("0x")


async def _rpc(session: aiohttp.ClientSession, url: str, method: str, params: list) -> Any:
    async with session.post(
        url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    ) as resp:
        data = await resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{method}: {data['error']}")
    return data["result"] if isinstance(data, dict) else data


# ---------------------------------------------------------------- Solana ---


def detect_solana_buys(tx_result: dict, wallet: str) -> list[tuple[str, float]]:
    """(mint, sol_spent) for mints whose balance INCREASED for `wallet`.

    Pure function over a getTransaction jsonParsed result — venue-agnostic.
    Returns [] when the tx has no usable meta.
    """
    meta = (tx_result or {}).get("meta") or {}
    pre = {
        (b.get("owner"), b.get("mint")): _ui_amount(b) for b in (meta.get("preTokenBalances") or [])
    }
    post = {
        (b.get("owner"), b.get("mint")): _ui_amount(b)
        for b in (meta.get("postTokenBalances") or [])
    }
    buys = [
        mint
        for (owner, mint), amt in post.items()
        if owner == wallet and is_tradeable_mint(mint) and amt > pre.get((owner, mint), 0.0) + 1e-9
    ]
    if not buys:
        return []
    return [(mint, _sol_spent(tx_result, wallet, meta)) for mint in buys]


def _ui_amount(balance_entry: dict) -> float:
    try:
        return float(balance_entry.get("uiTokenAmount", {}).get("uiAmount") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sol_spent(tx_result: dict, wallet: str, meta: dict) -> float:
    """Best-effort native SOL the wallet spent (pre/post lamport delta)."""
    try:
        keys = tx_result.get("transaction", {}).get("message", {}).get("accountKeys") or []
        norm = [k.get("pubkey") if isinstance(k, dict) else k for k in keys]
        idx = norm.index(wallet)
        pre = int(meta.get("preBalances", [])[idx])
        post = int(meta.get("postBalances", [])[idx])
        return max(0.0, (pre - post) / 1e9)
    except Exception:  # noqa: BLE001 - context flavor only
        return 0.0


async def poll_solana_wallet(
    session: aiohttp.ClientSession, rpc_url: str, wallet: str, seen: set[str], limit: int = 15
) -> tuple[list[tuple[str, float]], set[str]]:
    """New (mint, sol_spent) buys for a Solana wallet since `seen` signatures."""
    sigs = await _rpc(
        session,
        rpc_url,
        "getSignaturesForAddress",
        [wallet, {"limit": limit, "commitment": "finalized"}],
    )
    fresh = [s for s in reversed(sigs or []) if s.get("signature") not in seen and not s.get("err")]
    buys: list[tuple[str, float]] = []
    for s in fresh:
        sig = s["signature"]
        seen.add(sig)
        try:
            tx = await _rpc(
                session,
                rpc_url,
                "getTransaction",
                [
                    sig,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0,
                        "commitment": "finalized",
                    },
                ],
            )
        except Exception:  # noqa: BLE001 - one bad fetch shouldn't stop the poll
            continue
        if tx is None:
            continue
        buys.extend(detect_solana_buys(tx, wallet))
    # De-dup by mint within this poll, keep first sighting.
    dedup: dict[str, float] = {}
    for mint, sol in buys:
        dedup.setdefault(mint, sol)
    return list(dedup.items()), seen


# --------------------------------------------------------------- Robinhood ---


def detect_robinhood_buys(logs: list[dict]) -> list[str]:
    """Token contracts acquired via Transfer-to-wallet logs. Pure function."""
    mints: list[str] = []
    for log in logs or []:
        token = (log.get("address") or "").lower()
        if not token or token in BASE_CURRENCIES:
            continue
        if token not in mints:
            mints.append(token)
    return mints


async def poll_robinhood_wallet(
    session: aiohttp.ClientSession, rpc_url: str, wallet: str, last_block: int | None
) -> tuple[list[str], int]:
    """New token contracts a Robinhood-chain wallet acquired since last_block."""
    latest = int(await _rpc(session, rpc_url, "eth_blockNumber", []), 16)
    if last_block is None:
        # First run: seed just behind the tip, don't blast the backlog.
        return [], latest - 5
    from_block = last_block + 1
    if from_block > latest:
        return [], latest
    logs = await _rpc(
        session,
        rpc_url,
        "eth_getLogs",
        [
            {
                "topics": [TRANSFER_TOPIC0, None, _pad_topic(wallet)],
                "fromBlock": hex(from_block),
                "toBlock": hex(latest),
            }
        ],
    )
    return detect_robinhood_buys(logs), latest


# ------------------------------------------------------------------- main ---


def load_json(path: str, default: object) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


async def amain() -> int:
    ap = argparse.ArgumentParser(description="FENRIR wallet watch")
    ap.add_argument("--wallets", default=DEFAULT_WALLETS_PATH)
    ap.add_argument("--state", default=DEFAULT_STATE_PATH)
    ap.add_argument("--min-score", type=float, default=60.0)
    ap.add_argument(
        "--solana-rpc", default=os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    )
    ap.add_argument("--robinhood-rpc", default=os.getenv("ROBINHOOD_RPC_URL", PUBLIC_RH_RPC))
    args = ap.parse_args()

    wallets_cfg = load_json(args.wallets, {}) or {}
    state = load_json(args.state, {}) or {}
    sol_state: dict = state.setdefault("solana", {})
    rh_state: dict = state.setdefault("robinhood", {})

    ds = DexScreenerProvider(timeout_seconds=15)
    gp = GoPlusProvider(timeout_seconds=10)
    local_safety = RobinhoodSafetyProvider()
    accel = AccelTracker(AccelTracker.default_state_path())
    engine = FilterEngine()
    scorer = ScoringEngine()
    tagger = PlaybookTagger()

    # mint -> {"chain": Chain, "wallets": [(label, sol_spent)], "first_label": str}
    buys: dict[str, dict] = {}
    wallets_checked = 0
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(
            trust_env=True,
            timeout=timeout,
            headers={"User-Agent": "FENRIR/2.0 wallet-watch"},
        ) as session:
            for entry in wallets_cfg.get("solana") or []:
                addr, label = (
                    entry.get("address"),
                    entry.get("label") or entry.get("address", "")[:8],
                )
                if not addr:
                    continue
                wallets_checked += 1
                seen = set((sol_state.get(addr) or {}).get("last_sigs") or [])
                first_run = not seen
                try:
                    new_buys, seen = await poll_solana_wallet(session, args.solana_rpc, addr, seen)
                except Exception:  # noqa: BLE001 - one wallet failing skips it
                    continue
                # Bound the seen set; persist even on first run (seed, no alert).
                sol_state[addr] = {"last_sigs": sorted(seen)[-300:]}
                if first_run:
                    continue
                for mint, sol_spent in new_buys:
                    slot = buys.setdefault(
                        mint, {"chain": Chain.SOLANA, "wallets": [], "label": label}
                    )
                    slot["wallets"].append({"label": label, "sol_spent": round(sol_spent, 4)})

            for entry in wallets_cfg.get("robinhood") or []:
                addr, label = (
                    entry.get("address"),
                    entry.get("label") or entry.get("address", "")[:10],
                )
                if not addr:
                    continue
                wallets_checked += 1
                last_block = (rh_state.get(addr) or {}).get("last_block")
                try:
                    mints, latest = await poll_robinhood_wallet(
                        session, args.robinhood_rpc, addr, last_block
                    )
                except Exception:  # noqa: BLE001
                    continue
                rh_state[addr] = {"last_block": latest}
                if last_block is None:
                    continue  # seeded, no backlog blast
                for mint in mints:
                    slot = buys.setdefault(
                        mint, {"chain": Chain.ROBINHOOD, "wallets": [], "label": label}
                    )
                    slot["wallets"].append({"label": label})
    finally:
        # Persist state even if evaluation below fails.
        try:
            os.makedirs(os.path.dirname(args.state), exist_ok=True)
            with open(args.state, "w") as f:
                json.dump(state, f)
        except OSError:
            pass

    candidates: list[dict] = []
    try:
        for mint, info in buys.items():
            cand = await evaluate_address(
                f"wallet:{info['label']}",
                mint,
                info["chain"],
                ds,
                gp,
                engine,
                scorer,
                tagger,
                args.min_score,
                local_safety=local_safety,
                accel=accel,
            )
            if cand is None:
                continue
            # All tracked wallets that bought this mint this run (confluence).
            cand["wallet_buys"] = info["wallets"]
            candidates.append(cand)
    finally:
        await ds.close()
        await gp.close()
        accel.save()

    candidates.sort(key=lambda c: -c["score"]["overall"])
    print(
        json.dumps(
            {
                "ts": time.time(),
                "wallets_checked": wallets_checked,
                "buys_detected": sum(len(i["wallets"]) for i in buys.values()),
                "candidates": candidates,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
