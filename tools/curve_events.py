#!/usr/bin/env python3
"""Curve ignition watcher: catch pump.fun ignitions at block ~zero.

The poll-based discovery path (GeckoTerminal new-pools -> curve-state read)
lags launches by 10-25 minutes, so ignition plays graduate before we see
them. This daemon inverts the pipeline:

  discovery   websocket logsSubscribe on the pump.fun program -> new mints
              stream in ~1-2s (falls back to GeckoTerminal new-pools poll
              when the stream is unavailable, e.g. behind a proxy)
  tracking    every new curve enters the watchlist; its bonding-curve state
              is re-read every --poll-seconds and SOL inflow velocity is
              computed across readings (state survives restarts)
  ignition    progress 10-50% + >=1 SOL inflow in the window (+ accelerating
              when history exists) -> one Telegram alert per mint
  prune       age > 45m, progress >= 95% / complete (grad_snipe's territory),
              or no inflow for 10m

This is the ignition lane (10-50%); the scout's ``graduation`` source covers
50-85% and grad_snipe.py covers the migration itself. Nothing here trades.

Usage:
  python tools/curve_events.py                       # daemon, alerts live
  python tools/curve_events.py --dry-run             # print alerts, no send
  python tools/curve_events.py --source poll         # no websocket (sandbox)
  python tools/curve_events.py --run-once            # single poll cycle, exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import aiohttp  # noqa: E402

from fenrir.discovery.alerts import escape_md  # noqa: E402
from fenrir.discovery.models import Chain  # noqa: E402
from fenrir.discovery.providers.geckoterminal import GeckoTerminalProvider  # noqa: E402
from fenrir.discovery.providers.pumpfun import (  # noqa: E402
    PumpFunProvider,
)
from fenrir.discovery.providers.pumpfun_events import (  # noqa: E402
    CreateEvent,
    CreateEventFetcher,
    stream_create_signatures,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Ignition gates ───────────────────────────────────────────────────
# Mirrors the CURVE_IGNITION filter's intent (10-50% progress, inflow is the
# core signal) but on data available pre-indexing: on-chain curve state only.
# No DexScreener fields (volume/holders/buys) exist for a 2-minute-old curve.
MIN_PROGRESS_PCT = 10.0
MAX_PROGRESS_PCT = 50.0
MIN_INFLOW_SOL = 1.0  # SOL into the curve within the poll window
ACCEL_RATIO = 1.3  # current-window inflow must beat prior window by this
MAX_WATCH_AGE_S = 45 * 60  # older + still <50% = stalled, prune
STALE_NOFLOW_S = 10 * 60  # no inflow for this long = dead, prune
HANDOFF_PROGRESS_PCT = 95.0  # at/above: grad_snipe's territory, prune

POLL_SECONDS = 20.0
CREATE_QUEUE_MAX = 500

# Jupiter lite price API (SOL/USD for the est-mcap line; fail-open).
_SOL_PRICE_URL = (
    "https://lite-api.jup.ag/price/v3" "?ids=So11111111111111111111111111111111111111112"
)
_sol_price_cache: tuple[float, float] | None = None  # (price, fetched_at)


def _load_json(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ── Pure logic (unit tested) ─────────────────────────────────────────


def ignition_check(entry: dict, now: float) -> tuple[bool, list[str]]:
    """Apply the ignition gates to a watchlist entry. Returns (fire, reasons)."""
    fails: list[str] = []
    progress = entry.get("last_progress")
    if progress is None:
        return False, ["no curve progress"]
    if progress < MIN_PROGRESS_PCT:
        fails.append(f"progress {progress:.1f}% < {MIN_PROGRESS_PCT:.0f}%")
    if progress > MAX_PROGRESS_PCT:
        fails.append(f"progress {progress:.1f}% > {MAX_PROGRESS_PCT:.0f}%")
    inflow = entry.get("inflow_sol")
    if inflow is None:
        fails.append("no inflow reading yet")
    elif inflow < MIN_INFLOW_SOL:
        fails.append(f"inflow {inflow:.2f} SOL < {MIN_INFLOW_SOL:.0f} SOL")
    else:
        prev_inflow = entry.get("prev_inflow_sol")
        if prev_inflow is not None and prev_inflow > 0:
            if inflow < prev_inflow * ACCEL_RATIO:
                fails.append(f"not accelerating ({inflow:.2f} vs prev {prev_inflow:.2f} SOL)")
    age = now - entry.get("first_seen", now)
    if age > MAX_WATCH_AGE_S:
        fails.append(f"age {age / 60:.0f}m > {MAX_WATCH_AGE_S / 60:.0f}m")
    if entry.get("alerted"):
        fails.append("already alerted")
    return (not fails), fails


def should_prune(entry: dict, now: float) -> str | None:
    """Return a prune reason, or None to keep watching."""
    age = now - entry.get("first_seen", now)
    if age > MAX_WATCH_AGE_S:
        return f"age {age / 60:.0f}m"
    if entry.get("complete"):
        return "migrated"
    progress = entry.get("last_progress") or 0
    if progress >= HANDOFF_PROGRESS_PCT:
        return f"progress {progress:.0f}% (grad_snipe territory)"
    last_flow = entry.get("last_inflow_at") or entry.get("first_seen", now)
    if now - last_flow > STALE_NOFLOW_S and (entry.get("inflow_sol") or 0) <= 0:
        return "no inflow 10m"
    return None


def implied_mcap_sol(state) -> float | None:
    """Curve-implied mcap in SOL from virtual reserves (None on bad data)."""
    try:
        v_sol = state.virtual_sol_reserves
        v_tok = state.virtual_token_reserves
        supply = state.token_total_supply
        if not v_tok or not supply:
            return None
        result: float = (v_sol / v_tok) * supply / 1e9
        return result
    except (AttributeError, ZeroDivisionError, TypeError):
        return None


# ── Network helpers ──────────────────────────────────────────────────


async def fetch_sol_usd() -> float | None:
    """Cached SOL/USD (5 min TTL); None on failure."""
    global _sol_price_cache
    now = time.time()
    if _sol_price_cache and now - _sol_price_cache[1] < 300:
        return _sol_price_cache[0]
    try:
        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=10)
        ) as s:
            async with s.get(_SOL_PRICE_URL) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                px = data["So11111111111111111111111111111111111111112"]["usdPrice"]
                _sol_price_cache = (float(px), now)
                return float(px)
    except Exception:  # noqa: BLE001 - est-mcap is best-effort
        return None


def send_telegram(text: str) -> bool:
    import subprocess

    notify = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_notify.py")
    try:
        r = subprocess.run(  # noqa: S603 - argv is our own script + fixed flags; text is a data arg
            [sys.executable, notify, "--parse-mode", "Markdown", text],
            capture_output=True,
            text=True,
            timeout=90,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001 - fail-open
        return False


def format_ignition_alert(
    mint: str,
    entry: dict,
    est_mcap_usd: float | None,
    inflow_window_s: float,
) -> str:
    sym = escape_md(entry.get("symbol") or "???")
    name = escape_md(entry.get("name") or "")
    header = f"\u26a1 *{sym}*"
    if name and name.lower() != sym.lower().replace("\\", ""):
        header += f" \u2014 {name}"
    age_m = (time.time() - entry.get("first_seen", time.time())) / 60
    lines = [
        header,
        f"Solana \u00b7 curve ignition \u00b7 via curve_events \u00b7 {age_m:.0f}m old",
        "",
        f"\U0001f30a curve {entry.get('last_progress', 0):.0f}%"
        f" \u00b7 {entry.get('last_sol', 0):.1f} SOL in curve",
    ]
    inflow = entry.get("inflow_sol") or 0
    accel = ""
    prev = entry.get("prev_inflow_sol")
    if prev is not None and prev > 0 and inflow >= prev * ACCEL_RATIO:
        accel = " (accelerating \U0001f680)"
    lines.append(f"\U0001f4c8 inflow +{inflow:.2f} SOL / {inflow_window_s:.0f}s{accel}")
    if est_mcap_usd:
        lines.append(f"\U0001f4b0 est. mcap ~${est_mcap_usd:,.0f}")
    lines += [
        "\u26a0\ufe0f safety partial: pre-migration, distribution unverified",
        "",
        f"`{mint}`",
    ]
    return "\n".join(lines)


# ── Watcher ──────────────────────────────────────────────────────────


class CurveIgnitionWatcher:
    def __init__(
        self,
        state_dir: str,
        poll_seconds: float = POLL_SECONDS,
        dry_run: bool = False,
        source: str = "auto",
    ) -> None:
        self.state_dir = state_dir
        self.state_path = os.path.join(state_dir, "curve_events.json")
        self.poll_seconds = poll_seconds
        self.dry_run = dry_run
        self.source = source
        self.watch: dict = _load_json(self.state_path).get("mints", {})
        self.rpc_url = os.getenv("SOLANA_RPC_URL") or "https://api.mainnet-beta.solana.com"
        self.provider = PumpFunProvider(rpc_url=self.rpc_url)
        self.fetcher = CreateEventFetcher(self.rpc_url)
        self.sig_queue: asyncio.Queue = asyncio.Queue(maxsize=CREATE_QUEUE_MAX)
        self.stop = asyncio.Event()
        self.ws_ok = asyncio.Event()  # set once the stream delivers

    async def _poll_discovery(self) -> list[CreateEvent]:
        """Fallback discovery: GeckoTerminal new pools -> curve-state read."""
        gt = GeckoTerminalProvider(timeout_seconds=15)
        try:
            base = await gt.fetch_new_pool_addresses(Chain.SOLANA, 60)
            if not base:
                return []
            states = await self.provider.curve_states(base)
            now = time.time()
            out = []
            for mint, state in states.items():
                if state.complete or mint in self.watch:
                    continue
                out.append(
                    CreateEvent(
                        mint=mint,
                        bonding_curve=None,
                        creator=getattr(state, "creator", None),
                        name="",
                        symbol="",
                        uri="",
                        signature="poll",
                        slot=0,
                        seen_at=now,
                    )
                )
            return out
        finally:
            await gt.close()

    async def _ingest_creates(self, events: list[CreateEvent]) -> int:
        """Add new curves to the watchlist with an initial state read."""
        fresh = [e for e in events if e.mint not in self.watch]
        if not fresh:
            return 0
        states = await self.provider.curve_states([e.mint for e in fresh])
        now = time.time()
        n = 0
        for e in fresh:
            state = states.get(e.mint)
            if state is None or state.complete:
                continue
            progress = round(state.get_migration_progress(), 2)
            if progress >= HANDOFF_PROGRESS_PCT:
                continue  # already grad_snipe's territory
            self.watch[e.mint] = {
                "first_seen": e.seen_at,
                "last_check": now,
                "last_progress": progress,
                "last_sol": round(state.real_sol_reserves / 1e9, 4),
                "last_inflow_at": now,
                "prev_check": None,
                "prev_sol": None,
                "inflow_sol": None,
                "prev_inflow_sol": None,
                "complete": False,
                "alerted": False,
                "name": e.name,
                "symbol": e.symbol,
                "creator": e.creator,
            }
            n += 1
        return n

    async def _sig_drainer(self) -> None:
        """Background: signatures -> fetched CreateEvents -> ingest."""
        while not self.stop.is_set():
            try:
                sig, slot = await asyncio.wait_for(self.sig_queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            try:
                event = await self.fetcher.fetch_create(sig, slot)
                if event is not None:
                    await self._ingest_creates([event])
            except Exception as e:  # noqa: BLE001 - never die on one tx
                logger.debug("create ingest failed: %s", e)

    async def _tick(self) -> dict:
        """One watch cycle: re-read curves, fire ignitions, prune."""
        now = time.time()
        summary: dict = {"watched": len(self.watch), "alerted": [], "pruned": 0}
        if not self.watch:
            return summary
        mints = list(self.watch.keys())
        try:
            states = await self.provider.curve_states(mints)
        except Exception as e:  # noqa: BLE001 - a failed read prunes nothing
            logger.warning("curve_states read failed: %s", e)
            return summary
        sol_usd = await fetch_sol_usd()
        for mint in mints:
            entry = self.watch.get(mint)
            if entry is None:
                continue
            state = states.get(mint)
            if state is None:
                # Account gone with no complete flag: treat as migrated.
                entry["complete"] = True
            else:
                prev_check, prev_sol = entry.get("last_check"), entry.get("last_sol")
                progress = round(state.get_migration_progress(), 2)
                last_sol = round(state.real_sol_reserves / 1e9, 4)
                inflow = None
                if prev_check is not None and prev_sol is not None:
                    gap = now - prev_check
                    if gap > 0 and gap <= 30 * 60:
                        inflow = round(max(0.0, last_sol - prev_sol), 4)
                entry["prev_inflow_sol"] = entry.get("inflow_sol")
                entry["prev_check"], entry["prev_sol"] = prev_check, prev_sol
                entry["last_check"] = now
                entry["last_progress"] = progress
                entry["last_sol"] = last_sol
                entry["inflow_sol"] = inflow
                entry["complete"] = bool(state.complete)
                if inflow and inflow > 0:
                    entry["last_inflow_at"] = now
                fire, _reasons = ignition_check(entry, now)
                if fire:
                    est_mcap = None
                    mcap_sol = implied_mcap_sol(state)
                    if mcap_sol is not None and sol_usd:
                        est_mcap = mcap_sol * sol_usd
                    msg = format_ignition_alert(mint, entry, est_mcap, self.poll_seconds)
                    if self.dry_run:
                        print(msg + "\n" + "\u2500" * 40)
                        entry["alerted"] = True
                        summary["alerted"].append(mint)
                    elif send_telegram(msg):
                        entry["alerted"] = True
                        summary["alerted"].append(mint)
                        logger.info("ignition alert sent: %s", mint[:12])
                    else:
                        logger.warning("ignition telegram failed: %s", mint[:12])
            reason = should_prune(entry, now)
            if reason:
                del self.watch[mint]
                summary["pruned"] += 1
                logger.debug("pruned %s: %s", mint[:12], reason)
        if not self.dry_run:
            _save_json({"mints": self.watch}, self.state_path)
        return summary

    async def run(self, run_once: bool = False) -> None:
        tasks = []
        use_ws = self.source in ("auto", "websocket")
        if use_ws:
            tasks.append(
                asyncio.create_task(
                    stream_create_signatures(
                        self.rpc_url,
                        self.sig_queue,
                        self.stop,
                        on_activity=self.ws_ok.set,
                    )
                )
            )
            tasks.append(asyncio.create_task(self._sig_drainer()))
        poll_fallback_due = 0.0
        try:
            while not self.stop.is_set():
                # Poll fallback when the stream never delivered (proxy, etc.).
                if self.source in ("auto", "poll"):
                    now = time.time()
                    stream_dead = not self.ws_ok.is_set()
                    if (self.source == "poll" or stream_dead) and now >= poll_fallback_due:
                        try:
                            events = await self._poll_discovery()
                            added = await self._ingest_creates(events)
                            if added:
                                logger.info("poll discovery: +%d curves", added)
                        except Exception as e:  # noqa: BLE001
                            logger.debug("poll discovery failed: %s", e)
                        poll_fallback_due = now + 120.0
                summary = await self._tick()
                if summary["alerted"] or summary["pruned"]:
                    logger.info(
                        "tick: watched=%d alerted=%d pruned=%d",
                        summary["watched"],
                        len(summary["alerted"]),
                        summary["pruned"],
                    )
                if run_once:
                    break
                await asyncio.sleep(self.poll_seconds)
        finally:
            self.stop.set()
            for t in tasks:
                t.cancel()
            await self.provider.close()
            await self.fetcher.close()
            if not self.dry_run:
                _save_json({"mints": self.watch}, self.state_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="pump.fun curve ignition watcher")
    ap.add_argument("--state-dir", default=os.path.expanduser("~/.fenrir-scout"))
    ap.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    ap.add_argument(
        "--dry-run", action="store_true", help="print alerts, don't send or persist alerted flags"
    )
    ap.add_argument(
        "--source",
        choices=("auto", "websocket", "poll"),
        default="auto",
        help="auto: websocket with poll fallback (default)",
    )
    ap.add_argument(
        "--run-once", action="store_true", help="single cycle then exit (for cron-style testing)"
    )
    args = ap.parse_args()
    watcher = CurveIgnitionWatcher(
        state_dir=args.state_dir,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        source=args.source,
    )
    asyncio.run(watcher.run(run_once=args.run_once))


if __name__ == "__main__":
    main()
