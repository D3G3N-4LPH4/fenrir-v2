"""Event-driven pump.fun launch discovery.

The poll-based discovery path (GeckoTerminal new-pools -> curve-state read)
lags launches by 10-25 minutes, so ignition plays are already graduated by
the time we see them. This module inverts the pipeline: it listens to the
pump.fun program's own on-chain activity and surfaces new bonding curves at
block ~zero.

Primary transport is a Solana ``logsSubscribe`` websocket filtered with
``mentions: [PUMP_PROGRAM_ID]``. Every transaction touching the program
streams in within ~1-2s; notifications whose logs contain a create marker
are fetched via ``getTransaction`` and parsed with the existing
``TokenLaunchDetector.parse_create_event`` (handles both ``create`` and
``create_v2`` discriminators).

Fail-open throughout: a dead websocket or an unparsable transaction simply
yields no events. Callers are expected to fall back to poll-based discovery
(GeckoTerminal new pools) when the event stream is unavailable — e.g. in
the sandbox, where the proxy breaks websockets.

NOTE (verify live): the create log marker below assumes Anchor's standard
``Program log: Instruction: Create`` / ``CreateV2`` lines. If a live capture
shows different wording, update ``CREATE_LOG_MARKERS`` — the filter is
fail-safe (no match = no fetch, never spam).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import aiohttp
import base58

from fenrir.protocol.pumpfun import (
    PUMP_PROGRAM_ID,
    TokenLaunchDetector,
)

logger = logging.getLogger(__name__)

# Log substrings that mark a token-creation instruction inside a
# logsSubscribe notification. Matched case-insensitively.
CREATE_LOG_MARKERS = ("instruction: create",)

# Reconnect backoff for the websocket stream.
_WS_RECONNECT_BASE_S = 2.0
_WS_RECONNECT_MAX_S = 60.0
# If no notification arrives for this long the stream is presumed dead.
_WS_STALL_TIMEOUT_S = 180.0


@dataclass
class CreateEvent:
    """A newly launched pump.fun bonding curve, seen on-chain."""

    mint: str
    bonding_curve: str | None
    creator: str | None
    name: str
    symbol: str
    uri: str
    signature: str
    slot: int
    seen_at: float = field(default_factory=time.time)


def notification_has_create(logs: list[str]) -> bool:
    """True when a logsSubscribe notification's logs mark a create."""
    for line in logs:
        low = line.lower()
        if any(m in low for m in CREATE_LOG_MARKERS):
            return True
    return False


def extract_create_candidates(notification: dict) -> list[tuple[str, int, list[str]]]:
    """Pull (signature, slot, logs) create-candidates from a WS message.

    Pure function — unit tested. Returns [] for non-notification messages
    (subscription confirmations, etc.).
    """
    try:
        result = notification.get("params", {}).get("result", {})
        value = result.get("value", {})
        logs = value.get("logs") or []
        signature = value.get("signature") or ""
        slot = int(result.get("context", {}).get("slot") or 0)
    except (AttributeError, ValueError, TypeError):
        return []
    if not signature or not notification_has_create(logs):
        return []
    return [(signature, slot, logs)]


class CreateEventFetcher:
    """Fetches + parses create transactions found by the log filter."""

    def __init__(self, rpc_url: str, timeout_seconds: float = 15.0) -> None:
        self._rpc_url = rpc_url
        self._timeout = timeout_seconds
        self._session: aiohttp.ClientSession | None = None
        self._detector = TokenLaunchDetector()
        self._req_id = 0

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers={"User-Agent": "FENRIR/2.0 curve-events"},
            )
        return self._session

    async def _rpc(self, method: str, params: list) -> dict | None:
        self._req_id += 1
        payload = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params}
        try:
            s = await self._session_get()
            async with s.post(self._rpc_url, json=payload) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                result = data.get("result")
                return result if isinstance(result, dict) else None
        except Exception as e:  # noqa: BLE001 - discovery is fail-open
            logger.debug("create-event RPC %s failed: %s", method, e)
            return None

    async def fetch_create(self, signature: str, slot: int) -> CreateEvent | None:
        """getTransaction -> find create instruction -> CreateEvent (None on miss)."""
        tx = await self._rpc(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "json",
                    "commitment": "confirmed",
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        )
        if not tx:
            return None
        try:
            message = tx["transaction"]["message"]
            instructions = message.get("instructions") or []
            account_keys = message.get("accountKeys") or []
        except (KeyError, TypeError):
            return None
        for ix in instructions:
            try:
                program_idx = ix.get("programIdIndex")
                program_id = account_keys[program_idx] if program_idx is not None else ""
                if program_id != str(PUMP_PROGRAM_ID):
                    continue
                raw = base58.b58decode(ix.get("data") or "")
                if not self._detector.is_create_instruction(raw):
                    continue
                accts = [
                    account_keys[i]
                    for i in (ix.get("accounts") or [])
                    if isinstance(i, int) and 0 <= i < len(account_keys)
                ]
                parsed = self._detector.parse_create_event(raw, accts)
                if not parsed or not parsed.get("token_mint"):
                    continue
                return CreateEvent(
                    mint=str(parsed["token_mint"]),
                    bonding_curve=str(parsed["bonding_curve"])
                    if parsed.get("bonding_curve")
                    else None,
                    creator=str(parsed["creator"]) if parsed.get("creator") else None,
                    name=str(parsed.get("name") or ""),
                    symbol=str(parsed.get("symbol") or ""),
                    uri=str(parsed.get("uri") or ""),
                    signature=signature,
                    slot=slot,
                )
            except Exception as e:  # noqa: BLE001 - one bad ix must not kill the tx
                logger.debug("create ix parse failed for %s: %s", signature[:16], e)
                continue
        return None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None


def rpc_ws_url(http_url: str) -> str:
    """Convert an http(s) RPC URL to its websocket equivalent."""
    if http_url.startswith("https://"):
        return "wss://" + http_url[len("https://") :]
    if http_url.startswith("http://"):
        return "ws://" + http_url[len("http://") :]
    return http_url


async def stream_create_signatures(
    rpc_url: str,
    queue: asyncio.Queue,
    stop: asyncio.Event,
    on_activity: Callable[[], None] | None = None,
) -> None:
    """logsSubscribe loop: enqueue (signature, slot) for create-marked txs.

    Runs until ``stop`` is set. Reconnects with backoff on failure; a stall
    (no notification for _WS_STALL_TIMEOUT_S) also triggers reconnect, since
    pump.fun traffic makes true silence implausible. ``on_activity`` fires
    once on the first received notification so callers can tell a live
    stream from a dead one.
    """
    ws_url = rpc_ws_url(rpc_url)
    backoff = _WS_RECONNECT_BASE_S
    sub_id = 1
    while not stop.is_set():
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.ws_connect(
                    ws_url,
                    headers={"User-Agent": "FENRIR/2.0 curve-events"},
                ) as ws:
                    await ws.send_json(
                        {
                            "jsonrpc": "2.0",
                            "id": sub_id,
                            "method": "logsSubscribe",
                            "params": [
                                {"mentions": [str(PUMP_PROGRAM_ID)]},
                                {"commitment": "confirmed"},
                            ],
                        }
                    )
                    sub_id += 1
                    backoff = _WS_RECONNECT_BASE_S  # connected: reset backoff
                    logger.info("curve-events: logsSubscribe connected")
                    activity_fired = False
                    while not stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=_WS_STALL_TIMEOUT_S)
                        except TimeoutError:
                            logger.warning("curve-events: stream stalled, reconnecting")
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            if not activity_fired and on_activity is not None:
                                activity_fired = True
                                try:
                                    on_activity()
                                except Exception as e:  # noqa: BLE001
                                    logger.debug("on_activity callback failed: %s", e)
                            try:
                                data = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            for sig, slot, _logs in extract_create_candidates(data):
                                queue.put_nowait((sig, slot))
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            logger.warning("curve-events: ws closed, reconnecting")
                            break
        except Exception as e:  # noqa: BLE001 - reconnect, never die
            logger.warning("curve-events: ws error (%s), retry in %.0fs", e, backoff)
        if stop.is_set():
            break
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, _WS_RECONNECT_MAX_S)
