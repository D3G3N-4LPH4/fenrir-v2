#!/usr/bin/env python3
"""
FENRIR - Pump.fun Monitor

The eyes of FENRIR.
Watches the blockchain for fresh token launches on pump.fun.
"""

import asyncio
import base64 as b64
import json
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from typing import Any

import base58
from solders.pubkey import Pubkey
from websockets.exceptions import ConnectionClosed

# websockets is capped at <12 by solana 0.35.0, so use the legacy client's
# connect (the async context-manager API used below is unchanged on 11.x).
from websockets.legacy.client import connect as ws_connect

from fenrir.config import BotConfig
from fenrir.core.client import SolanaClient
from fenrir.logger import FenrirLogger
from fenrir.protocol.pumpfun import PumpFunProgram, TokenLaunchDetector
from fenrir.trading.migration import MigrationDetector

WSOL_MINT = "So11111111111111111111111111111111111111112"


class PumpFunMonitor:
    """
    The eyes of FENRIR.
    Watches the blockchain for fresh token launches on pump.fun.
    """

    MAX_WS_RECONNECT_ATTEMPTS = 5
    MAX_SEEN_TOKENS = 10_000

    def __init__(self, config: BotConfig, solana_client: SolanaClient, logger: FenrirLogger):
        self.config = config
        self.client = solana_client
        self.logger = logger
        self._seen_tokens_lru: OrderedDict[str, None] = OrderedDict()
        self.running = False
        self.ws_connection: Any = None
        self.ws_subscription_id: int | None = None
        self.launch_detector = TokenLaunchDetector()
        self.pumpfun_program = PumpFunProgram()
        self.migration_detector = MigrationDetector()

    def _is_seen(self, key: str) -> bool:
        return key in self._seen_tokens_lru

    def _mark_seen(self, key: str) -> None:
        self._seen_tokens_lru[key] = None
        if len(self._seen_tokens_lru) > self.MAX_SEEN_TOKENS:
            self._seen_tokens_lru.popitem(last=False)

    def _get_program_id(self, ix: Any, account_keys: Any) -> str:
        """
        Safely resolve program ID from any solders instruction type.
        - CompiledInstruction: has program_id_index, needs account_keys lookup
        - UiPartiallyDecodedInstruction / ParsedInstruction: exposes program_id directly
        """
        if hasattr(ix, "program_id_index"):
            try:
                return str(account_keys[ix.program_id_index])
            except (IndexError, TypeError):
                return ""
        if hasattr(ix, "program_id"):
            return str(ix.program_id)
        return ""

    def _get_ix_data(self, ix: Any) -> bytes:
        """
        Safely extract raw instruction bytes from any solders instruction type.
        ParsedInstruction may not have a data field at all.
        """
        data = getattr(ix, "data", None)
        if data is None:
            return b""
        if isinstance(data, bytes | bytearray):
            return bytes(data)
        if isinstance(data, str):
            # jsonParsed encoding returns instruction data as base58; older/base64
            # encodings fall back to b64. Try base58 first, then base64.
            try:
                return base58.b58decode(data)
            except Exception:
                try:
                    return b64.b64decode(data)
                except Exception:
                    return b""
        return b""

    async def start_monitoring(self, on_launch: Callable[..., Any]) -> None:
        """Begin the hunt. on_launch: async callback when new token detected."""
        self.running = True
        self.logger.info("FENRIR awakens... Monitoring pump.fun launches")

        if self.config.websocket_enabled:
            if self.config.migration_feed_enabled:
                # Run the launch feed and the (experimental) migration feed
                # concurrently — logsSubscribe allows only one address per
                # subscription, so the migration feed needs its own connection.
                self.logger.info("Migration feed enabled (experimental)")
                await asyncio.gather(
                    self._monitor_websocket(on_launch),
                    self._monitor_migration_websocket(on_launch),
                )
            else:
                await self._monitor_websocket(on_launch)
        else:
            await self._monitor_polling(on_launch)

    async def _monitor_websocket(self, on_launch: Callable[..., Any]) -> None:
        """
        Real-time monitoring via WebSocket logsSubscribe.
        Falls back to polling after repeated failures.
        """
        self.logger.info("WebSocket monitoring active")
        consecutive_failures = 0
        backoff = 1.0

        while self.running and consecutive_failures < self.MAX_WS_RECONNECT_ATTEMPTS:
            try:
                async with ws_connect(
                    self.config.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    self.ws_connection = ws
                    consecutive_failures = 0
                    backoff = 1.0

                    subscribe_msg = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "logsSubscribe",
                            "params": [
                                {"mentions": [str(self.client.pumpfun_program)]},
                                {"commitment": "confirmed"},
                            ],
                        }
                    )
                    await ws.send(subscribe_msg)

                    confirm = json.loads(await ws.recv())
                    if "result" in confirm:
                        self.ws_subscription_id = confirm["result"]
                        self.logger.info(
                            f"Subscribed to pump.fun logs (sub_id: {self.ws_subscription_id})"
                        )

                    async for message in ws:
                        if not self.running:
                            break

                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        if "method" not in data or data["method"] != "logsNotification":
                            continue

                        notification = data.get("params", {}).get("result", {})
                        value = notification.get("value", {})
                        signature: str = value.get("signature", "")
                        logs: list[str] = value.get("logs", [])
                        err = value.get("err")

                        if err or not signature:
                            continue

                        if self._is_seen(signature):
                            continue
                        self._mark_seen(signature)

                        # pump.fun's native token creation emits "Instruction: CreateV2".
                        # Matching that specific line (rather than a bare "Create")
                        # avoids fetching every CreateTokenAccount / CreateIdempotent
                        # from unrelated router transactions that merely touch pump.
                        has_create_hint = any("Instruction: CreateV2" in log for log in logs)

                        if not has_create_hint:
                            continue

                        tx = await self.client.get_transaction(signature)
                        if tx and self._is_token_launch(tx):
                            token_data = await self._extract_token_data(tx)
                            if token_data and self._meets_criteria(token_data):
                                await on_launch(token_data)

            except ConnectionClosed as e:
                consecutive_failures += 1
                self.logger.warning(
                    f"WebSocket disconnected: {e}. "
                    f"Reconnecting ({consecutive_failures}/{self.MAX_WS_RECONNECT_ATTEMPTS})..."
                )
                await asyncio.sleep(min(backoff, 30))
                backoff *= 2

            except Exception as e:
                consecutive_failures += 1
                self.logger.error("WebSocket error", e)
                await asyncio.sleep(min(backoff, 30))
                backoff *= 2

            finally:
                self.ws_connection = None
                self.ws_subscription_id = None

        if self.running:
            self.logger.warning(
                f"WebSocket failed {self.MAX_WS_RECONNECT_ATTEMPTS} times - falling back to polling"
            )
            await self._monitor_polling(on_launch)

    async def _monitor_migration_websocket(self, on_launch: Callable[..., Any]) -> None:
        """
        EXPERIMENTAL: second logsSubscribe watching the pump.fun migration
        authority for pump→Raydium graduations, feeding the migration_snipe
        strategy.

        Off unless config.migration_feed_enabled. The migration-tx parser
        (_extract_migration_token_data) relies on live account layouts and is
        not verified offline; it errs on the side of skipping (returns None)
        when the token mint is ambiguous rather than acting on a guess.
        """
        program = self.migration_detector.PUMP_MIGRATION_PROGRAM
        self.logger.info(f"Migration WebSocket active (program {program[:8]}...)")
        consecutive_failures = 0
        backoff = 1.0

        while self.running and consecutive_failures < self.MAX_WS_RECONNECT_ATTEMPTS:
            try:
                async with ws_connect(
                    self.config.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    consecutive_failures = 0
                    backoff = 1.0

                    subscribe_msg = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "logsSubscribe",
                            "params": [
                                {"mentions": [program]},
                                {"commitment": "confirmed"},
                            ],
                        }
                    )
                    await ws.send(subscribe_msg)

                    confirm = json.loads(await ws.recv())
                    if "result" in confirm:
                        self.logger.info(
                            f"Subscribed to migration logs (sub_id: {confirm['result']})"
                        )

                    async for message in ws:
                        if not self.running:
                            break

                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        if data.get("method") != "logsNotification":
                            continue

                        value = data.get("params", {}).get("result", {}).get("value", {})
                        signature = value.get("signature", "")
                        logs: list[str] = value.get("logs", [])
                        if value.get("err") or not signature:
                            continue

                        if not self.migration_detector.has_migration_hint(logs):
                            continue

                        if self._is_seen(signature):
                            continue
                        self._mark_seen(signature)

                        tx = await self.client.get_transaction(signature)
                        if not tx:
                            continue
                        token_data = self._extract_migration_token_data(tx)
                        if token_data:
                            # Migrated tokens bypass the pump-launch quality gate
                            # (_meets_criteria); the market filter + strategy
                            # evaluate_token do the gating.
                            await on_launch(token_data)

            except ConnectionClosed as e:
                consecutive_failures += 1
                self.logger.warning(
                    f"Migration WebSocket disconnected: {e}. "
                    f"Reconnecting ({consecutive_failures}/{self.MAX_WS_RECONNECT_ATTEMPTS})..."
                )
                await asyncio.sleep(min(backoff, 30))
                backoff *= 2

            except Exception as e:
                consecutive_failures += 1
                self.logger.error("Migration WebSocket error", e)
                await asyncio.sleep(min(backoff, 30))
                backoff *= 2

        if self.running:
            self.logger.warning("Migration feed stopped after repeated failures")

    def _extract_migration_token_data(self, tx: Any) -> dict[str, Any] | None:
        """
        EXPERIMENTAL best-effort: identify the migrated token mint from a
        migration transaction's token balances.

        Heuristic: a migration tx references the graduating token's mint in its
        pre/post token balances; the only non-WSOL mint is the candidate. If
        the mint is ambiguous (zero or multiple non-WSOL mints), return None so
        the bot skips rather than acting on the wrong token. Not verified
        offline — validate on devnet/mainnet before enabling live.
        """
        try:
            meta = getattr(tx.transaction, "meta", None)
            if meta is None:
                return None
            balances = list(getattr(meta, "post_token_balances", None) or []) + list(
                getattr(meta, "pre_token_balances", None) or []
            )
            mints = {
                str(b.mint)
                for b in balances
                if str(getattr(b, "mint", "")) and str(b.mint) != WSOL_MINT
            }
            if len(mints) != 1:
                return None
            return self.migration_detector.build_token_data(next(iter(mints)))
        except Exception as e:
            self.logger.debug(f"Migration token extraction failed: {e}")
            return None

    async def _monitor_polling(self, on_launch: Callable[..., Any]) -> None:
        """Polling-based monitoring. Reliable fallback when WebSocket is unavailable."""
        self.logger.info(f"Polling every {self.config.poll_interval_seconds}s")

        last_signature: Any = None

        while self.running:
            try:
                signatures: list[Any] = await self.client.get_recent_signatures(
                    self.client.pumpfun_program, limit=20
                )

                for sig_info in signatures:
                    signature = sig_info.signature

                    if signature == last_signature:
                        break

                    sig_str = str(signature)
                    if self._is_seen(sig_str):
                        continue
                    self._mark_seen(sig_str)

                    tx = await self.client.get_transaction(sig_str)
                    if tx and self._is_token_launch(tx):
                        token_data = await self._extract_token_data(tx)
                        if token_data and self._meets_criteria(token_data):
                            await on_launch(token_data)

                if signatures:
                    last_signature = signatures[0].signature

                await asyncio.sleep(self.config.poll_interval_seconds)

            except Exception as e:
                self.logger.error("Monitoring error", e)
                await asyncio.sleep(self.config.poll_interval_seconds)

    def _is_token_launch(self, tx: Any) -> bool:
        """
        Determine if a transaction contains a pump.fun token creation.
        Handles all three solders instruction types via _get_program_id().
        """
        try:
            meta = tx.transaction.meta
            message = tx.transaction.transaction.message

            if not meta or not message:
                return False

            pumpfun_id = str(self.client.pumpfun_program)

            # Check top-level instructions
            for ix in message.instructions:
                program_id = self._get_program_id(ix, message.account_keys)
                if program_id != pumpfun_id:
                    continue
                ix_data = self._get_ix_data(ix)
                if ix_data and self.launch_detector.is_create_instruction(ix_data):
                    return True

            # Also check inner instructions (pump.fun may use CPI)
            if meta.inner_instructions:
                for inner in meta.inner_instructions:
                    for ix in inner.instructions:
                        program_id = self._get_program_id(ix, message.account_keys)
                        if program_id != pumpfun_id:
                            continue
                        ix_data = self._get_ix_data(ix)
                        if ix_data and self.launch_detector.is_create_instruction(ix_data):
                            return True

            return False

        except Exception as e:
            self.logger.error("Error parsing transaction for launch detection", e)
            return False

    async def _extract_token_data(self, tx: Any) -> dict[str, Any] | None:
        """
        Extract token details from a launch transaction.
        Uses TokenLaunchDetector to parse instruction data and accounts,
        then fetches bonding curve state for pricing info.
        """
        try:
            message = tx.transaction.transaction.message
            pumpfun_id = str(self.client.pumpfun_program)

            for ix in message.instructions:
                program_id = self._get_program_id(ix, message.account_keys)
                if program_id != pumpfun_id:
                    continue

                ix_data = self._get_ix_data(ix)
                if not ix_data or not self.launch_detector.is_create_instruction(ix_data):
                    continue

                # Resolve account keys safely — CompiledInstruction has int indices,
                # UiPartiallyDecodedInstruction has already-resolved pubkey strings
                raw_accounts = getattr(ix, "accounts", [])
                if raw_accounts and isinstance(raw_accounts[0], int):
                    accounts: list[str] = [str(message.account_keys[idx]) for idx in raw_accounts]
                else:
                    accounts = [str(a) for a in raw_accounts]

                launch_info: dict[str, Any] | None = self.launch_detector.parse_create_event(
                    ix_data, accounts
                )
                if not launch_info:
                    continue

                token_mint: str = launch_info["token_mint"]
                bonding_curve_addr: str | None = launch_info.get("bonding_curve")

                initial_liquidity_sol = 0.0
                market_cap_sol = 0.0
                curve_state = None

                if bonding_curve_addr:
                    try:
                        bc_pubkey = Pubkey.from_string(bonding_curve_addr)
                        account_data = await self.client.get_account_info(bc_pubkey)
                        if account_data:
                            curve_state = self.pumpfun_program.decode_bonding_curve(account_data)
                            if curve_state:
                                initial_liquidity_sol = curve_state.real_sol_reserves / 1e9
                                market_cap_sol = curve_state.get_market_cap_sol()
                    except Exception as e:
                        self.logger.warning(
                            f"Could not fetch bonding curve for {bonding_curve_addr}: {e}"
                        )

                return {
                    "token_address": token_mint,
                    "bonding_curve": bonding_curve_addr,
                    "creator": launch_info.get("creator"),
                    "name": launch_info.get("name", "Unknown"),
                    "symbol": launch_info.get("symbol", "???"),
                    "uri": launch_info.get("uri"),
                    "initial_liquidity_sol": initial_liquidity_sol,
                    "market_cap_sol": market_cap_sol,
                    "bonding_curve_state": curve_state,
                    "launch_time": datetime.now(),
                }

            return None

        except Exception as e:
            self.logger.error("Error extracting token data", e)
            return None

    def _meets_criteria(self, token_data: dict[str, Any]) -> bool:
        """Quality filter. Not all launches are created equal."""
        liq: float = token_data.get("initial_liquidity_sol", 0)
        mcap: float = token_data.get("market_cap_sol", float("inf"))

        if liq < self.config.min_initial_liquidity_sol:
            self.logger.info(
                f"Skipping {token_data.get('symbol', '???')}: liquidity too low ({liq:.2f} SOL)"
            )
            return False

        if mcap > self.config.max_initial_market_cap_sol:
            self.logger.info(
                f"Skipping {token_data.get('symbol', '???')}: market cap too high ({mcap:.2f} SOL)"
            )
            return False

        return True

    async def stop(self) -> None:
        """Cease the hunt and unsubscribe from WebSocket."""
        self.running = False

        if self.ws_connection and self.ws_subscription_id is not None:
            try:
                unsub_msg = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "logsUnsubscribe",
                        "params": [self.ws_subscription_id],
                    }
                )
                await self.ws_connection.send(unsub_msg)
                await self.ws_connection.close()
            except Exception as e:
                self.logger.debug(f"WebSocket cleanup error: {e}")

        self.logger.info("Monitoring stopped")
