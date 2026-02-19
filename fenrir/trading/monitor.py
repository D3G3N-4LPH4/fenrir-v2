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

import websockets
from solders.pubkey import Pubkey

from fenrir.config import BotConfig
from fenrir.core.client import SolanaClient
from fenrir.logger import FenrirLogger
from fenrir.protocol.pumpfun import PumpFunProgram, TokenLaunchDetector


class PumpFunMonitor:
    """
    The eyes of FENRIR.
    Watches the blockchain for fresh token launches on pump.fun.
    """

    MAX_WS_RECONNECT_ATTEMPTS = 5
    MAX_SEEN_TOKENS = 10_000  # Bound the set to prevent unbounded memory growth

    def __init__(self, config: BotConfig, solana_client: SolanaClient, logger: FenrirLogger):
        self.config = config
        self.client = solana_client
        self.logger = logger
        # Bounded seen-set: evicts oldest entries when capacity is exceeded
        self._seen_tokens_lru: OrderedDict = OrderedDict()
        self.running = False
        self.ws_connection = None
        self.ws_subscription_id = None

        # Real launch detection via engine modules
        self.launch_detector = TokenLaunchDetector()
        self.pumpfun_program = PumpFunProgram()

    def _is_seen(self, key: str) -> bool:
        """Check if a signature has already been processed."""
        return key in self._seen_tokens_lru

    def _mark_seen(self, key: str) -> None:
        """Mark a signature as seen, evicting oldest if at capacity."""
        self._seen_tokens_lru[key] = None
        if len(self._seen_tokens_lru) > self.MAX_SEEN_TOKENS:
            self._seen_tokens_lru.popitem(last=False)

    async def start_monitoring(self, on_launch: Callable):
        """
        Begin the hunt.
        on_launch: async callback function when new token detected
        """
        self.running = True
        self.logger.info("FENRIR awakens... Monitoring pump.fun launches")

        if self.config.websocket_enabled:
            await self._monitor_websocket(on_launch)
        else:
            await self._monitor_polling(on_launch)

    async def _monitor_websocket(self, on_launch: Callable):
        """
        Real-time monitoring via WebSocket logsSubscribe.
        Subscribes to pump.fun program logs and detects INITIALIZE instructions.
        Falls back to polling after repeated failures.
        """
        self.logger.info("WebSocket monitoring active")
        consecutive_failures = 0
        backoff = 1.0

        while self.running and consecutive_failures < self.MAX_WS_RECONNECT_ATTEMPTS:
            try:
                async with websockets.connect(
                    self.config.ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    self.ws_connection = ws
                    consecutive_failures = 0
                    backoff = 1.0

                    # Subscribe to pump.fun program logs
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

                    # Wait for subscription confirmation
                    confirm = json.loads(await ws.recv())
                    if "result" in confirm:
                        self.ws_subscription_id = confirm["result"]
                        self.logger.info(
                            f"Subscribed to pump.fun logs (sub_id: {self.ws_subscription_id})"
                        )

                    # Listen for log notifications
                    async for message in ws:
                        if not self.running:
                            break

                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            continue

                        # Process notification
                        if "method" not in data or data["method"] != "logsNotification":
                            continue

                        notification = data.get("params", {}).get("result", {})
                        value = notification.get("value", {})
                        signature = value.get("signature")
                        logs = value.get("logs", [])
                        err = value.get("err")

                        # Skip failed transactions
                        if err or not signature:
                            continue

                        # Skip if already seen
                        if self._is_seen(signature):
                            continue
                        self._mark_seen(signature)

                        # Quick check: does this look like a create instruction?
                        has_create_hint = any(
                            "Program log: Instruction: Create" in log or "InitializeMint" in log
                            for log in logs
                        )

                        if not has_create_hint:
                            continue

                        # Fetch full transaction for detailed parsing
                        tx = await self.client.get_transaction(signature)
                        if tx and self._is_token_launch(tx):
                            token_data = await self._extract_token_data(tx)
                            if token_data and self._meets_criteria(token_data):
                                await on_launch(token_data)

            except websockets.ConnectionClosed as e:
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

        # Exhausted retries - fall back to polling
        if self.running:
            self.logger.warning(
                f"WebSocket failed {self.MAX_WS_RECONNECT_ATTEMPTS} times - falling back to polling"
            )
            await self._monitor_polling(on_launch)

    async def _monitor_polling(self, on_launch: Callable):
        """
        Polling-based monitoring.
        Reliable fallback when WebSocket is unavailable.
        """
        self.logger.info(f"Polling every {self.config.poll_interval_seconds}s")

        last_signature = None

        while self.running:
            try:
                # Get recent transactions to pump.fun program
                signatures = await self.client.get_recent_signatures(
                    self.client.pumpfun_program, limit=20
                )

                for sig_info in signatures:
                    signature = sig_info.signature

                    # Skip if we've seen it
                    if signature == last_signature:
                        break

                    if self._is_seen(str(signature)):
                        continue

                    # Mark as seen
                    self._mark_seen(str(signature))

                    # Analyze the transaction
                    tx = await self.client.get_transaction(str(signature))
                    if tx and self._is_token_launch(tx):
                        token_data = await self._extract_token_data(tx)
                        if token_data and self._meets_criteria(token_data):
                            await on_launch(token_data)

                # Update last signature
                if signatures:
                    last_signature = signatures[0].signature

                # Wait before next poll
                await asyncio.sleep(self.config.poll_interval_seconds)

            except Exception as e:
                self.logger.error("Monitoring error", e)
                await asyncio.sleep(self.config.poll_interval_seconds)

    def _is_token_launch(self, tx) -> bool:
        """
        Determine if a transaction contains a pump.fun token creation.
        Parses instruction data to match the INITIALIZE discriminator.
        """
        try:
            # Navigate the parsed transaction structure
            meta = tx.transaction.meta
            message = tx.transaction.transaction.message

            if not meta or not message:
                return False

            # Check inner instructions for pump.fun create calls
            pumpfun_id = str(self.client.pumpfun_program)

            # Check top-level instructions
            for ix in message.instructions:
                program_id = str(message.account_keys[ix.program_id_index])
                if program_id == pumpfun_id:
                    ix_data = b64.b64decode(ix.data) if isinstance(ix.data, str) else bytes(ix.data)
                    if self.launch_detector.is_create_instruction(ix_data):
                        return True

            # Also check inner instructions (pump.fun may use CPI)
            if meta.inner_instructions:
                for inner in meta.inner_instructions:
                    for ix in inner.instructions:
                        if hasattr(ix, "program_id") and str(ix.program_id) == pumpfun_id:
                            ix_data = (
                                b64.b64decode(ix.data)
                                if isinstance(ix.data, str)
                                else bytes(ix.data)
                            )
                            if self.launch_detector.is_create_instruction(ix_data):
                                return True

            return False

        except Exception as e:
            self.logger.error("Error parsing transaction for launch detection", e)
            return False

    async def _extract_token_data(self, tx) -> dict | None:
        """
        Extract token details from a launch transaction.
        Uses TokenLaunchDetector to parse instruction data and accounts,
        then fetches bonding curve state for pricing info.
        """
        try:
            message = tx.transaction.transaction.message
            pumpfun_id = str(self.client.pumpfun_program)

            # Find the create instruction and extract data
            for ix in message.instructions:
                program_id = str(message.account_keys[ix.program_id_index])
                if program_id != pumpfun_id:
                    continue

                ix_data = b64.b64decode(ix.data) if isinstance(ix.data, str) else bytes(ix.data)
                if not self.launch_detector.is_create_instruction(ix_data):
                    continue

                # Resolve account keys for this instruction
                accounts = [str(message.account_keys[idx]) for idx in ix.accounts]

                # Parse create event
                launch_info = self.launch_detector.parse_create_event(ix_data, accounts)
                if not launch_info:
                    continue

                token_mint = launch_info["token_mint"]
                bonding_curve_addr = launch_info.get("bonding_curve")

                # Fetch bonding curve state for liquidity/price data
                initial_liquidity_sol = 0.0
                market_cap_sol = 0.0
                curve_state = None

                if bonding_curve_addr:
                    bc_pubkey = Pubkey.from_string(bonding_curve_addr)
                    account_data = await self.client.get_account_info(bc_pubkey)
                    if account_data:
                        curve_state = self.pumpfun_program.decode_bonding_curve(account_data)
                        if curve_state:
                            initial_liquidity_sol = curve_state.real_sol_reserves / 1e9
                            market_cap_sol = curve_state.get_market_cap_sol()

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

    def _meets_criteria(self, token_data: dict) -> bool:
        """
        Quality filter. Not all launches are created equal.
        """
        liq = token_data.get("initial_liquidity_sol", 0)
        mcap = token_data.get("market_cap_sol", float("inf"))

        if liq < self.config.min_initial_liquidity_sol:
            self.logger.info(
                f"Skipping {token_data.get('symbol', '???')}: " f"liquidity too low ({liq:.2f} SOL)"
            )
            return False

        if mcap > self.config.max_initial_market_cap_sol:
            self.logger.info(
                f"Skipping {token_data.get('symbol', '???')}: "
                f"market cap too high ({mcap:.2f} SOL)"
            )
            return False

        return True

    async def stop(self):
        """Cease the hunt and unsubscribe from WebSocket."""
        self.running = False

        # Unsubscribe from WebSocket if active
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
