#!/usr/bin/env python3
"""
FENRIR - Trading Engine

The heart of FENRIR.
Executes trades directly against pump.fun's bonding curve program.
"""

import asyncio
import base64

import base58
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.message import Message
from solders.pubkey import Pubkey
from solders.transaction import Transaction, VersionedTransaction

from fenrir.config import BotConfig, TradingMode
from fenrir.core.client import SolanaClient
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.core.positions import PositionManager
from fenrir.core.wallet import WalletManager
from fenrir.logger import FenrirLogger
from fenrir.protocol.pumpfun import (
    BUY_DISCRIMINATOR,
    PUMP_GLOBAL,
    PUMP_PROGRAM_ID,
    TOKEN_PROGRAM,
    BondingCurveState,
    PumpFunProgram,
)
from fenrir.trading.tx_config import TxConfigManager

LAMPORTS_PER_SOL = 1_000_000_000
DEFAULT_COMPUTE_UNITS = 200_000


class TradingEngine:
    """
    The heart of FENRIR.
    Executes trades directly against pump.fun's bonding curve program.
    """

    SOL_MINT = "So11111111111111111111111111111111111111112"

    def __init__(
        self,
        config: BotConfig,
        wallet: WalletManager,
        solana_client: SolanaClient,
        jupiter: JupiterSwapEngine,
        positions: PositionManager,
        logger: FenrirLogger,
        jito=None,
    ):
        self.config = config
        self.wallet = wallet
        self.client = solana_client
        self.jupiter = jupiter
        self.positions = positions
        self.logger = logger
        self.jito = jito

        # Per-strategy execution profiles (slippage/priority-fee/jito). Built
        # only when opted in; otherwise the flat BotConfig settings are used.
        self.tx_config: TxConfigManager | None = (
            TxConfigManager(rpc_url=config.rpc_url) if config.tx_profiles_enabled else None
        )

        # Direct pump.fun program interface
        self.pumpfun = PumpFunProgram()

        # Per-token v2 buyback fee accounts (idx16/17). They're a per-token
        # fee-sharing config (not a simple PDA), so we resolve them by shadowing
        # the token's own most recent successful buy and cache per mint — the
        # config is stable per token. Wrong accounts fail simulation (6024), so a
        # miss is safe, never a silent bad send.
        self._fee_extras_cache: dict[str, tuple[Pubkey, Pubkey]] = {}

    # ── Execution-parameter resolution ─────────────────────────────────
    # When tx profiles are enabled, resolve slippage / priority-fee / jito
    # per strategy; otherwise fall back to the flat BotConfig values so
    # default behavior is unchanged.

    async def _resolve_priority_fee(self, strategy_id: str) -> int:
        if self.tx_config is not None:
            return await self.tx_config.get_priority_fee_lamports(strategy_id)
        return self.config.priority_fee_lamports

    def _resolve_slippage_bps(self, strategy_id: str) -> int:
        if self.tx_config is not None:
            return self.tx_config.get_slippage_bps(strategy_id)
        return self.config.max_slippage_bps

    def _resolve_use_jito(self, strategy_id: str) -> bool:
        if self.jito is None:
            return False
        if self.tx_config is not None:
            return self.tx_config.jito_enabled(strategy_id)
        return bool(self.config.use_jito)

    def _resolve_jito_tip_lamports(self, strategy_id: str) -> int:
        if self.tx_config is not None:
            return self.tx_config.jito_tip_lamports(strategy_id)
        return self.config.jito_tip_lamports

    async def _live_fee_recipient(self) -> Pubkey | None:
        """Read the current pump fee_recipient from the global account.

        Returns None on any failure; the instruction builder then falls back to
        the module constant. The value rotates on-chain, so trusting a stale
        constant risks 6000 NotAuthorized.
        """
        try:
            global_data = await self.client.get_account_info(PUMP_GLOBAL)
            if global_data:
                return self.pumpfun.parse_global_fee_recipient(global_data)
        except Exception as e:
            self.logger.debug(f"fee_recipient fetch failed, using fallback: {e}")
        return None

    async def _resolve_fee_extras(self, token_mint: Pubkey) -> tuple[Pubkey, Pubkey] | None:
        """Resolve the token's two v2 buyback fee accounts (buy idx16/17).

        They are a per-token fee-sharing config, not a derivable PDA, so we copy
        them from the token's most recent successful on-chain buy (18 accounts)
        and cache per mint. Returns None if none can be found (the builder then
        falls back to the module constants). A wrong value fails simulation, so
        this never risks a bad send.
        """
        key = str(token_mint)
        cached = self._fee_extras_cache.get(key)
        if cached is not None:
            return cached
        try:
            sigs = await self.client.get_recent_signatures(token_mint, limit=25)
            for s in sigs:
                if getattr(s, "err", None) is not None:
                    continue
                tx = await self.client.get_transaction(str(s.signature))
                if not tx:
                    continue
                msg = tx.transaction.transaction.message
                keys = [str(k) for k in msg.account_keys]
                for ix in msg.instructions:
                    pid = (
                        str(ix.program_id)
                        if hasattr(ix, "program_id")
                        else keys[ix.program_id_index]
                    )
                    if pid != str(PUMP_PROGRAM_ID):
                        continue
                    data = getattr(ix, "data", None)
                    raw = base58.b58decode(data) if isinstance(data, str) else bytes(data or b"")
                    if raw[:8] != BUY_DISCRIMINATOR:
                        continue
                    accs = getattr(ix, "accounts", [])
                    resolved = (
                        [keys[i] for i in accs]
                        if (accs and isinstance(accs[0], int))
                        else [str(a) for a in accs]
                    )
                    if len(resolved) >= 18:
                        extras = (
                            Pubkey.from_string(resolved[16]),
                            Pubkey.from_string(resolved[17]),
                        )
                        self._fee_extras_cache[key] = extras
                        return extras
        except Exception as e:
            self.logger.debug(f"fee-extras resolve failed for {key[:8]}...: {e}")
        return None

    async def close(self) -> None:
        """Release resources held by the engine (tx-profile RPC session)."""
        if self.tx_config is not None:
            await self.tx_config.close()

    async def fetch_curve_state(self, token_address: str) -> BondingCurveState | None:
        """
        Fetch and decode a token's pump.fun bonding curve from chain.

        Shared price source for both entry and mark-to-market so PnL stays
        scale-consistent (PnL depends only on the current/entry price ratio).
        Returns None if the curve can't be fetched/decoded.
        """
        try:
            token_mint = Pubkey.from_string(token_address)
            bonding_curve, _ = self.pumpfun.derive_bonding_curve_address(token_mint)
            account_data = await self.client.get_account_info(bonding_curve)
            if not account_data:
                return None
            return self.pumpfun.decode_bonding_curve(account_data)
        except Exception as e:
            self.logger.debug(f"Bonding curve fetch failed for {token_address[:8]}...: {e}")
            return None

    async def execute_buy(
        self,
        token_data: dict,
        amount_sol: float | None = None,
        strategy_id: str = "default",
    ) -> bool:
        """
        Snipe a new token launch via direct pump.fun bonding curve buy.

        Args:
            token_data: Token metadata from monitor
            amount_sol: SOL to spend. Defaults to config.buy_amount_sol.
                        Passed explicitly to avoid mutating shared config
                        during concurrent calls.
            strategy_id: Which strategy opened this position (for position tracking).
        """
        token_address = token_data["token_address"]
        amount_sol = amount_sol if amount_sol is not None else self.config.buy_amount_sol

        self.logger.info(f"Executing buy: {token_address[:8]}... for {amount_sol} SOL")

        if self.config.mode == TradingMode.SIMULATION:
            self.logger.info("SIMULATION MODE - No real transaction sent")
            # Price the position from the pump.fun bonding curve so that entry
            # and later mark-to-market updates share the SAME source and scale.
            # PnL then depends only on the price ratio, which is what we want.
            # If the curve is unavailable, refuse rather than fabricate a bogus
            # position (the old fallback minted 100k tokens at a made-up price,
            # which produced nonsensical multi-million-SOL PnL against the feed).
            curve_state = token_data.get("bonding_curve_state")
            if not (curve_state and isinstance(curve_state, BondingCurveState)):
                curve_state = await self.fetch_curve_state(token_address)
            if not curve_state:
                self.logger.warning(
                    f"Cannot price {token_address[:8]}... (no bonding curve) - skipping sim buy"
                )
                return False

            entry_price = curve_state.get_price()
            if entry_price <= 0:
                self.logger.warning(
                    f"Non-positive curve price for {token_address[:8]}... - skipping sim buy"
                )
                return False

            # tokens_out * entry_price == amount_sol (invariant preserved)
            tokens_out = amount_sol / entry_price

            self.positions.open_position(
                token_address=token_address,
                entry_price=entry_price,
                amount_tokens=tokens_out,
                amount_sol=amount_sol,
                strategy_id=strategy_id,
                token_symbol=token_data.get("symbol", "???"),
            )
            return True

        # -- Live execution via direct pump.fun program --
        try:
            token_mint = Pubkey.from_string(token_address)
            amount_lamports = int(amount_sol * LAMPORTS_PER_SOL)

            # Resolve per-strategy execution params (tx profile or flat config)
            priority_fee = await self._resolve_priority_fee(strategy_id)
            slippage_bps = self._resolve_slippage_bps(strategy_id)
            use_jito = self._resolve_use_jito(strategy_id)
            jito_tip = self._resolve_jito_tip_lamports(strategy_id)

            # 0. Pre-check wallet balance
            balance = await self.client.get_balance(self.wallet.pubkey)
            total_cost = amount_sol + (priority_fee / LAMPORTS_PER_SOL)
            if balance < total_cost:
                self.logger.warning(
                    f"Insufficient SOL balance: {balance:.4f} < {total_cost:.4f} required"
                )
                return False

            # 1. Derive bonding curve PDA
            bonding_curve, _ = self.pumpfun.derive_bonding_curve_address(token_mint)

            # 2. Fetch and decode bonding curve state
            account_data = await self.client.get_account_info(bonding_curve)
            if not account_data:
                self.logger.warning("Failed to fetch bonding curve account")
                return False

            curve_state = self.pumpfun.decode_bonding_curve(account_data)
            if not curve_state:
                self.logger.warning("Failed to decode bonding curve state")
                return False

            if curve_state.complete:
                self.logger.warning("Token already migrated to Raydium - skipping")
                return False

            # 3. Calculate expected tokens and price impact
            tokens_out, price_impact = curve_state.calculate_buy_price(amount_sol)
            max_impact = slippage_bps / 100
            if price_impact > max_impact:
                self.logger.warning(
                    f"Price impact too high: {price_impact:.2f}% > {max_impact:.2f}%"
                )
                return False

            if tokens_out <= 0:
                self.logger.warning("Zero tokens output - skipping buy")
                return False

            # 4. Detect the mint's token program (classic SPL or Token-2022) and
            #    the coin creator (needed for the creator_vault account).
            token_program = await self.client.get_account_owner(token_mint) or TOKEN_PROGRAM
            if not curve_state.creator:
                self.logger.warning("No creator on bonding curve - cannot build buy")
                return False
            creator = Pubkey.from_string(curve_state.creator)

            # 5. Read the live fee_recipient (it rotates; a stale one → 6000) and
            #    the token's per-token v2 buyback fee accounts (a stale/wrong one
            #    → 6024 Overflow). Both fail simulation if wrong, never spend SOL.
            fee_recipient = await self._live_fee_recipient()
            extras = await self._resolve_fee_extras(token_mint)
            buyback_recipient = extras[0] if extras else None
            fee_pool_recipient = extras[1] if extras else None

            # 6. Idempotently create the buyer's ATA (with the correct token
            #    program), then build the buy — pump.fun requires the ATA to
            #    exist, else the buy fails with AccountNotInitialized (3012).
            #    pump `buy` takes the EXACT token amount to receive + a SOL cap:
            #    request slightly fewer tokens than quoted so slippage doesn't
            #    trip TooMuchSolRequired, and cap SOL at amount * (1 + slippage).
            slippage = slippage_bps / 10_000
            buy_amount_tokens = max(1, int(tokens_out * (1 - slippage)))
            max_sol_cost = int(amount_lamports * (1 + slippage))
            create_ata_ix = self.pumpfun.build_create_ata_instruction(
                self.wallet.pubkey, self.wallet.pubkey, token_mint, token_program
            )
            buy_ix = self.pumpfun.build_buy_instruction(
                buyer=self.wallet.pubkey,
                token_mint=token_mint,
                bonding_curve=bonding_curve,
                creator=creator,
                token_program=token_program,
                amount_tokens=buy_amount_tokens,
                max_sol_cost=max_sol_cost,
                fee_recipient=fee_recipient,
                buyback_fee_recipient=buyback_recipient,
                fee_pool_recipient=fee_pool_recipient,
            )

            # 7. Build compute budget instructions for priority
            compute_price_ix = set_compute_unit_price(priority_fee)
            compute_limit_ix = set_compute_unit_limit(DEFAULT_COMPUTE_UNITS)

            # 8. Get recent blockhash
            blockhash = await self.client.get_latest_blockhash()
            if not blockhash:
                self.logger.warning("Failed to get recent blockhash")
                return False

            # 9. Build and sign transaction (idempotent ATA-create before the buy)
            instructions = [compute_limit_ix, compute_price_ix, create_ata_ix, buy_ix]
            message = Message.new_with_blockhash(
                instructions,
                self.wallet.pubkey,
                blockhash,
            )
            transaction = Transaction.new_unsigned(message)
            transaction.sign([self.wallet.keypair], blockhash)

            # 10. Simulate first
            sim_ok = await self.client.simulate_transaction(transaction)
            if not sim_ok:
                self.logger.warning("Transaction simulation failed - aborting buy")
                return False

            # 11. Send transaction
            signature = None
            if use_jito and self.jito:
                self.logger.info("Sending via Jito bundle for MEV protection")
                result = await self.jito.send_transaction_with_tip(
                    transaction, self.wallet.keypair, blockhash, tip_lamports=jito_tip
                )
                if result.landed:
                    signature = result.bundle_id
                else:
                    self.logger.warning(f"Jito bundle failed: {result.error}")
                    return False
            else:
                signature = await self.client.send_transaction(transaction, skip_preflight=True)

            if not signature:
                self.logger.warning("Transaction send returned no signature")
                return False

            # 12. Confirm transaction landed on-chain
            confirmed = await self._confirm_transaction(signature)
            if not confirmed:
                self.logger.warning(f"Buy transaction not confirmed: {signature}")
                return False

            # 13. Calculate entry price and open position.
            # Price from the bonding curve (get_price) so entry and later
            # mark-to-market share one scale — PnL then tracks the price ratio.
            # (The old (amount_sol * LAMPORTS_PER_SOL)/tokens_out gave lamports
            # per token, which didn't match the SOL-scale mark price.)
            entry_price = curve_state.get_price()
            if entry_price <= 0:
                entry_price = (amount_sol / tokens_out) if tokens_out > 0 else 0.0
            self.logger.buy_executed(token_address, amount_sol, entry_price)

            # Track a curve-consistent notional so amount_tokens * entry_price
            # == amount_sol. Real sells use the on-chain balance, not this value.
            amount_tokens = (amount_sol / entry_price) if entry_price > 0 else tokens_out
            self.positions.open_position(
                token_address=token_address,
                entry_price=entry_price,
                amount_tokens=amount_tokens,
                amount_sol=amount_sol,
                strategy_id=strategy_id,
                token_symbol=token_data.get("symbol", "???"),
            )

            self.logger.info(f"TX Signature: {signature}")
            return True

        except Exception as e:
            self.logger.error(f"Buy execution failed for {token_address}", e)
            return False

    async def execute_sell(self, token_address: str, reason: str) -> bool:
        """
        Exit a position via direct pump.fun bonding curve sell.
        """
        position = self.positions.positions.get(token_address)
        if not position:
            self.logger.warning(f"No position found for {token_address}")
            return False

        self.logger.info(f"Executing sell: {token_address[:8]}... | Reason: {reason}")

        if self.config.mode == TradingMode.SIMULATION:
            self.logger.info("SIMULATION MODE - No real transaction sent")
            # Calculate simulated exit price from current position data
            pnl_pct = position.get_pnl_percent()
            pnl_sol = position.get_pnl_sol()
            self.logger.info(f"   Sim exit: PnL {pnl_pct:+.2f}% ({pnl_sol:+.4f} SOL)")
            self.positions.close_position(token_address, reason)
            return True

        # -- Live execution via direct pump.fun program --
        try:
            token_mint = Pubkey.from_string(token_address)

            # Resolve per-strategy execution params from the owning position.
            strategy_id = getattr(position, "strategy_id", "default")
            priority_fee = await self._resolve_priority_fee(strategy_id)
            slippage_bps = self._resolve_slippage_bps(strategy_id)
            use_jito = self._resolve_use_jito(strategy_id)
            jito_tip = self._resolve_jito_tip_lamports(strategy_id)

            # 1. Derive bonding curve PDA
            bonding_curve, _ = self.pumpfun.derive_bonding_curve_address(token_mint)

            # 2. Fetch bonding curve state
            account_data = await self.client.get_account_info(bonding_curve)
            if not account_data:
                self.logger.warning("Failed to fetch bonding curve for sell")
                return False

            curve_state = self.pumpfun.decode_bonding_curve(account_data)
            if not curve_state:
                self.logger.warning("Failed to decode bonding curve for sell")
                return False

            if curve_state.complete:
                self.logger.info("Token migrated to Raydium - attempting Jupiter DEX fallback")
                return await self._execute_sell_via_jupiter(
                    token_address, token_mint, position, reason
                )

            # 3. Get seller's token account. Retry briefly: right after a buy the
            #    RPC may not have indexed the freshly-created ATA yet, and a single
            #    empty read would orphan the position ("nothing to sell").
            seller_token_info = None
            for attempt in range(4):
                seller_token_info = await self.client.get_token_accounts_by_owner(
                    self.wallet.pubkey, token_mint
                )
                if seller_token_info and seller_token_info["amount"] > 0:
                    break
                if attempt < 3:
                    await asyncio.sleep(2.0)
            if not seller_token_info or seller_token_info["amount"] <= 0:
                self.logger.warning("No token balance found after retries - nothing to sell")
                return False

            actual_balance = seller_token_info["amount"]

            # Sell the on-chain balance (authoritative, avoids truncation of fractional tokens)
            sell_amount = actual_balance
            if sell_amount <= 0:
                self.logger.warning("Zero token balance - cannot sell")
                return False

            # 4. Calculate expected SOL output
            sol_out_lamports, price_impact = curve_state.calculate_sell_price(sell_amount)

            # Apply slippage for minimum output protection
            min_sol_output = int(sol_out_lamports * (1 - slippage_bps / 10000))

            # 5. Detect token program + creator, then build the sell instruction
            #    (associated accounts + creator_vault are derived inside).
            token_program = await self.client.get_account_owner(token_mint) or TOKEN_PROGRAM
            if not curve_state.creator:
                self.logger.warning("No creator on bonding curve - cannot build sell")
                return False
            creator = Pubkey.from_string(curve_state.creator)
            fee_recipient = await self._live_fee_recipient()
            extras = await self._resolve_fee_extras(token_mint)
            sell_ix = self.pumpfun.build_sell_instruction(
                seller=self.wallet.pubkey,
                token_mint=token_mint,
                bonding_curve=bonding_curve,
                creator=creator,
                token_program=token_program,
                amount_tokens=sell_amount,
                min_sol_output=min_sol_output,
                fee_recipient=fee_recipient,
                buyback_fee_recipient=extras[0] if extras else None,
                fee_pool_recipient=extras[1] if extras else None,
            )

            # 7. Build compute budget instructions
            compute_price_ix = set_compute_unit_price(priority_fee)
            compute_limit_ix = set_compute_unit_limit(DEFAULT_COMPUTE_UNITS)

            # 8. Get recent blockhash
            blockhash = await self.client.get_latest_blockhash()
            if not blockhash:
                self.logger.warning("Failed to get recent blockhash")
                return False

            # 9. Build and sign transaction
            instructions = [compute_limit_ix, compute_price_ix, sell_ix]
            message = Message.new_with_blockhash(
                instructions,
                self.wallet.pubkey,
                blockhash,
            )
            transaction = Transaction.new_unsigned(message)
            transaction.sign([self.wallet.keypair], blockhash)

            # 10. Simulate first
            sim_ok = await self.client.simulate_transaction(transaction)
            if not sim_ok:
                self.logger.warning("Sell simulation failed - aborting")
                return False

            # 11. Send transaction
            signature = None
            if use_jito and self.jito:
                self.logger.info("Sending sell via Jito bundle")
                result = await self.jito.send_transaction_with_tip(
                    transaction, self.wallet.keypair, blockhash, tip_lamports=jito_tip
                )
                if result.landed:
                    signature = result.bundle_id
                else:
                    self.logger.warning(f"Jito sell bundle failed: {result.error}")
                    return False
            else:
                signature = await self.client.send_transaction(transaction, skip_preflight=True)

            if not signature:
                self.logger.warning("Sell transaction returned no signature")
                return False

            # 12. Confirm transaction landed on-chain
            confirmed = await self._confirm_transaction(signature)
            if not confirmed:
                self.logger.warning(f"Sell transaction not confirmed: {signature}")
                return False

            # 13. Close position
            self.positions.close_position(token_address, reason)
            self.logger.info(f"Sell TX: {signature}")
            return True

        except Exception as e:
            self.logger.error(f"Sell execution failed for {token_address}", e)
            return False

    async def _execute_sell_via_jupiter(
        self,
        token_address: str,
        token_mint: Pubkey,
        position,
        reason: str,
    ) -> bool:
        """Fallback: sell via Jupiter DEX after token has migrated to Raydium."""
        try:
            # Get token balance
            seller_token_info = await self.client.get_token_accounts_by_owner(
                self.wallet.pubkey, token_mint
            )
            if not seller_token_info or seller_token_info["amount"] <= 0:
                self.logger.warning("No token balance for Jupiter sell fallback")
                return False

            token_amount = seller_token_info["amount"]

            # Get Jupiter quote: token -> SOL
            slippage_bps = self._resolve_slippage_bps(getattr(position, "strategy_id", "default"))
            quote = await self.jupiter.get_quote(
                input_mint=token_address,
                output_mint=self.SOL_MINT,
                amount=token_amount,
                slippage_bps=slippage_bps,
            )
            if not quote:
                self.logger.warning("Jupiter quote failed for migrated token sell")
                return False

            # Get swap transaction
            swap_tx_b64 = await self.jupiter.get_swap_transaction(quote, str(self.wallet.pubkey))
            if not swap_tx_b64:
                self.logger.warning("Jupiter swap transaction build failed")
                return False

            # Sign and send it. Jupiter returns a base64 VersionedTransaction with
            # the fee-payer/blockhash already set; we re-sign the message with our
            # keypair and broadcast. (Previously this was built and then dropped,
            # so the position was marked closed while the tokens were never sold.)
            signature = await self._sign_send_jupiter_swap(swap_tx_b64)
            if not signature:
                self.logger.warning("Jupiter swap send returned no signature")
                return False

            confirmed = await self._confirm_transaction(signature)
            if not confirmed:
                self.logger.warning(f"Jupiter sell not confirmed: {signature}")
                return False

            self.logger.info(
                f"Jupiter sell fallback: {token_address[:8]}... | {reason} | TX: {signature}"
            )
            self.positions.close_position(token_address, reason)
            return True

        except Exception as e:
            self.logger.error(f"Jupiter sell fallback failed for {token_address}", e)
            return False

    async def _sign_send_jupiter_swap(self, swap_tx_b64: str) -> str | None:
        """Sign a Jupiter base64 VersionedTransaction with our keypair and send it."""
        unsigned = VersionedTransaction.from_bytes(base64.b64decode(swap_tx_b64))
        signed = VersionedTransaction(unsigned.message, [self.wallet.keypair])
        return await self.client.send_raw_transaction(bytes(signed))

    async def _confirm_transaction(
        self, signature: str, max_attempts: int = 20, interval: float = 2.0
    ) -> bool:
        """Poll for transaction confirmation with exponential backoff.

        confirmation_status is a solders enum whose str() is e.g.
        "TransactionConfirmationStatus.Confirmed" — match by substring (lowered)
        rather than exact equality, else a landed tx is reported as failed and we
        orphan the position. Falls back to get_transaction (a landed tx with no
        meta error) since some RPCs drop the status once the sig is finalized.
        """
        for attempt in range(max_attempts):
            statuses = await self.client.get_signature_statuses([signature])
            if statuses and statuses[0]:
                status = statuses[0]
                if getattr(status, "err", None):
                    self.logger.warning(f"Transaction failed on-chain: {status.err}")
                    return False
                cs = str(getattr(status, "confirmation_status", "")).lower()
                if "confirmed" in cs or "finalized" in cs:
                    return True
            await asyncio.sleep(min(interval * (1.3**attempt), 10))

        # Final fallback: the sig may have finalized past the status window.
        tx = await self.client.get_transaction(signature)
        if tx is not None:
            err = getattr(getattr(tx, "transaction", None), "meta", None)
            if err is None or getattr(err, "err", None) is None:
                return True
            self.logger.warning(f"Transaction failed on-chain: {err.err}")
            return False

        self.logger.warning(
            f"Transaction {signature[:16]}... not confirmed after {max_attempts} attempts"
        )
        return False

    async def manage_positions(self):
        """
        Continuous position management.
        The autopilot that never sleeps.
        """
        exits = self.positions.check_exit_conditions()

        for token_address, reason in exits:
            await self.execute_sell(token_address, reason)
