#!/usr/bin/env python3
"""
FENRIR - Trading Engine

The heart of FENRIR.
Executes trades directly against pump.fun's bonding curve program.
"""

import asyncio

from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.message import Message
from solders.pubkey import Pubkey
from solders.token.associated import get_associated_token_address
from solders.transaction import Transaction

from fenrir.config import BotConfig, TradingMode
from fenrir.core.client import SolanaClient
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.core.positions import PositionManager
from fenrir.core.wallet import WalletManager
from fenrir.logger import FenrirLogger
from fenrir.protocol.pumpfun import BondingCurveState, PumpFunProgram
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
        # Jito requires a constructed client either way. Per-strategy tip
        # override is a follow-up; the client's configured tip is used.
        if self.jito is None:
            return False
        if self.tx_config is not None:
            return self.tx_config.jito_enabled(strategy_id)
        return bool(self.config.use_jito)

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

            # 4. Get buyer's associated token account
            buyer_ata = get_associated_token_address(self.wallet.pubkey, token_mint)

            # 5. Get associated bonding curve token account
            assoc_bonding_curve = self.pumpfun.get_associated_bonding_curve_address(
                bonding_curve, token_mint
            )

            # 6. Build buy instruction
            buy_ix = self.pumpfun.build_buy_instruction(
                buyer=self.wallet.pubkey,
                token_mint=token_mint,
                bonding_curve=bonding_curve,
                associated_bonding_curve=assoc_bonding_curve,
                buyer_token_account=buyer_ata,
                amount_sol=amount_lamports,
                max_slippage_bps=slippage_bps,
            )

            # 7. Build compute budget instructions for priority
            compute_price_ix = set_compute_unit_price(priority_fee)
            compute_limit_ix = set_compute_unit_limit(DEFAULT_COMPUTE_UNITS)

            # 8. Get recent blockhash
            blockhash = await self.client.get_latest_blockhash()
            if not blockhash:
                self.logger.warning("Failed to get recent blockhash")
                return False

            # 9. Build and sign transaction
            instructions = [compute_limit_ix, compute_price_ix, buy_ix]
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
                    transaction, self.wallet.keypair, blockhash
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

            # 3. Get seller's token account
            seller_token_info = await self.client.get_token_accounts_by_owner(
                self.wallet.pubkey, token_mint
            )
            if not seller_token_info:
                self.logger.warning("No token account found - nothing to sell")
                return False

            seller_ata = seller_token_info["address"]
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

            # 5. Get associated bonding curve token account
            assoc_bonding_curve = self.pumpfun.get_associated_bonding_curve_address(
                bonding_curve, token_mint
            )

            # 6. Build sell instruction
            sell_ix = self.pumpfun.build_sell_instruction(
                seller=self.wallet.pubkey,
                token_mint=token_mint,
                bonding_curve=bonding_curve,
                associated_bonding_curve=assoc_bonding_curve,
                seller_token_account=seller_ata,
                amount_tokens=sell_amount,
                min_sol_output=min_sol_output,
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
                    transaction, self.wallet.keypair, blockhash
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

            self.logger.info(f"Jupiter sell fallback: {token_address[:8]}... | Reason: {reason}")
            self.positions.close_position(token_address, reason)
            return True

        except Exception as e:
            self.logger.error(f"Jupiter sell fallback failed for {token_address}", e)
            return False

    async def _confirm_transaction(
        self, signature: str, max_attempts: int = 10, interval: float = 2.0
    ) -> bool:
        """Poll for transaction confirmation with exponential backoff."""
        for attempt in range(max_attempts):
            statuses = await self.client.get_signature_statuses([signature])
            if statuses and statuses[0]:
                status = statuses[0]
                if hasattr(status, "err") and status.err:
                    self.logger.warning(f"Transaction failed on-chain: {status.err}")
                    return False
                if hasattr(status, "confirmation_status"):
                    cs = str(status.confirmation_status)
                    if cs in ("confirmed", "finalized"):
                        return True
            await asyncio.sleep(min(interval * (1.3**attempt), 10))
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
