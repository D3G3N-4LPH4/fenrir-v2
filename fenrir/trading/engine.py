#!/usr/bin/env python3
"""
FENRIR - Trading Engine

The heart of FENRIR.
Executes trades directly against pump.fun's bonding curve program.
"""

import asyncio
from typing import Optional, Dict

from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solders.message import Message
from solders.compute_budget import set_compute_unit_price, set_compute_unit_limit
from solders.token.associated import get_associated_token_address

from fenrir.config import BotConfig, TradingMode
from fenrir.logger import FenrirLogger
from fenrir.core.wallet import WalletManager
from fenrir.core.client import SolanaClient
from fenrir.core.jupiter import JupiterSwapEngine
from fenrir.core.positions import PositionManager
from fenrir.protocol.pumpfun import PumpFunProgram, BondingCurveState
from fenrir.exceptions import (
    ExecutionError, BondingCurveMigratedError, SlippageExceededError,
)


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

        # Direct pump.fun program interface
        self.pumpfun = PumpFunProgram()

    async def execute_buy(
        self, token_data: Dict, amount_sol: Optional[float] = None
    ) -> bool:
        """
        Snipe a new token launch via direct pump.fun bonding curve buy.

        Args:
            token_data: Token metadata from monitor
            amount_sol: SOL to spend. Defaults to config.buy_amount_sol.
                        Passed explicitly to avoid mutating shared config
                        during concurrent calls.
        """
        token_address = token_data["token_address"]
        amount_sol = amount_sol if amount_sol is not None else self.config.buy_amount_sol

        self.logger.info(f"Executing buy: {token_address[:8]}... for {amount_sol} SOL")

        if self.config.mode == TradingMode.SIMULATION:
            self.logger.info("SIMULATION MODE - No real transaction sent")
            # Use bonding curve state if available for realistic sim pricing
            curve_state = token_data.get("bonding_curve_state")
            if curve_state and isinstance(curve_state, BondingCurveState):
                tokens_out, _ = curve_state.calculate_buy_price(amount_sol)
                sim_price = (amount_sol * LAMPORTS_PER_SOL) / tokens_out if tokens_out > 0 else 0.000001
            else:
                tokens_out = int(amount_sol / 0.000001)
                sim_price = 0.000001

            self.positions.open_position(
                token_address=token_address,
                entry_price=sim_price,
                amount_tokens=tokens_out,
                amount_sol=amount_sol
            )
            return True

        # -- Live execution via direct pump.fun program --
        try:
            token_mint = Pubkey.from_string(token_address)
            amount_lamports = int(amount_sol * LAMPORTS_PER_SOL)

            # 0. Pre-check wallet balance
            balance = await self.client.get_balance(self.wallet.pubkey)
            total_cost = amount_sol + (self.config.priority_fee_lamports / LAMPORTS_PER_SOL)
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
            max_impact = self.config.max_slippage_bps / 100
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
                max_slippage_bps=self.config.max_slippage_bps,
            )

            # 7. Build compute budget instructions for priority
            compute_price_ix = set_compute_unit_price(
                self.config.priority_fee_lamports
            )
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
            transaction.sign([self.wallet.keypair])

            # 10. Simulate first
            sim_ok = await self.client.simulate_transaction(transaction)
            if not sim_ok:
                self.logger.warning("Transaction simulation failed - aborting buy")
                return False

            # 11. Send transaction
            signature = None
            if self.config.use_jito and self.jito:
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

            # 13. Calculate entry price and open position
            entry_price = (amount_sol * LAMPORTS_PER_SOL) / tokens_out if tokens_out > 0 else 0
            self.logger.buy_executed(token_address, amount_sol, entry_price)

            self.positions.open_position(
                token_address=token_address,
                entry_price=entry_price,
                amount_tokens=tokens_out,
                amount_sol=amount_sol
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
            exit_price = position.current_price or position.entry_price
            pnl_pct = position.get_pnl_percent()
            pnl_sol = position.get_pnl_sol()
            self.logger.info(
                f"   Sim exit: PnL {pnl_pct:+.2f}% ({pnl_sol:+.4f} SOL)"
            )
            self.positions.close_position(token_address, reason)
            return True

        # -- Live execution via direct pump.fun program --
        try:
            token_mint = Pubkey.from_string(token_address)

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
                self.logger.info(
                    "Token migrated to Raydium - attempting Jupiter DEX fallback"
                )
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
            min_sol_output = int(
                sol_out_lamports * (1 - self.config.max_slippage_bps / 10000)
            )

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
            compute_price_ix = set_compute_unit_price(
                self.config.priority_fee_lamports
            )
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
            transaction.sign([self.wallet.keypair])

            # 10. Simulate first
            sim_ok = await self.client.simulate_transaction(transaction)
            if not sim_ok:
                self.logger.warning("Sell simulation failed - aborting")
                return False

            # 11. Send transaction
            signature = None
            if self.config.use_jito and self.jito:
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
            quote = await self.jupiter.get_quote(
                input_mint=token_address,
                output_mint=self.SOL_MINT,
                amount=token_amount,
                slippage_bps=self.config.max_slippage_bps,
            )
            if not quote:
                self.logger.warning("Jupiter quote failed for migrated token sell")
                return False

            # Get swap transaction
            swap_tx_b64 = await self.jupiter.get_swap_transaction(
                quote, str(self.wallet.pubkey)
            )
            if not swap_tx_b64:
                self.logger.warning("Jupiter swap transaction build failed")
                return False

            self.logger.info(
                f"Jupiter sell fallback: {token_address[:8]}... | Reason: {reason}"
            )
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
            await asyncio.sleep(min(interval * (1.3 ** attempt), 10))
        self.logger.warning(f"Transaction {signature[:16]}... not confirmed after {max_attempts} attempts")
        return False

    async def manage_positions(self):
        """
        Continuous position management.
        The autopilot that never sleeps.
        """
        exits = self.positions.check_exit_conditions()

        for token_address, reason in exits:
            await self.execute_sell(token_address, reason)
