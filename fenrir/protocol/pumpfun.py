#!/usr/bin/env python3
"""
FENRIR - Pump.fun Program Interface

This module handles direct interaction with the pump.fun bonding curve program.
It decodes on-chain state, calculates prices, and builds buy/sell instructions.

The bonding curve formula (simplified):
- Linear curve: price = base_price + (tokens_sold * slope)
- Reserves: virtual_sol_reserves, virtual_token_reserves
- Migration: Happens at 100% curve completion (~69 SOL raised)
"""

import struct
import hashlib
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.system_program import ID as SYS_PROGRAM_ID
from solders.token.associated import get_associated_token_address
import base58


# ═══════════════════════════════════════════════════════════════════════════
#                        PUMP.FUN PROGRAM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PUMP_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
PUMP_GLOBAL = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
PUMP_FEE_RECIPIENT = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
PUMP_EVENT_AUTHORITY = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")

# Token program
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

# Instruction discriminators (Anchor-style: sha256("global:<method>")[:8])
INITIALIZE_DISCRIMINATOR = hashlib.sha256(b"global:create").digest()[:8]
BUY_DISCRIMINATOR = hashlib.sha256(b"global:buy").digest()[:8]
SELL_DISCRIMINATOR = hashlib.sha256(b"global:sell").digest()[:8]

# Bonding curve parameters
INITIAL_VIRTUAL_TOKEN_RESERVES = 1_073_000_000  # 1.073B tokens
INITIAL_VIRTUAL_SOL_RESERVES = 30_000_000_000  # 30 SOL (in lamports)
INITIAL_REAL_TOKEN_RESERVES = 793_100_000  # 793.1M real tokens
MIGRATION_THRESHOLD_SOL = 85_000_000_000  # 85 SOL triggers Raydium migration


@dataclass
class BondingCurveState:
    """
    Decoded bonding curve account state.
    This represents the current state of a token's launch.
    """
    virtual_token_reserves: int  # Virtual token reserves
    virtual_sol_reserves: int  # Virtual SOL reserves (lamports)
    real_token_reserves: int  # Real token reserves
    real_sol_reserves: int  # Real SOL reserves (lamports)
    token_total_supply: int
    complete: bool  # Has migrated to Raydium?
    
    def get_price(self) -> float:
        """
        Calculate current price in SOL per token.
        Price = virtual_sol_reserves / virtual_token_reserves
        """
        if self.virtual_token_reserves == 0:
            return 0.0
        return self.virtual_sol_reserves / self.virtual_token_reserves / 1e9
    
    def get_market_cap_sol(self) -> float:
        """Calculate market cap in SOL."""
        return self.get_price() * self.token_total_supply
    
    def get_migration_progress(self) -> float:
        """
        Calculate how close to Raydium migration (0-100%).
        Migration happens at ~85 SOL raised.
        """
        return min(100.0, (self.real_sol_reserves / MIGRATION_THRESHOLD_SOL) * 100)
    
    def calculate_buy_price(self, sol_amount: float) -> Tuple[int, float]:
        """
        Calculate tokens out for a given SOL amount.
        Uses constant product formula: x * y = k
        
        Returns: (tokens_out, price_impact_pct)
        """
        sol_lamports = int(sol_amount * 1e9)
        
        # New virtual reserves after buy
        new_virtual_sol = self.virtual_sol_reserves + sol_lamports
        
        # Constant product: k = x * y
        k = self.virtual_token_reserves * self.virtual_sol_reserves
        new_virtual_tokens = k // new_virtual_sol
        
        # Tokens received
        tokens_out = self.virtual_token_reserves - new_virtual_tokens
        
        # Price impact
        original_price = self.get_price()
        new_price = new_virtual_sol / new_virtual_tokens / 1e9
        price_impact = ((new_price - original_price) / original_price) * 100
        
        return tokens_out, price_impact
    
    def calculate_sell_price(self, token_amount: int) -> Tuple[int, float]:
        """
        Calculate SOL out for a given token amount.
        
        Returns: (sol_out_lamports, price_impact_pct)
        """
        # New virtual reserves after sell
        new_virtual_tokens = self.virtual_token_reserves + token_amount
        
        # Constant product
        k = self.virtual_token_reserves * self.virtual_sol_reserves
        new_virtual_sol = k // new_virtual_tokens
        
        # SOL received
        sol_out = self.virtual_sol_reserves - new_virtual_sol
        
        # Price impact
        original_price = self.get_price()
        new_price = new_virtual_sol / new_virtual_tokens / 1e9
        price_impact = ((original_price - new_price) / original_price) * 100
        
        return sol_out, price_impact


class PumpFunProgram:
    """
    Interface to the pump.fun bonding curve program.
    Handles account decoding and instruction building.
    """
    
    def __init__(self):
        self.program_id = PUMP_PROGRAM_ID
    
    def decode_bonding_curve(self, account_data: bytes) -> Optional[BondingCurveState]:
        """
        Decode bonding curve account data.
        
        Account layout (simplified):
        - 8 bytes: discriminator
        - 8 bytes: virtual_token_reserves
        - 8 bytes: virtual_sol_reserves
        - 8 bytes: real_token_reserves
        - 8 bytes: real_sol_reserves
        - 8 bytes: token_total_supply
        - 1 byte: complete
        - ... additional fields
        """
        try:
            if len(account_data) < 73:
                return None
            
            # Skip discriminator (first 8 bytes)
            offset = 8
            
            # Unpack reserves and supply
            virtual_token_reserves = struct.unpack('<Q', account_data[offset:offset+8])[0]
            offset += 8
            
            virtual_sol_reserves = struct.unpack('<Q', account_data[offset:offset+8])[0]
            offset += 8
            
            real_token_reserves = struct.unpack('<Q', account_data[offset:offset+8])[0]
            offset += 8
            
            real_sol_reserves = struct.unpack('<Q', account_data[offset:offset+8])[0]
            offset += 8
            
            token_total_supply = struct.unpack('<Q', account_data[offset:offset+8])[0]
            offset += 8
            
            complete = bool(account_data[offset])
            
            return BondingCurveState(
                virtual_token_reserves=virtual_token_reserves,
                virtual_sol_reserves=virtual_sol_reserves,
                real_token_reserves=real_token_reserves,
                real_sol_reserves=real_sol_reserves,
                token_total_supply=token_total_supply,
                complete=complete
            )
        except Exception as e:
            logger.error("Failed to decode bonding curve: %s", e)
            return None
    
    def build_buy_instruction(
        self,
        buyer: Pubkey,
        token_mint: Pubkey,
        bonding_curve: Pubkey,
        associated_bonding_curve: Pubkey,
        buyer_token_account: Pubkey,
        amount_sol: int,  # lamports
        max_slippage_bps: int = 500  # 5%
    ) -> Instruction:
        """
        Build a buy instruction for the pump.fun program.
        
        Instruction format:
        - 8 bytes: discriminator (buy)
        - 8 bytes: amount (SOL in lamports)
        - 8 bytes: max_sol_cost (for slippage protection)
        """
        # Calculate max SOL with slippage
        max_sol_cost = int(amount_sol * (1 + max_slippage_bps / 10000))
        
        # Build instruction data
        instruction_data = BUY_DISCRIMINATOR
        instruction_data += struct.pack('<Q', amount_sol)
        instruction_data += struct.pack('<Q', max_sol_cost)
        
        # Build accounts list
        accounts = [
            AccountMeta(pubkey=PUMP_GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FEE_RECIPIENT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=buyer_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=buyer, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
        ]
        
        return Instruction(
            program_id=self.program_id,
            accounts=accounts,
            data=instruction_data
        )
    
    def build_sell_instruction(
        self,
        seller: Pubkey,
        token_mint: Pubkey,
        bonding_curve: Pubkey,
        associated_bonding_curve: Pubkey,
        seller_token_account: Pubkey,
        amount_tokens: int,
        min_sol_output: int  # minimum SOL to receive (slippage protection)
    ) -> Instruction:
        """
        Build a sell instruction for the pump.fun program.
        
        Instruction format:
        - 8 bytes: discriminator (sell)
        - 8 bytes: amount (tokens)
        - 8 bytes: min_sol_output (for slippage protection)
        """
        # Build instruction data
        instruction_data = SELL_DISCRIMINATOR
        instruction_data += struct.pack('<Q', amount_tokens)
        instruction_data += struct.pack('<Q', min_sol_output)
        
        # Build accounts list
        accounts = [
            AccountMeta(pubkey=PUMP_GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FEE_RECIPIENT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=seller_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=seller, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
        ]
        
        return Instruction(
            program_id=self.program_id,
            accounts=accounts,
            data=instruction_data
        )
    
    def derive_bonding_curve_address(self, token_mint: Pubkey) -> Tuple[Pubkey, int]:
        """
        Derive the bonding curve PDA for a token.
        PDA = findProgramAddress(["bonding-curve", token_mint], PUMP_PROGRAM)
        """
        seeds = [b"bonding-curve", bytes(token_mint)]
        return Pubkey.find_program_address(seeds, self.program_id)
    
    def get_associated_bonding_curve_address(
        self,
        bonding_curve: Pubkey,
        token_mint: Pubkey
    ) -> Pubkey:
        """
        Get the associated token account for the bonding curve.
        This holds the actual token reserves.
        """
        return get_associated_token_address(bonding_curve, token_mint)


class TokenLaunchDetector:
    """
    Detect new token launches by monitoring pump.fun program logs.
    """
    
    def __init__(self):
        self.program = PumpFunProgram()
    
    def is_create_instruction(self, instruction_data: bytes) -> bool:
        """Check if instruction is a token creation."""
        if len(instruction_data) < 8:
            return False
        discriminator = instruction_data[:8]
        return discriminator == INITIALIZE_DISCRIMINATOR
    
    def parse_create_event(
        self,
        instruction_data: bytes,
        accounts: list
    ) -> Optional[Dict]:
        """
        Parse token creation instruction to extract launch details.
        
        Returns:
        {
            "token_mint": str,
            "bonding_curve": str,
            "creator": str,
            "name": str,
            "symbol": str,
            "uri": str
        }
        """
        try:
            # Skip discriminator
            offset = 8
            
            # Parse name (variable length string)
            name_len = struct.unpack('<I', instruction_data[offset:offset+4])[0]
            offset += 4
            name = instruction_data[offset:offset+name_len].decode('utf-8')
            offset += name_len
            
            # Parse symbol
            symbol_len = struct.unpack('<I', instruction_data[offset:offset+4])[0]
            offset += 4
            symbol = instruction_data[offset:offset+symbol_len].decode('utf-8')
            offset += symbol_len
            
            # Parse URI (metadata)
            uri_len = struct.unpack('<I', instruction_data[offset:offset+4])[0]
            offset += 4
            uri = instruction_data[offset:offset+uri_len].decode('utf-8')
            
            # Extract accounts
            token_mint = str(accounts[0]) if len(accounts) > 0 else None
            bonding_curve = str(accounts[2]) if len(accounts) > 2 else None
            creator = str(accounts[7]) if len(accounts) > 7 else None
            
            return {
                "token_mint": token_mint,
                "bonding_curve": bonding_curve,
                "creator": creator,
                "name": name,
                "symbol": symbol,
                "uri": uri
            }
        except Exception as e:
            logger.error("Failed to parse create event: %s", e)
            return None
    
    def is_buy_instruction(self, instruction_data: bytes) -> bool:
        """Check if instruction is a buy."""
        if len(instruction_data) < 8:
            return False
        return instruction_data[:8] == BUY_DISCRIMINATOR
    
    def is_sell_instruction(self, instruction_data: bytes) -> bool:
        """Check if instruction is a sell."""
        if len(instruction_data) < 8:
            return False
        return instruction_data[:8] == SELL_DISCRIMINATOR


# ═══════════════════════════════════════════════════════════════════════════
#                              UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_optimal_buy_amount(
    bonding_curve_state: BondingCurveState,
    max_price_impact_pct: float = 5.0
) -> float:
    """
    Calculate the maximum SOL amount to buy without exceeding price impact.
    Uses binary search to find optimal amount.
    """
    low, high = 0.001, 100.0  # Search range: 0.001 to 100 SOL
    epsilon = 0.001  # Precision
    
    best_amount = 0.0
    
    while high - low > epsilon:
        mid = (low + high) / 2
        _, price_impact = bonding_curve_state.calculate_buy_price(mid)
        
        if price_impact <= max_price_impact_pct:
            best_amount = mid
            low = mid
        else:
            high = mid
    
    return best_amount


def estimate_profit_at_migration(
    entry_price: float,
    bonding_curve_state: BondingCurveState
) -> float:
    """
    Estimate profit % if you hold until Raydium migration.
    Migration creates initial Raydium pool with remaining liquidity.
    """
    # At migration, curve is complete (85 SOL raised)
    migration_price = MIGRATION_THRESHOLD_SOL / INITIAL_VIRTUAL_TOKEN_RESERVES / 1e9
    
    # Calculate expected ROI
    roi = ((migration_price - entry_price) / entry_price) * 100
    return roi


if __name__ == "__main__":
    # Example usage
    print("🐺 FENRIR - Pump.fun Program Interface")
    print("=" * 70)
    
    # Initialize program interface
    program = PumpFunProgram()
    
    # Example: Calculate buy for 0.1 SOL on fresh launch
    fresh_curve = BondingCurveState(
        virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
        virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
        real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
        real_sol_reserves=0,
        token_total_supply=1_000_000_000,
        complete=False
    )
    
    buy_amount_sol = 0.1
    tokens_out, price_impact = fresh_curve.calculate_buy_price(buy_amount_sol)
    
    print(f"\n📊 Fresh Launch Analysis:")
    print(f"   Initial Price: ${fresh_curve.get_price():.10f} per token")
    print(f"   Market Cap: {fresh_curve.get_market_cap_sol():.2f} SOL")
    print(f"   Migration Progress: {fresh_curve.get_migration_progress():.1f}%")
    print(f"\n💰 Buying {buy_amount_sol} SOL:")
    print(f"   Tokens Out: {tokens_out:,}")
    print(f"   Price Impact: {price_impact:.2f}%")
    print(f"   Avg Entry Price: ${buy_amount_sol * 1e9 / tokens_out:.10f}")
    
    # Calculate optimal buy amount for 5% max price impact
    optimal_amount = calculate_optimal_buy_amount(fresh_curve, max_price_impact_pct=5.0)
    print(f"\n🎯 Optimal Buy (5% max impact): {optimal_amount:.3f} SOL")
    
    # Estimate profit at migration
    entry_price = buy_amount_sol * 1e9 / tokens_out
    profit_at_migration = estimate_profit_at_migration(entry_price, fresh_curve)
    print(f"📈 Estimated Profit at Migration: {profit_at_migration:+.1f}%")
