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

import hashlib
import logging
import struct
from dataclasses import dataclass

from solders.instruction import AccountMeta, Instruction
from solders.pubkey import Pubkey
from solders.system_program import ID as SYS_PROGRAM_ID
from solders.token.associated import get_associated_token_address

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#                        PUMP.FUN PROGRAM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PUMP_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
PUMP_GLOBAL = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
# Primary protocol fee recipient. This ROTATES on-chain — the engine reads the
# live value from the global account (offset 41) at trade time; this constant is
# only the fallback. Verified current value as of the live round-trip below.
PUMP_FEE_RECIPIENT = Pubkey.from_string("62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV")
PUMP_EVENT_AUTHORITY = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")

# Offset of `fee_recipient` inside the global account: 8 (disc) + 1 (initialized
# bool) + 32 (authority) = 41. Read at runtime so fee-recipient rotation can't
# brick trading (a stale value → on-chain error 6000 NotAuthorized).
GLOBAL_FEE_RECIPIENT_OFFSET = 41

# Token programs (pump.fun mints may be classic SPL or Token-2022)
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
TOKEN_2022_PROGRAM = Pubkey.from_string("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")
ASSOCIATED_TOKEN_PROGRAM = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

# Fee accounts added by pump.fun's creator-fee update (fixed global addresses,
# verified against live buy/sell txs).
PUMP_FEE_CONFIG = Pubkey.from_string("8Wf5TiAheLUqBrKXeYg2JtAFFMWtKdG2BSFgqUcPVwTt")
PUMP_FEE_PROGRAM = Pubkey.from_string("pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ")

# pump.fun's v2 "buyback" fee-sharing appends two *writable* fee accounts to buy
# and sell as remaining-accounts (the published IDL predates this, so it lists
# only 16/14 accounts — the deployed program needs 18/16). Order and writability
# matter: idx0 is the buyback recipient, idx1 a fee-pool recipient. Verified
# against real mainnet txs and a live 0.01 SOL buy+sell round-trip:
#   buy  2Q7fZUF... (bal 0 -> 10_000_000), sell 3gWCcrm... (10_000_000 -> 0).
# Wrong values fail simulation (6000 NotAuthorized / 6024 Overflow / 6057/6062
# BuybackFeeRecipient), so a bad refresh can never silently spend SOL. If buys
# start failing with those codes, refresh these from a recent successful tx.
PUMP_BUYBACK_FEE_RECIPIENT = Pubkey.from_string("Etb9fCF6PyY9grPPj9h8SZt5qimYHhySbfrGS7wfFqBz")
PUMP_FEE_POOL_RECIPIENT = Pubkey.from_string("GXPFM2caqTtQYC2cJ5yJRi9VDkpsYZXzYdwYpGnLmtDL")

# Instruction discriminators (Anchor-style: sha256("global:<method>")[:8])
# pump.fun migrated token creation from `create` to `create_v2`; accept both so
# detection survives the upgrade and any lingering legacy launches.
INITIALIZE_DISCRIMINATOR = hashlib.sha256(b"global:create").digest()[:8]
CREATE_V2_DISCRIMINATOR = hashlib.sha256(b"global:create_v2").digest()[:8]
CREATE_DISCRIMINATORS = frozenset({INITIALIZE_DISCRIMINATOR, CREATE_V2_DISCRIMINATOR})
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
    creator: str | None = None  # coin creator (for the creator_vault PDA)

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

    def calculate_buy_price(self, sol_amount: float) -> tuple[int, float]:
        """
        Calculate tokens out for a given SOL amount.
        Uses constant product formula: x * y = k

        Returns: (tokens_out, price_impact_pct)
        """
        sol_lamports = int(sol_amount * 1e9)

        # New virtual reserves after buy
        new_virtual_sol = self.virtual_sol_reserves + sol_lamports
        if new_virtual_sol == 0:
            return 0, 100.0

        # Constant product: k = x * y
        k = self.virtual_token_reserves * self.virtual_sol_reserves
        new_virtual_tokens = k // new_virtual_sol
        if new_virtual_tokens == 0:
            return 0, 100.0

        # Tokens received
        tokens_out = self.virtual_token_reserves - new_virtual_tokens

        # Price impact
        original_price = self.get_price()
        if original_price == 0:
            return tokens_out, 0.0
        new_price = new_virtual_sol / new_virtual_tokens / 1e9
        price_impact = ((new_price - original_price) / original_price) * 100

        return tokens_out, price_impact

    def calculate_sell_price(self, token_amount: int) -> tuple[int, float]:
        """
        Calculate SOL out for a given token amount.

        Returns: (sol_out_lamports, price_impact_pct)
        """
        # New virtual reserves after sell
        new_virtual_tokens = self.virtual_token_reserves + token_amount
        if new_virtual_tokens == 0:
            return 0, 100.0

        # Constant product
        k = self.virtual_token_reserves * self.virtual_sol_reserves
        new_virtual_sol = k // new_virtual_tokens

        # SOL received
        sol_out = self.virtual_sol_reserves - new_virtual_sol

        # Price impact
        original_price = self.get_price()
        if original_price == 0 or new_virtual_tokens == 0:
            return sol_out, 0.0
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

    def decode_bonding_curve(self, account_data: bytes) -> BondingCurveState | None:
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
            virtual_token_reserves = struct.unpack("<Q", account_data[offset : offset + 8])[0]
            offset += 8

            virtual_sol_reserves = struct.unpack("<Q", account_data[offset : offset + 8])[0]
            offset += 8

            real_token_reserves = struct.unpack("<Q", account_data[offset : offset + 8])[0]
            offset += 8

            real_sol_reserves = struct.unpack("<Q", account_data[offset : offset + 8])[0]
            offset += 8

            token_total_supply = struct.unpack("<Q", account_data[offset : offset + 8])[0]
            offset += 8

            complete = bool(account_data[offset])
            offset += 1

            # creator pubkey (added by the creator-fee update) — needed to
            # derive the creator_vault account for buy/sell.
            creator = None
            if len(account_data) >= offset + 32:
                creator = str(Pubkey.from_bytes(account_data[offset : offset + 32]))

            return BondingCurveState(
                virtual_token_reserves=virtual_token_reserves,
                virtual_sol_reserves=virtual_sol_reserves,
                real_token_reserves=real_token_reserves,
                real_sol_reserves=real_sol_reserves,
                token_total_supply=token_total_supply,
                complete=complete,
                creator=creator,
            )
        except Exception as e:
            logger.error("Failed to decode bonding curve: %s", e)
            return None

    # ── PDA / ATA derivation (IDL-accurate) ────────────────────────────

    def derive_ata(self, owner: Pubkey, token_mint: Pubkey, token_program: Pubkey) -> Pubkey:
        """Associated token account, using the mint's token program."""
        return Pubkey.find_program_address(
            [bytes(owner), bytes(token_program), bytes(token_mint)], ASSOCIATED_TOKEN_PROGRAM
        )[0]

    def derive_creator_vault(self, creator: Pubkey) -> Pubkey:
        return Pubkey.find_program_address([b"creator-vault", bytes(creator)], self.program_id)[0]

    def derive_global_volume_accumulator(self) -> Pubkey:
        return Pubkey.find_program_address([b"global_volume_accumulator"], self.program_id)[0]

    def derive_user_volume_accumulator(self, user: Pubkey) -> Pubkey:
        return Pubkey.find_program_address(
            [b"user_volume_accumulator", bytes(user)], self.program_id
        )[0]

    def parse_global_fee_recipient(self, global_data: bytes) -> Pubkey | None:
        """Read the live primary fee_recipient from the pump global account.

        It rotates on-chain, so the engine reads it fresh each trade rather than
        trusting the constant (a stale value → 6000 NotAuthorized).
        """
        end = GLOBAL_FEE_RECIPIENT_OFFSET + 32
        if len(global_data) < end:
            return None
        return Pubkey.from_bytes(global_data[GLOBAL_FEE_RECIPIENT_OFFSET:end])

    def build_create_ata_instruction(
        self, payer: Pubkey, owner: Pubkey, token_mint: Pubkey, token_program: Pubkey
    ) -> Instruction:
        """Idempotent create-ATA (data=[1]) using the mint's token program."""
        ata = self.derive_ata(owner, token_mint, token_program)
        accounts = [
            AccountMeta(pubkey=payer, is_signer=True, is_writable=True),
            AccountMeta(pubkey=ata, is_signer=False, is_writable=True),
            AccountMeta(pubkey=owner, is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_program, is_signer=False, is_writable=False),
        ]
        return Instruction(program_id=ASSOCIATED_TOKEN_PROGRAM, accounts=accounts, data=bytes([1]))

    def build_buy_instruction(
        self,
        buyer: Pubkey,
        token_mint: Pubkey,
        bonding_curve: Pubkey,
        creator: Pubkey,
        token_program: Pubkey,
        amount_tokens: int,  # exact token base units to buy
        max_sol_cost: int,  # lamports cap (slippage protection)
        fee_recipient: Pubkey | None = None,
        buyback_fee_recipient: Pubkey | None = None,
        fee_pool_recipient: Pubkey | None = None,
    ) -> Instruction:
        """
        Build a pump.fun `buy` instruction (live 18-account layout).

        pump.fun's `buy` takes the EXACT token amount to receive plus a SOL cap —
        NOT a SOL amount. Data: discriminator(8) + amount(u64, token base units) +
        max_sol_cost(u64, lamports) + track_volume(1 byte, 0x00 = don't track).

        The published IDL lists 16 accounts; the deployed program needs two extra
        *writable* v2 buyback fee accounts appended (see PUMP_BUYBACK_FEE_RECIPIENT).
        fee_recipient defaults to the module constant but the engine passes the
        live value read from the global account (it rotates).
        """
        data = (
            BUY_DISCRIMINATOR + struct.pack("<Q", amount_tokens) + struct.pack("<Q", max_sol_cost)
        )
        data += b"\x00"  # track_volume = None

        accounts = [
            AccountMeta(pubkey=PUMP_GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(
                pubkey=fee_recipient or PUMP_FEE_RECIPIENT, is_signer=False, is_writable=True
            ),
            AccountMeta(pubkey=token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(
                pubkey=self.derive_ata(bonding_curve, token_mint, token_program),
                is_signer=False,
                is_writable=True,
            ),
            AccountMeta(
                pubkey=self.derive_ata(buyer, token_mint, token_program),
                is_signer=False,
                is_writable=True,
            ),
            AccountMeta(pubkey=buyer, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_program, is_signer=False, is_writable=False),
            AccountMeta(
                pubkey=self.derive_creator_vault(creator), is_signer=False, is_writable=True
            ),
            AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(
                pubkey=self.derive_global_volume_accumulator(), is_signer=False, is_writable=False
            ),
            AccountMeta(
                pubkey=self.derive_user_volume_accumulator(buyer), is_signer=False, is_writable=True
            ),
            AccountMeta(pubkey=PUMP_FEE_CONFIG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FEE_PROGRAM, is_signer=False, is_writable=False),
            # v2 buyback fee accounts (writable) — appended remaining-accounts.
            AccountMeta(
                pubkey=buyback_fee_recipient or PUMP_BUYBACK_FEE_RECIPIENT,
                is_signer=False,
                is_writable=True,
            ),
            AccountMeta(
                pubkey=fee_pool_recipient or PUMP_FEE_POOL_RECIPIENT,
                is_signer=False,
                is_writable=True,
            ),
        ]
        return Instruction(program_id=self.program_id, accounts=accounts, data=data)

    def build_sell_instruction(
        self,
        seller: Pubkey,
        token_mint: Pubkey,
        bonding_curve: Pubkey,
        creator: Pubkey,
        token_program: Pubkey,
        amount_tokens: int,
        min_sol_output: int,  # minimum SOL to receive (slippage protection)
        fee_recipient: Pubkey | None = None,
        buyback_fee_recipient: Pubkey | None = None,
        fee_pool_recipient: Pubkey | None = None,
    ) -> Instruction:
        """
        Build a pump.fun `sell` instruction (live 16-account layout).

        Note vs buy: creator_vault comes BEFORE token_program, and there are no
        volume accumulators / track_volume arg. Data: disc(8) + amount(u64) +
        min_sol_output(u64). Like buy, the deployed program needs two extra
        *writable* v2 buyback fee accounts appended (IDL lists only 14).
        """
        data = (
            SELL_DISCRIMINATOR
            + struct.pack("<Q", amount_tokens)
            + struct.pack("<Q", min_sol_output)
        )

        accounts = [
            AccountMeta(pubkey=PUMP_GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(
                pubkey=fee_recipient or PUMP_FEE_RECIPIENT, is_signer=False, is_writable=True
            ),
            AccountMeta(pubkey=token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(
                pubkey=self.derive_ata(bonding_curve, token_mint, token_program),
                is_signer=False,
                is_writable=True,
            ),
            AccountMeta(
                pubkey=self.derive_ata(seller, token_mint, token_program),
                is_signer=False,
                is_writable=True,
            ),
            AccountMeta(pubkey=seller, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(
                pubkey=self.derive_creator_vault(creator), is_signer=False, is_writable=True
            ),
            AccountMeta(pubkey=token_program, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FEE_CONFIG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FEE_PROGRAM, is_signer=False, is_writable=False),
            # v2 buyback fee accounts (writable) — appended remaining-accounts.
            AccountMeta(
                pubkey=buyback_fee_recipient or PUMP_BUYBACK_FEE_RECIPIENT,
                is_signer=False,
                is_writable=True,
            ),
            AccountMeta(
                pubkey=fee_pool_recipient or PUMP_FEE_POOL_RECIPIENT,
                is_signer=False,
                is_writable=True,
            ),
        ]
        return Instruction(program_id=self.program_id, accounts=accounts, data=data)

    def derive_bonding_curve_address(self, token_mint: Pubkey) -> tuple[Pubkey, int]:
        """
        Derive the bonding curve PDA for a token.
        PDA = findProgramAddress(["bonding-curve", token_mint], PUMP_PROGRAM)
        """
        seeds = [b"bonding-curve", bytes(token_mint)]
        return Pubkey.find_program_address(seeds, self.program_id)

    def get_associated_bonding_curve_address(
        self, bonding_curve: Pubkey, token_mint: Pubkey
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
        return instruction_data[:8] in CREATE_DISCRIMINATORS

    def parse_create_event(self, instruction_data: bytes, accounts: list) -> dict | None:
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
            name_len = struct.unpack("<I", instruction_data[offset : offset + 4])[0]
            offset += 4
            name = instruction_data[offset : offset + name_len].decode("utf-8")
            offset += name_len

            # Parse symbol
            symbol_len = struct.unpack("<I", instruction_data[offset : offset + 4])[0]
            offset += 4
            symbol = instruction_data[offset : offset + symbol_len].decode("utf-8")
            offset += symbol_len

            # Parse URI (metadata)
            uri_len = struct.unpack("<I", instruction_data[offset : offset + 4])[0]
            offset += 4
            uri = instruction_data[offset : offset + uri_len].decode("utf-8")

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
                "uri": uri,
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
    bonding_curve_state: BondingCurveState, max_price_impact_pct: float = 5.0
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
    entry_price: float, bonding_curve_state: BondingCurveState
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
        complete=False,
    )

    buy_amount_sol = 0.1
    tokens_out, price_impact = fresh_curve.calculate_buy_price(buy_amount_sol)

    print("\n📊 Fresh Launch Analysis:")
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
