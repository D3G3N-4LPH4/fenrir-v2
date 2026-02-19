#!/usr/bin/env python3
"""
FENRIR - Devnet Integration Tests

Real-network tests against Solana devnet.
These verify that our RPC client, wallet operations, and transaction
building work end-to-end against a live (devnet) Solana cluster.

Pump.fun, Jupiter, and Jito do NOT exist on devnet.
These tests exercise the Solana primitive layer only.

Run with: pytest tests/test_devnet_integration.py -v -m devnet
"""

import asyncio

import pytest
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import ID as SYS_PROGRAM_ID
from solders.system_program import TransferParams, transfer
from solders.token.associated import get_associated_token_address

from fenrir.config import BotConfig, TradingMode
from fenrir.core.client import SolanaClient
from fenrir.core.wallet import WalletManager
from fenrir.logger import FenrirLogger
from fenrir.protocol.pumpfun import (
    INITIAL_REAL_TOKEN_RESERVES,
    INITIAL_VIRTUAL_SOL_RESERVES,
    INITIAL_VIRTUAL_TOKEN_RESERVES,
    BondingCurveState,
    PumpFunProgram,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

DEVNET_RPC = "https://api.devnet.solana.com"
DEVNET_WS = "wss://api.devnet.solana.com"
DEVNET_TIMEOUT_SECONDS = 30

pytestmark = pytest.mark.devnet

# ═══════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def devnet_config():
    """BotConfig wired to devnet."""
    return BotConfig(
        rpc_url=DEVNET_RPC,
        ws_url=DEVNET_WS,
        mode=TradingMode.SIMULATION,
        priority_fee_lamports=1_000,
    )


@pytest.fixture
def devnet_logger(devnet_config):
    return FenrirLogger(devnet_config)


@pytest.fixture
def devnet_client(devnet_config, devnet_logger):
    """SolanaClient connected to devnet. Fresh per test to avoid event loop issues."""
    return SolanaClient(devnet_config, devnet_logger)


@pytest.fixture
def devnet_wallet():
    """Ephemeral keypair for devnet tests. No real funds at risk."""
    return WalletManager("", simulation_mode=True)


@pytest.fixture
def devnet_keypair():
    """Raw Keypair for low-level transaction tests."""
    return Keypair()


# ═══════════════════════════════════════════════════════════════════════════
#  RPC Connectivity
# ═══════════════════════════════════════════════════════════════════════════


class TestDevnetRPCConnectivity:
    """Verify basic RPC operations work against devnet."""

    async def test_get_latest_blockhash(self, devnet_client):
        """Fetching a blockhash from devnet should return a non-None value."""
        # Devnet can be intermittent; retry once
        blockhash = await devnet_client.get_latest_blockhash()
        if blockhash is None:
            await asyncio.sleep(2)
            blockhash = await devnet_client.get_latest_blockhash()
        assert blockhash is not None

    async def test_get_balance_of_system_program(self, devnet_client):
        """System program account should exist."""
        balance = await devnet_client.get_balance(SYS_PROGRAM_ID)
        assert isinstance(balance, float)
        assert balance >= 0.0

    async def test_get_account_info_for_known_account(self, devnet_client):
        """System program account info should complete without error."""
        data = await devnet_client.get_account_info(SYS_PROGRAM_ID)
        assert data is None or isinstance(data, bytes)

    async def test_get_account_info_nonexistent_returns_none(self, devnet_client):
        """Random nonexistent account should return None."""
        random_pubkey = Keypair().pubkey()
        data = await devnet_client.get_account_info(random_pubkey)
        assert data is None

    async def test_get_balance_of_ephemeral_wallet(self, devnet_client, devnet_wallet):
        """Freshly created wallet should have zero balance on devnet."""
        pubkey = Pubkey.from_string(devnet_wallet.get_address())
        balance = await devnet_client.get_balance(pubkey)
        assert balance == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Wallet Operations
# ═══════════════════════════════════════════════════════════════════════════


class TestDevnetWalletOperations:
    """Test wallet-related operations on devnet."""

    def test_simulation_wallet_generates_valid_keypair(self, devnet_wallet):
        """Simulation wallet should produce a valid Solana pubkey."""
        address = devnet_wallet.get_address()
        assert len(address) >= 32
        pubkey = Pubkey.from_string(address)
        assert str(pubkey) == address

    def test_keypair_pubkey_is_valid(self, devnet_keypair):
        """Keypair should produce a valid public key."""
        pubkey = devnet_keypair.pubkey()
        assert pubkey is not None
        assert len(bytes(pubkey)) == 32


# ═══════════════════════════════════════════════════════════════════════════
#  Transaction Building & Simulation
# ═══════════════════════════════════════════════════════════════════════════


class TestDevnetTransactionBuilding:
    """Test building and simulating transactions on devnet."""

    async def test_build_transfer_instruction(self, devnet_keypair):
        """Building a SOL transfer instruction should produce a valid instruction."""
        receiver = Keypair().pubkey()
        ix = transfer(
            TransferParams(
                from_pubkey=devnet_keypair.pubkey(),
                to_pubkey=receiver,
                lamports=1000,
            )
        )
        assert ix is not None

    async def test_blockhash_is_fresh(self, devnet_client):
        """Two consecutive blockhash fetches should work."""
        bh1 = await devnet_client.get_latest_blockhash()
        bh2 = await devnet_client.get_latest_blockhash()
        assert bh1 is not None
        assert bh2 is not None

    async def test_get_signatures_for_system_program(self, devnet_client):
        """System program should return a list of signatures."""
        sigs = await devnet_client.get_recent_signatures(SYS_PROGRAM_ID, limit=5)
        assert isinstance(sigs, list)


# ═══════════════════════════════════════════════════════════════════════════
#  Bonding Curve Math (Pure Logic)
# ═══════════════════════════════════════════════════════════════════════════


def _fresh_curve() -> BondingCurveState:
    """Create a fresh bonding curve at initial state."""
    return BondingCurveState(
        virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
        virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
        real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
        real_sol_reserves=0,
        token_total_supply=1_000_000_000,
        complete=False,
    )


class TestBondingCurveMath:
    """Pure math tests for bonding curve calculations."""

    def test_fresh_curve_price(self):
        """Initial bonding curve should produce a consistent price."""
        curve = _fresh_curve()
        price = curve.get_price()
        assert price > 0
        # Price ~ 30 SOL / 1.073B tokens
        assert 0.00000001 < price < 0.001

    def test_buy_price_positive_tokens(self):
        """Buying any SOL amount should yield positive tokens."""
        curve = _fresh_curve()
        tokens, impact = curve.calculate_buy_price(0.1)
        assert tokens > 0
        assert impact > 0

    def test_sell_price_positive_sol(self):
        """Selling tokens should yield positive SOL (in lamports)."""
        curve = _fresh_curve()
        sol_out, impact = curve.calculate_sell_price(1_000_000)
        assert sol_out > 0
        assert impact > 0

    def test_buy_moves_price_up(self):
        """Buying tokens should increase the price (fundamental bonding curve property)."""
        curve = _fresh_curve()
        price_before = curve.get_price()

        buy_sol = 1.0
        tokens_out, _ = curve.calculate_buy_price(buy_sol)

        # Update curve state as if the buy happened
        buy_lamports = int(buy_sol * 1e9)
        post_buy_curve = BondingCurveState(
            virtual_token_reserves=curve.virtual_token_reserves - tokens_out,
            virtual_sol_reserves=curve.virtual_sol_reserves + buy_lamports,
            real_token_reserves=curve.real_token_reserves - tokens_out,
            real_sol_reserves=curve.real_sol_reserves + buy_lamports,
            token_total_supply=curve.token_total_supply,
            complete=False,
        )

        price_after = post_buy_curve.get_price()
        assert price_after > price_before

    def test_pda_derivation_deterministic(self):
        """PDA derivation should be deterministic for the same mint."""
        program = PumpFunProgram()
        mint = Pubkey.from_string("So11111111111111111111111111111111111111112")
        pda1, bump1 = program.derive_bonding_curve_address(mint)
        pda2, bump2 = program.derive_bonding_curve_address(mint)
        assert pda1 == pda2
        assert bump1 == bump2

    def test_migration_progress_zero_at_start(self):
        """Migration progress should be 0% on a fresh curve."""
        curve = _fresh_curve()
        assert curve.get_migration_progress() == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Pump.fun Instruction Building (Offline)
# ═══════════════════════════════════════════════════════════════════════════


class TestPumpfunInstructionBuilding:
    """Build pump.fun instructions and verify structure."""

    def test_build_buy_instruction_structure(self, devnet_keypair):
        """Buy instruction should have correct number of accounts and data layout."""
        program = PumpFunProgram()
        fake_mint = Keypair().pubkey()
        bonding_curve, _ = program.derive_bonding_curve_address(fake_mint)
        assoc = program.get_associated_bonding_curve_address(bonding_curve, fake_mint)
        buyer_ata = get_associated_token_address(devnet_keypair.pubkey(), fake_mint)

        ix = program.build_buy_instruction(
            buyer=devnet_keypair.pubkey(),
            token_mint=fake_mint,
            bonding_curve=bonding_curve,
            associated_bonding_curve=assoc,
            buyer_token_account=buyer_ata,
            amount_sol=100_000_000,
            max_slippage_bps=500,
        )

        assert ix is not None
        assert len(ix.accounts) == 10
        assert len(ix.data) == 24  # 8 (discriminator) + 8 (amount) + 8 (max_sol_cost)

    def test_build_sell_instruction_structure(self, devnet_keypair):
        """Sell instruction should have correct number of accounts and data layout."""
        program = PumpFunProgram()
        fake_mint = Keypair().pubkey()
        bonding_curve, _ = program.derive_bonding_curve_address(fake_mint)
        assoc = program.get_associated_bonding_curve_address(bonding_curve, fake_mint)
        seller_ata = get_associated_token_address(devnet_keypair.pubkey(), fake_mint)

        ix = program.build_sell_instruction(
            seller=devnet_keypair.pubkey(),
            token_mint=fake_mint,
            bonding_curve=bonding_curve,
            associated_bonding_curve=assoc,
            seller_token_account=seller_ata,
            amount_tokens=1_000_000,
            min_sol_output=50_000_000,
        )

        assert ix is not None
        assert len(ix.accounts) == 10
        assert len(ix.data) == 24  # 8 + 8 + 8
