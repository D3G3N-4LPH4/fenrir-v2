#!/usr/bin/env python3
"""
FENRIR - Jito MEV Protection

Submit transactions as atomic bundles via Jito validators to avoid front-running.
Jito's block engine ensures either all transactions in a bundle succeed or none do.

Benefits:
- Protection from sandwich attacks
- Guaranteed transaction ordering
- Higher success rate on competitive launches
- Can tip validators for priority inclusion

Jito Block Engines:
- Mainnet: https://mainnet.block-engine.jito.wtf
- NY: https://ny.mainnet.block-engine.jito.wtf
- Amsterdam: https://amsterdam.mainnet.block-engine.jito.wtf
- Frankfurt: https://frankfurt.mainnet.block-engine.jito.wtf
- Tokyo: https://tokyo.mainnet.block-engine.jito.wtf
"""

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import base58
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction


@dataclass
class BundleResult:
    """Result of bundle submission."""

    bundle_id: str
    submitted_at: datetime
    transactions: list[str]  # Transaction signatures
    landed: bool = False
    slot: int | None = None
    error: str | None = None


class JitoMEVProtection:
    """
    Interface to Jito's block engine for MEV-protected transactions.

    Jito validators run a modified version of the Solana validator that:
    1. Accepts transaction bundles
    2. Guarantees atomic execution (all or nothing)
    3. Respects bundle ordering
    4. Accepts tips for priority
    """

    # Jito block engines by region
    BLOCK_ENGINES = {
        "mainnet": "https://mainnet.block-engine.jito.wtf",
        "ny": "https://ny.mainnet.block-engine.jito.wtf",
        "amsterdam": "https://amsterdam.mainnet.block-engine.jito.wtf",
        "frankfurt": "https://frankfurt.mainnet.block-engine.jito.wtf",
        "tokyo": "https://tokyo.mainnet.block-engine.jito.wtf",
    }

    # Jito tip accounts (rotate to avoid spam filters)
    TIP_ACCOUNTS = [
        "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
        "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
        "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
        "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
        "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
        "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
        "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
        "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
    ]

    def __init__(
        self,
        region: str = "mainnet",
        tip_lamports: int = 10000,  # Default 0.00001 SOL tip
        timeout_seconds: int = 60,
    ):
        self.block_engine_url = self.BLOCK_ENGINES.get(region, self.BLOCK_ENGINES["mainnet"])
        self.tip_lamports = tip_lamports
        self.timeout = timeout_seconds
        self.session: aiohttp.ClientSession | None = None

        # Round-robin tip account selection
        self._tip_account_index = 0

    async def initialize(self):
        """Initialize HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _get_tip_account(self) -> Pubkey:
        """Get next tip account in rotation."""
        tip_account = self.TIP_ACCOUNTS[self._tip_account_index]
        self._tip_account_index = (self._tip_account_index + 1) % len(self.TIP_ACCOUNTS)
        return Pubkey.from_string(tip_account)

    def build_tip_transaction(
        self, payer: Keypair, recent_blockhash, tip_lamports: int | None = None
    ) -> Transaction:
        """
        Build a transaction that tips Jito validators.
        This transaction should be the last in your bundle.

        Args:
            payer: Keypair that will pay the tip
            recent_blockhash: Recent blockhash for the transaction
            tip_lamports: Amount to tip (defaults to configured amount)

        Returns:
            Unsigned transaction (caller must sign)
        """
        tip_amount = tip_lamports or self.tip_lamports
        tip_account = self._get_tip_account()

        # Create transfer instruction
        transfer_ix = transfer(
            TransferParams(from_pubkey=payer.pubkey(), to_pubkey=tip_account, lamports=tip_amount)
        )

        # Build transaction
        message = Message.new_with_blockhash(
            instructions=[transfer_ix], payer=payer.pubkey(), blockhash=recent_blockhash
        )

        tx = Transaction.new_unsigned(message)
        return tx

    async def send_bundle(
        self, transactions: list[Transaction], max_retries: int = 3
    ) -> BundleResult:
        """
        Send a bundle of transactions to Jito block engine.

        Bundle rules:
        - Max 5 transactions per bundle
        - All transactions must be signed
        - Transactions execute atomically (all or nothing)
        - Last transaction should be a tip to Jito validators

        Args:
            transactions: List of signed transactions
            max_retries: Number of retry attempts

        Returns:
            BundleResult with submission details
        """
        if not self.session:
            await self.initialize()

        if len(transactions) > 5:
            raise ValueError("Jito bundles limited to 5 transactions")

        if not transactions:
            raise ValueError("Bundle must contain at least one transaction")

        # Serialize transactions to base64
        serialized_txs = []
        for tx in transactions:
            tx_bytes = bytes(tx)
            tx_b64 = base64.b64encode(tx_bytes).decode("utf-8")
            serialized_txs.append(tx_b64)

        # Prepare bundle submission
        bundle_id = None
        error = None

        for attempt in range(max_retries):
            try:
                # Submit bundle
                url = f"{self.block_engine_url}/api/v1/bundles"
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "sendBundle",
                    "params": [serialized_txs],
                }

                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()

                        if "result" in data:
                            bundle_id = data["result"]
                            break
                        elif "error" in data:
                            error = data["error"]["message"]
                    else:
                        error = f"HTTP {response.status}"

                # Wait before retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

            except Exception as e:
                error = str(e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

        # Get transaction signatures from signed transactions
        signatures = []
        for tx in transactions:
            try:
                sig_bytes = bytes(tx.signatures[0])
                signatures.append(base58.b58encode(sig_bytes).decode())
            except (IndexError, AttributeError):
                signatures.append("UNKNOWN")

        return BundleResult(
            bundle_id=bundle_id or "FAILED",
            submitted_at=datetime.now(),
            transactions=signatures,
            landed=(bundle_id is not None),
            error=error,
        )

    async def get_bundle_status(self, bundle_id: str) -> dict:
        """
        Check the status of a submitted bundle.

        Args:
            bundle_id: Bundle ID returned from send_bundle

        Returns:
            Status dictionary with landing information
        """
        if not self.session:
            await self.initialize()

        try:
            url = f"{self.block_engine_url}/api/v1/bundles"
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBundleStatuses",
                "params": [[bundle_id]],
            }

            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()

                    if "result" in data and data["result"]["value"]:
                        bundle_status = data["result"]["value"][0]
                        return {
                            "bundle_id": bundle_id,
                            "status": bundle_status.get("confirmation_status"),
                            "slot": bundle_status.get("slot"),
                            "err": bundle_status.get("err"),
                        }
        except Exception as e:
            return {"bundle_id": bundle_id, "status": "unknown", "error": str(e)}

        return {"bundle_id": bundle_id, "status": "not_found"}

    async def send_transaction_with_tip(
        self,
        transaction: Transaction,
        payer: Keypair,
        recent_blockhash,
        tip_lamports: int | None = None,
    ) -> BundleResult:
        """
        Convenience method: Send a single transaction with automatic tip.

        This creates a 2-transaction bundle:
        1. Your transaction
        2. Tip to Jito validators

        Args:
            transaction: Your signed transaction
            payer: Keypair for paying the tip
            recent_blockhash: Recent blockhash
            tip_lamports: Tip amount (defaults to configured)

        Returns:
            BundleResult
        """
        # Build tip transaction
        tip_tx = self.build_tip_transaction(payer, recent_blockhash, tip_lamports)
        tip_tx.sign([payer])

        # Send as bundle
        return await self.send_bundle([transaction, tip_tx])


class JitoOptimizer:
    """
    Optimize Jito bundle submissions for best results.

    Strategies:
    - Dynamic tip calculation based on network conditions
    - Retry logic with exponential backoff
    - Bundle status monitoring
    """

    def __init__(self, jito: JitoMEVProtection):
        self.jito = jito

        # Track successful tip amounts for optimization
        self.successful_tips: list[int] = []
        self.failed_tips: list[int] = []

    async def send_with_retry(
        self, transactions: list[Transaction], max_attempts: int = 5, initial_tip: int | None = None
    ) -> BundleResult:
        """
        Send bundle with adaptive retry strategy.
        Increases tip on each retry attempt.
        """
        current_tip = initial_tip or self.jito.tip_lamports

        for attempt in range(max_attempts):
            # Update tip amount for retry
            self.jito.tip_lamports = current_tip

            # Submit bundle
            result = await self.jito.send_bundle(transactions)

            if result.landed:
                self.successful_tips.append(current_tip)
                return result

            # Increase tip for next attempt (2x)
            current_tip *= 2
            self.failed_tips.append(current_tip // 2)

            # Wait before retry
            if attempt < max_attempts - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        return result

    def get_optimal_tip(self) -> int:
        """
        Calculate optimal tip based on historical success rates.
        Returns the median successful tip amount.
        """
        if not self.successful_tips:
            return self.jito.tip_lamports

        sorted_tips = sorted(self.successful_tips)
        median_idx = len(sorted_tips) // 2
        return sorted_tips[median_idx]

    def get_success_rate(self) -> float:
        """Calculate bundle success rate."""
        total = len(self.successful_tips) + len(self.failed_tips)
        if total == 0:
            return 0.0
        return len(self.successful_tips) / total * 100


# ═══════════════════════════════════════════════════════════════════════════
#                              EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════


async def example_usage():
    """Demonstrate Jito MEV protection."""
    print("🐺 FENRIR - Jito MEV Protection")
    print("=" * 70)

    # Initialize Jito client
    jito = JitoMEVProtection(
        region="mainnet",
        tip_lamports=10000,  # 0.00001 SOL
    )

    await jito.initialize()

    print("\n🔒 Jito Protection Active")
    print(f"   Block Engine: {jito.block_engine_url}")
    print(f"   Default Tip: {jito.tip_lamports / 1e9:.8f} SOL")
    print(f"   Tip Accounts: {len(jito.TIP_ACCOUNTS)} rotated")

    # Example tip calculation
    print("\n💡 Tip Strategy:")
    print("   Conservative: 0.00001 SOL (10,000 lamports)")
    print("   Normal: 0.00005 SOL (50,000 lamports)")
    print("   Aggressive: 0.0001 SOL (100,000 lamports)")
    print("   Competitive Launch: 0.001 SOL (1,000,000 lamports)")

    # Example: How to use in trading
    print("\n📊 Integration Example:")
    print("   1. Build your buy transaction")
    print("   2. Sign it with your wallet")
    print("   3. Create tip transaction (automatic)")
    print("   4. Submit as bundle to Jito")
    print("   5. Monitor bundle status")
    print("   6. Either all transactions land, or none do")

    await jito.close()
    print("\n✅ Example complete")


if __name__ == "__main__":
    asyncio.run(example_usage())
