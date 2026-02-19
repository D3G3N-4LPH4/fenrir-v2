#!/usr/bin/env python3
"""
FENRIR - Solana Client

The interface between FENRIR and Solana.
Every RPC call is a question. This class asks them eloquently.
"""

import asyncio
from typing import Optional, Dict, List

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts, TokenAccountOpts
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solders.signature import Signature

from fenrir.config import BotConfig
from fenrir.logger import FenrirLogger

# Default timeout for individual RPC calls (seconds)
RPC_TIMEOUT_SECONDS = 15


class SolanaClient:
    """
    The interface between FENRIR and Solana.
    Every RPC call is a question. This class asks them eloquently.
    """

    def __init__(self, config: BotConfig, logger: FenrirLogger):
        self.config = config
        self.logger = logger
        self.client = AsyncClient(config.rpc_url)
        self.pumpfun_program = Pubkey.from_string(config.pumpfun_program)

    async def get_balance(self, pubkey: Pubkey) -> float:
        """Check SOL balance with grace."""
        try:
            response = await asyncio.wait_for(
                self.client.get_balance(pubkey),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            return response.value / 1e9  # Lamports to SOL
        except asyncio.TimeoutError:
            self.logger.warning("RPC timeout: get_balance")
            return 0.0
        except Exception as e:
            self.logger.error("Failed to fetch balance", e)
            return 0.0

    async def get_recent_signatures(
        self,
        address: Pubkey,
        limit: int = 10
    ) -> List[Dict]:
        """Retrieve recent transactions."""
        try:
            response = await asyncio.wait_for(
                self.client.get_signatures_for_address(address, limit=limit),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            return response.value if response.value else []
        except asyncio.TimeoutError:
            self.logger.warning("RPC timeout: get_signatures_for_address")
            return []
        except Exception as e:
            self.logger.error("Failed to fetch signatures", e)
            return []

    async def get_transaction(self, signature: str) -> Optional[Dict]:
        """Decode a transaction's story."""
        try:
            sig = Signature.from_string(signature)
            response = await asyncio.wait_for(
                self.client.get_transaction(
                    sig,
                    encoding="jsonParsed",
                    max_supported_transaction_version=0,
                ),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            return response.value if response.value else None
        except asyncio.TimeoutError:
            self.logger.warning(f"RPC timeout: get_transaction {signature[:16]}...")
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch tx {signature}", e)
            return None

    async def simulate_transaction(self, transaction: Transaction) -> bool:
        """
        Test the waters before diving in.
        Returns True if simulation succeeds.
        """
        try:
            response = await asyncio.wait_for(
                self.client.simulate_transaction(transaction),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            if response.value.err:
                self.logger.warning(f"Simulation failed: {response.value.err}")
                return False
            return True
        except asyncio.TimeoutError:
            self.logger.warning("RPC timeout: simulate_transaction")
            return False
        except Exception as e:
            self.logger.error("Simulation error", e)
            return False

    async def send_transaction(
        self,
        transaction: Transaction,
        skip_preflight: bool = False
    ) -> Optional[str]:
        """
        Broadcast to the network.
        The moment code becomes action on-chain.
        """
        try:
            opts = TxOpts(
                skip_preflight=skip_preflight,
                preflight_commitment=Confirmed
            )
            response = await asyncio.wait_for(
                self.client.send_transaction(transaction, opts),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            return str(response.value)
        except asyncio.TimeoutError:
            self.logger.warning("RPC timeout: send_transaction")
            return None
        except Exception as e:
            self.logger.error("Failed to send transaction", e)
            return None

    async def get_latest_blockhash(self):
        """Fetch a recent blockhash for building transactions."""
        try:
            response = await asyncio.wait_for(
                self.client.get_latest_blockhash(commitment=Confirmed),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            return response.value.blockhash
        except asyncio.TimeoutError:
            self.logger.warning("RPC timeout: get_latest_blockhash")
            return None
        except Exception as e:
            self.logger.error("Failed to fetch latest blockhash", e)
            return None

    async def get_account_info(self, pubkey: Pubkey) -> Optional[bytes]:
        """Fetch raw account data bytes."""
        try:
            response = await asyncio.wait_for(
                self.client.get_account_info(pubkey, commitment=Confirmed),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            if response.value and response.value.data:
                return bytes(response.value.data)
            return None
        except asyncio.TimeoutError:
            self.logger.warning(f"RPC timeout: get_account_info {pubkey}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch account info for {pubkey}", e)
            return None

    async def get_token_accounts_by_owner(
        self,
        owner: Pubkey,
        mint: Pubkey
    ) -> Optional[Dict]:
        """Get token account and balance for a specific mint."""
        try:
            response = await asyncio.wait_for(
                self.client.get_token_accounts_by_owner_json_parsed(
                    owner, TokenAccountOpts(mint=mint)
                ),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            if response.value:
                for account_info in response.value:
                    parsed = account_info.account.data.parsed
                    token_amount = parsed["info"]["tokenAmount"]
                    return {
                        "address": account_info.pubkey,
                        "amount": int(token_amount["amount"]),
                        "decimals": token_amount["decimals"],
                        "ui_amount": float(token_amount["uiAmount"] or 0),
                    }
            return None
        except asyncio.TimeoutError:
            self.logger.warning(f"RPC timeout: get_token_accounts_by_owner {owner}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch token accounts for {owner}", e)
            return None

    async def get_signature_statuses(
        self, signatures: List[str]
    ) -> Optional[List[Dict]]:
        """Check confirmation status of transaction signatures."""
        try:
            sigs = [Signature.from_string(s) for s in signatures]
            response = await asyncio.wait_for(
                self.client.get_signature_statuses(sigs),
                timeout=RPC_TIMEOUT_SECONDS,
            )
            return response.value if response.value else None
        except asyncio.TimeoutError:
            self.logger.warning("RPC timeout: get_signature_statuses")
            return None
        except Exception as e:
            self.logger.error("Failed to fetch signature statuses", e)
            return None

    async def close(self):
        """Graceful shutdown."""
        await self.client.close()
