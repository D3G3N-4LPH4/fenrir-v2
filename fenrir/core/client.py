#!/usr/bin/env python3
"""
FENRIR - Solana Client

The interface between FENRIR and Solana.
Every RPC call is a question. This class asks them eloquently.
"""

import asyncio

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TokenAccountOpts, TxOpts
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import Transaction

from fenrir.config import BotConfig
from fenrir.core.circuit_breaker import CircuitBreaker, CircuitOpen
from fenrir.logger import FenrirLogger

# Default timeout for individual RPC calls (seconds)
RPC_TIMEOUT_SECONDS = 15


class SolanaClient:
    """
    The interface between FENRIR and Solana.
    Every RPC call is a question. This class asks them eloquently.
    """

    def __init__(
        self, config: BotConfig, logger: FenrirLogger, breaker: CircuitBreaker | None = None
    ):
        self.config = config
        self.logger = logger
        self._breaker = breaker
        self.client = AsyncClient(config.rpc_url)
        self.pumpfun_program = Pubkey.from_string(config.pumpfun_program)

    async def _rpc(self, coro, name: str, timeout: float = RPC_TIMEOUT_SECONDS):  # noqa: ASYNC109 - timeout drives the asyncio.wait_for below
        """Run one RPC coroutine with circuit-breaker protection and timeout."""
        try:
            if self._breaker:
                self._breaker.check()
            result = await asyncio.wait_for(coro, timeout=timeout)
            if self._breaker:
                self._breaker.record_success()
            return result
        except CircuitOpen:
            self.logger.warning(f"RPC circuit OPEN: {name}")
            return None
        except TimeoutError:
            if self._breaker:
                self._breaker.record_failure("timeout")
            self.logger.warning(f"RPC timeout: {name}")
            return None
        except Exception as e:
            if self._breaker:
                self._breaker.record_failure(type(e).__name__)
            self.logger.error(f"RPC error: {name}", e)
            return None

    async def get_balance(self, pubkey: Pubkey) -> float:
        """Check SOL balance with grace."""
        resp = await self._rpc(self.client.get_balance(pubkey), "get_balance")
        return resp.value / 1e9 if resp else 0.0

    async def get_recent_signatures(self, address: Pubkey, limit: int = 10) -> list:
        """Retrieve recent transactions."""
        resp = await self._rpc(
            self.client.get_signatures_for_address(address, limit=limit),
            "get_signatures_for_address",
        )
        return resp.value if resp and resp.value else []

    async def get_transaction(self, signature: str):
        """Decode a transaction's story."""
        sig = Signature.from_string(signature)
        resp = await self._rpc(
            self.client.get_transaction(
                sig,
                encoding="jsonParsed",
                commitment=Confirmed,
                max_supported_transaction_version=0,
            ),
            f"get_transaction:{signature[:16]}",
        )
        return resp.value if resp else None

    async def simulate_transaction(self, transaction: Transaction) -> bool:
        """Test the waters before diving in. Returns True if simulation succeeds.

        Simulate at Confirmed commitment to match the blockhash fetched by
        get_latest_blockhash() and the send preflight. Without this, the client
        defaults to the finalized bank (~32 slots behind), which doesn't yet
        know the recent confirmed blockhash → spurious BlockhashNotFound.
        """
        resp = await self._rpc(
            self.client.simulate_transaction(transaction, commitment=Confirmed),
            "simulate_transaction",
        )
        if resp is None:
            return False
        if resp.value.err:
            self.logger.warning(f"Simulation failed: {resp.value.err}")
            return False
        return True

    async def send_transaction(
        self, transaction: Transaction, skip_preflight: bool = False
    ) -> str | None:
        """Broadcast to the network. The moment code becomes action on-chain."""
        opts = TxOpts(skip_preflight=skip_preflight, preflight_commitment=Confirmed)
        resp = await self._rpc(self.client.send_transaction(transaction, opts), "send_transaction")
        return str(resp.value) if resp else None

    async def get_latest_blockhash(self):
        """Fetch a recent blockhash for building transactions."""
        resp = await self._rpc(
            self.client.get_latest_blockhash(commitment=Confirmed), "get_latest_blockhash"
        )
        return resp.value.blockhash if resp else None

    async def get_account_info(self, pubkey: Pubkey) -> bytes | None:
        """Fetch raw account data bytes."""
        resp = await self._rpc(
            self.client.get_account_info(pubkey, commitment=Confirmed),
            f"get_account_info:{pubkey}",
        )
        if resp and resp.value and resp.value.data:
            return bytes(resp.value.data)
        return None

    async def get_token_accounts_by_owner(self, owner: Pubkey, mint: Pubkey) -> dict | None:
        """Get token account and balance for a specific mint."""
        resp = await self._rpc(
            self.client.get_token_accounts_by_owner_json_parsed(owner, TokenAccountOpts(mint=mint)),
            f"get_token_accounts:{owner}",
        )
        if resp and resp.value:
            for account_info in resp.value:
                parsed = account_info.account.data.parsed
                token_amount = parsed["info"]["tokenAmount"]
                return {
                    "address": account_info.pubkey,
                    "amount": int(token_amount["amount"]),
                    "decimals": token_amount["decimals"],
                    "ui_amount": float(token_amount["uiAmount"] or 0),
                }
        return None

    async def get_signature_statuses(self, signatures: list[str]):
        """Check confirmation status of transaction signatures."""
        sigs = [Signature.from_string(s) for s in signatures]
        resp = await self._rpc(self.client.get_signature_statuses(sigs), "get_signature_statuses")
        return resp.value if resp and resp.value else None

    async def close(self):
        """Graceful shutdown."""
        await self.client.close()
