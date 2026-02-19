#!/usr/bin/env python3
"""
FENRIR - Wallet Management

Your keys, your crypto. Handle with the reverence they deserve.
"""

import base58
from solders.keypair import Keypair
from solders.transaction import Transaction


class WalletManager:
    """
    Your keys, your crypto. Handle with the reverence they deserve.
    Never logs private keys. Never stores them unencrypted.
    """

    def __init__(self, private_key_b58: str, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode

        if simulation_mode:
            # Generate a throwaway keypair for testing
            self.keypair = Keypair()
            self.pubkey = self.keypair.pubkey()
        else:
            if not private_key_b58:
                raise ValueError("Private key required for live trading")

            try:
                private_key_bytes = base58.b58decode(private_key_b58)
                self.keypair = Keypair.from_bytes(private_key_bytes)
                self.pubkey = self.keypair.pubkey()
            except Exception as e:
                raise ValueError(f"Invalid private key format: {e}") from e

    def get_address(self) -> str:
        """Return the wallet's public address."""
        return str(self.pubkey)

    def sign_transaction(self, transaction: Transaction) -> Transaction:
        """Sign with the elegance of a digital signature."""
        if self.simulation_mode:
            return transaction  # Don't actually sign in sim
        transaction.sign([self.keypair])
        return transaction
