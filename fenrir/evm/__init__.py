"""
FENRIR EVM — on-chain EVM (Ethereum/Base/BNB) support, read-only first.

Bridges the multi-chain discovery ``TokenSnapshot`` onto the existing strategy / signal
/ AI-brain machinery so EVM tokens run through the SAME decision code as Solana, without
chain leakage. Read-only: evaluation only, no execution (on-chain swaps are a later,
gated PR).
"""

from fenrir.evm.adapters import snapshot_to_market_data, snapshot_to_token_data
from fenrir.evm.evaluator import EvmEvaluation, EvmTokenEvaluator

__all__ = [
    "snapshot_to_market_data",
    "snapshot_to_token_data",
    "EvmTokenEvaluator",
    "EvmEvaluation",
]
