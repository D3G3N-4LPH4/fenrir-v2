"""Tests for the scout's newer discovery sources (GeckoTerminal, DS profiles)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.discovery.models import Chain  # noqa: E402
from fenrir.discovery.providers.dexscreener import extract_profiled_addresses  # noqa: E402
from fenrir.discovery.providers.geckoterminal import extract_pool_token_addresses  # noqa: E402
from tools.scout import dedupe_sources  # noqa: E402


def _gt_pool(token_id: str) -> dict:
    return {"relationships": {"base_token": {"data": {"id": token_id, "type": "token"}},
                              "quote_token": {"data": {"id": "solana_So11111111111111111111111111111111111111112",
                                                       "type": "token"}}},
            "attributes": {"name": "X / SOL"}}


def test_geckoterminal_parser_strips_prefix_and_dedupes() -> None:
    payload = {"data": [
        _gt_pool("solana_AAA111"),
        _gt_pool("solana_BBB222"),
        _gt_pool("solana_AAA111"),  # dup
        _gt_pool("solana_So11111111111111111111111111111111111111112"),  # wSOL quote
        _gt_pool("ethereum_CCC333"),  # wrong network prefix
        {"relationships": {}},  # malformed
        "not-a-dict",
    ]}
    out = extract_pool_token_addresses(payload, "solana", Chain.SOLANA)
    assert out == ["AAA111", "BBB222"]


def test_geckoterminal_parser_never_raises() -> None:
    assert extract_pool_token_addresses(None, "solana", Chain.SOLANA) == []
    assert extract_pool_token_addresses({"data": "nope"}, "solana", Chain.SOLANA) == []
    assert extract_pool_token_addresses({}, "solana", Chain.SOLANA) == []


def test_ds_profiles_parser_filters_chain_and_dedupes() -> None:
    payload = [
        {"chainId": "solana", "tokenAddress": "AAA111"},
        {"chainId": "bsc", "tokenAddress": "0xBBB"},
        {"chainId": "solana", "tokenAddress": "AAA111"},  # dup
        {"chainId": "solana", "tokenAddress": "CCC333"},
        {"chainId": "robinhood", "tokenAddress": "0xDDD"},
        "junk",
    ]
    assert extract_profiled_addresses(payload, Chain.SOLANA) == ["AAA111", "CCC333"]
    assert extract_profiled_addresses(payload, Chain.ROBINHOOD) == ["0xDDD"]
    assert extract_profiled_addresses(None, Chain.SOLANA) == []


def test_dedupe_sources_first_source_keeps_credit() -> None:
    srcs = [("boosted", ["A", "B"]), ("gecko_new", ["B", "C"]), ("ds_profile", ["C", "D"])]
    assert dedupe_sources(srcs) == [("boosted", "A"), ("boosted", "B"),
                                    ("gecko_new", "C"), ("ds_profile", "D")]
