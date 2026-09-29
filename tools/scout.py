#!/usr/bin/env python3
"""FENRIR market scout — find buy candidates using the bot's own pipeline.

Each run:
  1. Pull boosted (promoted) token addresses per chain via DexScreenerProvider.
  2. Build a TokenSnapshot for each (most-liquid pair).
  3. Enrich safety: GoPlus for covered EVM chains, RugCheck for Solana.
  4. Run FilterEngine (low_cap_alpha / mid_cap_momentum / high_cap) + ScoringEngine.
  5. Emit candidates: passes >= 1 filter, score >= MIN_SCORE, no hard safety fail.

Stdout: JSON {"ts": ..., "scanned": N, "candidates": [...]}.
Dedup / alerting is the caller's job (the cron worker keeps a seen-list).

Usage:
  python tools/scout.py [--chains solana robinhood] [--limit 25] [--min-score 60]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.evaluate import enrich_safety  # noqa: E402

from fenrir.discovery.filters import FilterEngine, FilterName  # noqa: E402
from fenrir.discovery.models import Chain  # noqa: E402
from fenrir.discovery.playbooks import PlaybookTagger  # noqa: E402
from fenrir.discovery.providers.dexscreener import DexScreenerProvider  # noqa: E402
from fenrir.discovery.providers.goplus import GoPlusProvider  # noqa: E402
from fenrir.discovery.scoring import ScoringEngine  # noqa: E402


def safety_unknown(snap) -> bool:
    """True when no safety provider supplied any meaningful signal.

    Delegates to the model's ``is_empty`` — the single definition of
    "safety unknown" used by the scoring cap.
    """
    return snap.safety.is_empty


def hard_fail(snap) -> str | None:
    s = snap.safety
    if s.honeypot:
        return "honeypot"
    if (s.sell_tax_pct or 0) > 15 or (s.buy_tax_pct or 0) > 15:
        return f"tax buy={s.buy_tax_pct}% sell={s.sell_tax_pct}%"
    if s.mint_disabled is False:
        return "mint authority live"
    return None


async def scout_chain(chain: Chain, ds: DexScreenerProvider, gp: GoPlusProvider,
                     engine: FilterEngine, scorer: ScoringEngine, tagger: PlaybookTagger,
                     limit: int, min_score: float) -> tuple[list[dict], int]:
    candidates: list[dict] = []
    scanned = 0
    try:
        addrs = await ds.fetch_boosted_addresses(chain)
    except Exception:
        return [], 0
    for addr in addrs[:limit]:
        try:
            snap = await ds.fetch_snapshot(addr, chain=chain)
        except Exception:
            continue
        if snap is None:
            continue
        scanned += 1
        try:
            await enrich_safety(snap, gp)
        except Exception:
            pass
        fail = hard_fail(snap)
        results = {fn.value: engine.evaluate(snap, fn) for fn in FilterName}
        passed = [k for k, r in results.items() if r.passed]
        score = scorer.score(snap)
        if fail or not passed or score.overall < min_score:
            await asyncio.sleep(0.4)
            continue
        ratio_1h = snap.buy_sell_ratio_1h
        candidates.append({
            "address": snap.token_address,
            "chain": snap.chain.value,
            "symbol": snap.symbol,
            "name": snap.name,
            "price_usd": snap.price_usd,
            "market_cap_usd": round(snap.market_cap_usd, 2),
            "liquidity_usd": round(snap.liquidity_usd, 2),
            "volume_24h_usd": round(snap.volume_24h_usd, 2),
            "age_minutes": round(snap.age_minutes or 0),
            "buys_1h": snap.txns_1h_buys, "sells_1h": snap.txns_1h_sells,
            "buys_24h": snap.txns_24h_buys, "sells_24h": snap.txns_24h_sells,
            "buy_sell_ratio_1h": round(ratio_1h, 2) if ratio_1h != float("inf") else None,
            "volume_1h_share_pct": round(snap.volume_1h_share * 100, 1),
            "turnover_24h": round(snap.turnover_24h, 2) if snap.turnover_24h else None,
            "price_change_1h_pct": snap.price_change_1h_pct,
            "price_change_24h_pct": snap.price_change_24h_pct,
            "holder_count": snap.holder_count,
            "top_holder_pct": snap.top_holder_pct,
            "top10_holder_pct": snap.top10_holder_pct,
            "passed_filters": passed,
            "filter_warnings": [w for r in results.values() for w in r.warnings],
            "playbooks": tagger.tag(snap).as_dict(),
            "score": score.as_dict(),
            "safety_unknown": safety_unknown(snap),
            "dexscreener": f"https://dexscreener.com/{snap.chain.value}/{snap.token_address}",
        })
        await asyncio.sleep(0.4)
    return candidates, scanned


async def amain() -> int:
    ap = argparse.ArgumentParser(description="FENRIR market scout")
    ap.add_argument("--chains", nargs="+", default=["solana", "robinhood"],
                    choices=[c.value for c in Chain])
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--min-score", type=float, default=60.0)
    args = ap.parse_args()

    ds = DexScreenerProvider(timeout_seconds=15)
    gp = GoPlusProvider(timeout_seconds=10)
    engine = FilterEngine()
    scorer = ScoringEngine()
    tagger = PlaybookTagger()
    all_cands: list[dict] = []
    scanned = 0
    try:
        for c in args.chains:
            cands, n = await scout_chain(Chain(c), ds, gp, engine, scorer, tagger,
                                        args.limit, args.min_score)
            all_cands.extend(cands)
            scanned += n
    finally:
        await ds.close()
        await gp.close()

    all_cands.sort(key=lambda c: -c["score"]["overall"])
    print(json.dumps({
        "ts": time.time(),
        "scanned": scanned,
        "candidates": all_cands,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
