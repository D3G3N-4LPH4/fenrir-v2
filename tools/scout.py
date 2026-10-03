#!/usr/bin/env python3
"""FENRIR market scout — find buy candidates using the bot's own pipeline.

Each run:
  1. Pull token addresses per chain from the discovery sources:
       - boosted:    DexScreener boosted (promoted) tokens
       - gecko_new:  GeckoTerminal newest pools (earliest post-launch listings)
       - gecko_trending: GeckoTerminal trending pools (momentum)
       - ds_profile: DexScreener latest paid token profiles (promotion signal)
       - graduation: pump.fun tokens at 50-85% of the bonding curve (Solana)
       - rh_onchain: Uniswap v4 pools initialized on Robinhood Chain in the
         last 6h, seen at block zero via eth_getLogs (Robinhood)
  2. Build a TokenSnapshot for each (most-liquid pair), deduped across sources.
  3. Enrich safety: GoPlus for covered EVM chains, RugCheck for Solana.
  4. Run FilterEngine (low_cap_alpha / mid_cap_momentum / high_cap) + ScoringEngine.
  5. Emit candidates: passes >= 1 filter, score >= MIN_SCORE, no hard safety fail.

Stdout: JSON {"ts": ..., "scanned": N, "by_source": {...}, "candidates": [...]}.
Dedup / alerting is the caller's job (the cron worker keeps a seen-list).

Usage:
  python tools/scout.py [--chains solana robinhood] [--limit 25]
      [--extra-limit 12] [--min-score 60] [--sources boosted gecko ds_profile]
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

from fenrir.discovery.acceleration import AccelTracker  # noqa: E402
from fenrir.discovery.filters import FilterEngine, FilterName  # noqa: E402
from fenrir.discovery.models import Chain  # noqa: E402
from fenrir.discovery.playbooks import PlaybookTagger  # noqa: E402
from fenrir.discovery.providers.dexscreener import DexScreenerProvider  # noqa: E402
from fenrir.discovery.providers.geckoterminal import GeckoTerminalProvider  # noqa: E402
from fenrir.discovery.providers.goplus import GoPlusProvider  # noqa: E402
from fenrir.discovery.providers.perceptor import PerceptorProvider  # noqa: E402
from fenrir.discovery.providers.robinhood_safety import (  # noqa: E402
    RobinhoodSafetyProvider,
    enrich_robinhood_safety,
)
from fenrir.discovery.scoring import ScoringEngine  # noqa: E402

# Source names in dedup-priority order (first source keeps credit on overlap).
SOURCE_BOOSTED = "boosted"
SOURCE_GECKO_NEW = "gecko_new"
SOURCE_GECKO_TRENDING = "gecko_trending"
SOURCE_DS_PROFILE = "ds_profile"
SOURCE_GRADUATION = "graduation"
SOURCE_ONCHAIN = "rh_onchain"

SOURCE_GROUPS = ("boosted", "gecko", "ds_profile", "graduation", "onchain")

# Hard deadlines. Every provider already carries a per-request timeout, but a
# single wedged coroutine (or a provider whose timeout never fires) must not
# be able to stall the batch — the scout runs on a 10-minute cadence.
TOKEN_TIMEOUT_SECONDS = 90.0  # one address, snapshot -> verdict
PERCEPTOR_TIMEOUT_SECONDS = 45.0  # the Perceptor enrichment leg of one token
SOURCE_TIMEOUT_SECONDS = 60.0  # one discovery source fetch
RUN_TIMEOUT_SECONDS = 420.0  # the whole run; emits partial results on breach


def safety_unknown(snap) -> bool:
    """True when no safety provider supplied any meaningful signal.

    Delegates to the model's ``is_empty`` — the single definition of
    "safety unknown" used by the scoring cap.
    """
    return bool(snap.safety.is_empty)


def hard_fail(snap) -> str | None:
    s = snap.safety
    if s.honeypot:
        return "honeypot"
    if (s.sell_tax_pct or 0) > 15 or (s.buy_tax_pct or 0) > 15:
        return f"tax buy={s.buy_tax_pct}% sell={s.sell_tax_pct}%"
    if s.mint_disabled is False:
        return "mint authority live"
    return None


async def fetch_graduation_addresses(
    gt: GeckoTerminalProvider, chain: Chain, limit: int
) -> list[str]:
    """Solana tokens sitting at 50-85% of the pump.fun bonding curve.

    The base universe is GeckoTerminal's fresh pools; each address gets one
    batched curve-state read and the reading is recorded for velocity. This
    is the pre-DexScreener-momentum discovery the other sources can't see.
    """
    from fenrir.discovery.providers.pumpfun import PumpFunProvider

    provider = PumpFunProvider()
    try:
        base = await gt.fetch_new_pool_addresses(chain, limit * 2)
        if not base:
            return []
        states = await provider.curve_states(base)
        now = time.time()
        out: list[str] = []
        for mint, state in states.items():
            if state.complete:
                continue
            progress = state.get_migration_progress()
            if 50.0 <= progress <= 85.0:
                provider.record_reading(mint, state, now)
                out.append(mint)
        provider.prune_state()
        return out
    finally:
        await provider.close()


async def fetch_onchain_addresses(limit: int) -> list[str]:
    """Robinhood tokens first seen on-chain within the fresh window.

    The ``rh-pair-watch`` cron keeps the registry warm every 2 minutes; this
    just reads it (no RPC here, so the scout stays fast). Tokens DexScreener
    hasn't indexed yet evaluate to None downstream and are retried next
    cycle while still fresh.
    """
    from fenrir.discovery.providers.rh_onchain import RobinhoodPairMonitor

    monitor = RobinhoodPairMonitor()
    try:
        # Opportunistic sync: if the fast cron hasn't run yet (e.g. first
        # run after migration), advance the cursor here instead of waiting.
        await monitor.sync()
        return monitor.fresh_addresses(max_age_hours=6.0)[:limit]
    finally:
        await monitor.close()


async def fetch_source_addresses(
    chain: Chain,
    ds: DexScreenerProvider,
    gt: GeckoTerminalProvider,
    sources: list[str],
    limit: int,
    extra_limit: int,
) -> list[tuple[str, list[str]]]:
    """Pull addresses per source for one chain. Fail-open per source."""
    out: list[tuple[str, list[str]]] = []

    async def safe(coro, name: str) -> list[str]:
        try:
            # Hard deadline per source: a wedged feed must not stall discovery.
            result: list[str] = await asyncio.wait_for(coro, timeout=SOURCE_TIMEOUT_SECONDS)
            return result
        except Exception:
            return []

    if "boosted" in sources:
        out.append((SOURCE_BOOSTED, await safe(ds.fetch_boosted_addresses(chain), SOURCE_BOOSTED)))
    if "gecko" in sources:
        out.append(
            (
                SOURCE_GECKO_NEW,
                await safe(gt.fetch_new_pool_addresses(chain, extra_limit), SOURCE_GECKO_NEW),
            )
        )
        out.append(
            (
                SOURCE_GECKO_TRENDING,
                await safe(gt.fetch_trending_addresses(chain, extra_limit), SOURCE_GECKO_TRENDING),
            )
        )
    if "ds_profile" in sources:
        out.append(
            (
                SOURCE_DS_PROFILE,
                await safe(ds.fetch_profiled_addresses(chain, extra_limit), SOURCE_DS_PROFILE),
            )
        )
    if "graduation" in sources and chain is Chain.SOLANA:
        out.append(
            (
                SOURCE_GRADUATION,
                await safe(fetch_graduation_addresses(gt, chain, extra_limit), SOURCE_GRADUATION),
            )
        )
    if "onchain" in sources and chain is Chain.ROBINHOOD:
        out.append(
            (
                SOURCE_ONCHAIN,
                await safe(fetch_onchain_addresses(extra_limit), SOURCE_ONCHAIN),
            )
        )

    # Per-source caps (boosted uses --limit; extras use --extra-limit).
    # On-chain is high-volume (~100 pools/h on Robinhood) and cheap to check
    # (an unindexed token is one fast DexScreener miss), so it gets headroom.
    caps = {
        SOURCE_BOOSTED: limit,
        SOURCE_GECKO_NEW: extra_limit,
        SOURCE_GECKO_TRENDING: extra_limit,
        SOURCE_DS_PROFILE: extra_limit,
        SOURCE_GRADUATION: extra_limit,
        SOURCE_ONCHAIN: extra_limit * 3,
    }
    return [(name, addrs[: caps[name]]) for name, addrs in out]


def dedupe_sources(source_addrs: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
    """Flatten (source, addresses) into (source, address), deduped by address.

    First source in list order keeps credit on overlap. Pure — unit tested.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, addrs in source_addrs:
        for addr in addrs:
            if addr in seen:
                continue
            seen.add(addr)
            out.append((name, addr))
    return out


async def evaluate_address(
    source: str,
    addr: str,
    chain: Chain | None,
    ds: DexScreenerProvider,
    gp: GoPlusProvider,
    engine: FilterEngine,
    scorer: ScoringEngine,
    tagger: PlaybookTagger,
    min_score: float,
    perceptor: PerceptorProvider | None = None,
    accel: AccelTracker | None = None,
    timings: list[dict] | None = None,
    local_safety: RobinhoodSafetyProvider | None = None,
) -> dict | None:
    """Run one address through snapshot + safety + filters + scoring.

    Returns the candidate dict, or None when it doesn't clear the bar.
    When ``timings`` is given, per-phase seconds are appended as
    {"addr", "phase", "seconds"} dicts for run-level diagnostics.
    """

    def _phase(name: str, t0: float) -> None:
        if timings is not None:
            timings.append(
                {"addr": addr, "phase": name, "seconds": round(time.perf_counter() - t0, 3)}
            )

    try:
        t0 = time.perf_counter()
        snap = await ds.fetch_snapshot(addr, chain=chain)
        _phase("snapshot", t0)
    except Exception:
        return None
    if snap is None:
        return None
    try:
        t0 = time.perf_counter()
        await enrich_safety(snap, gp)
        _phase("safety", t0)
    except Exception:
        pass
    # Solana: live bonding-curve position (graduation_watch filter data).
    if snap.chain is Chain.SOLANA:
        t0 = time.perf_counter()
        try:
            from fenrir.discovery.providers.pumpfun import annotate_bond_curve

            await annotate_bond_curve(snap)
        except Exception:
            pass
        _phase("bond", t0)
    # Acceleration: seed this poll's observation and attach poll-over-poll
    # growth before the filters run (momentum_transition fails closed without
    # a prior sighting).
    if accel is not None:
        try:
            accel.record(snap)
        except Exception:
            pass
    fail = hard_fail(snap)
    t0 = time.perf_counter()
    results = {fn.value: engine.evaluate(snap, fn) for fn in FilterName}
    passed = [k for k, r in results.items() if r.passed]
    score = scorer.score(snap)
    _phase("score", t0)
    # Robinhood safety net: when GoPlus has nothing, Perceptor's on-chain
    # forensics can still clear (or kill) the candidate. A landed verdict
    # re-runs the hard-fail check and the score — the safety_unknown score
    # cap lifts when safety becomes verifiable.
    perceptor_info: dict | None = None
    if (
        (local_safety is not None or perceptor is not None)
        and snap.chain is Chain.ROBINHOOD
        and safety_unknown(snap)
    ):
        t0 = time.perf_counter()
        try:
            # Hard deadline on top of the provider's own request timeouts: a
            # wedged safety call must not eat this token's whole budget.
            # Local on-chain reader first; Perceptor's stale cache as fallback
            # (its API is auth-walled — no new investigations are attempted).
            report = await asyncio.wait_for(
                enrich_robinhood_safety(snap, local_safety, perceptor),
                timeout=PERCEPTOR_TIMEOUT_SECONDS,
            )
        except Exception:  # noqa: BLE001 - fail-open (includes TimeoutError)
            report = None
        _phase("perceptor", t0)
        if report is not None:
            fail = hard_fail(snap)
            score = scorer.score(snap)
            perceptor_info = {
                "status": "complete",
                "band": report.band,
                "band_label": report.band_label,
                "headline": report.headline,
                "investigation_id": report.investigation_id,
            }
    if fail or not passed or score.overall < min_score:
        return None
    ratio_1h = snap.buy_sell_ratio_1h
    return {
        "address": snap.token_address,
        "chain": snap.chain.value,
        "symbol": snap.symbol,
        "name": snap.name,
        "source": source,
        "price_usd": snap.price_usd,
        "market_cap_usd": round(snap.market_cap_usd, 2),
        "liquidity_usd": round(snap.liquidity_usd, 2),
        "volume_24h_usd": round(snap.volume_24h_usd, 2),
        "age_minutes": round(snap.age_minutes or 0),
        "buys_1h": snap.txns_1h_buys,
        "sells_1h": snap.txns_1h_sells,
        "buys_24h": snap.txns_24h_buys,
        "sells_24h": snap.txns_24h_sells,
        "buy_sell_ratio_1h": (
            round(ratio_1h, 2) if ratio_1h is not None and ratio_1h != float("inf") else None
        ),
        "volume_1h_share_pct": (
            round(snap.volume_1h_share * 100, 1) if snap.volume_1h_share is not None else None
        ),
        "turnover_24h": round(snap.turnover_24h, 2) if snap.turnover_24h else None,
        "price_change_1h_pct": snap.price_change_1h_pct,
        "price_change_24h_pct": snap.price_change_24h_pct,
        "holder_count": snap.holder_count,
        "top_holder_pct": snap.top_holder_pct,
        "top10_holder_pct": snap.top10_holder_pct,
        "passed_filters": passed,
        "filter_warnings": [w for r in results.values() for w in r.warnings],
        "bond_progress_pct": snap.bond_progress_pct,
        "bond_inflow_sol": snap.bond_inflow_sol,
        "bond_sol_remaining": snap.bond_sol_remaining,
        "playbooks": tagger.tag(snap).as_dict(),
        "score": score.as_dict(),
        "safety_unknown": safety_unknown(snap),
        "perceptor": perceptor_info,
        "dexscreener": f"https://dexscreener.com/{snap.chain.value}/{snap.token_address}",
    }


async def scout_chain(
    chain: Chain,
    ds: DexScreenerProvider,
    gt: GeckoTerminalProvider,
    gp: GoPlusProvider,
    engine: FilterEngine,
    scorer: ScoringEngine,
    tagger: PlaybookTagger,
    sources: list[str],
    limit: int,
    extra_limit: int,
    min_score: float,
    perceptor: PerceptorProvider | None = None,
    accel: AccelTracker | None = None,
    timings: list[dict] | None = None,
    local_safety: RobinhoodSafetyProvider | None = None,
) -> tuple[list[dict], dict[str, int]]:
    candidates: list[dict] = []
    by_source: dict[str, int] = {}
    source_addrs = await fetch_source_addresses(chain, ds, gt, sources, limit, extra_limit)
    addrs = list(dedupe_sources(source_addrs))
    # Bounded concurrency: serial evaluation turns provider latency into
    # total runtime (102 tokens x ~20s = 33+ min, blowing the 10-min cadence).
    # asyncio.gather preserves input order, so candidate ordering is unchanged.
    # 8-wide: the work is network-IO-bound, and each token already carries a
    # 90s hard deadline, so a slow token can't hog the batch.
    sem = asyncio.Semaphore(8)

    async def _one(source: str, addr: str, stagger: float) -> tuple[str, str, dict | None]:
        async with sem:
            if stagger:
                await asyncio.sleep(stagger)
            try:
                # Per-token hard deadline: no single address may stall the
                # batch, no matter which provider call wedges underneath.
                cand = await asyncio.wait_for(
                    evaluate_address(
                        source,
                        addr,
                        chain,
                        ds,
                        gp,
                        engine,
                        scorer,
                        tagger,
                        min_score,
                        perceptor,
                        accel,
                        timings,
                        local_safety=local_safety,
                    ),
                    timeout=TOKEN_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                if timings is not None:
                    # Record how far the token got before the deadline fired —
                    # the last completed phase is where it was stuck.
                    done = [e["phase"] for e in timings if e["addr"] == addr]
                    timings.append(
                        {
                            "addr": addr,
                            "phase": "timeout",
                            "seconds": TOKEN_TIMEOUT_SECONDS,
                            "after": done[-1] if done else None,
                        }
                    )
                cand = None
            except Exception:  # noqa: BLE001 - one bad token must not kill the batch
                cand = None
            return source, addr, cand

    results = await asyncio.gather(
        *[_one(source, addr, (i % 8) * 0.3) for i, (source, addr) in enumerate(addrs)]
    )
    for source, addr, cand in results:
        by_source[source] = by_source.get(source, 0) + 1
        if cand is not None:
            candidates.append(cand)
    return candidates, by_source


def summarize_timings(timings: list[dict]) -> dict:
    """Aggregate per-phase/per-token timings into a compact run diagnostic."""
    by_phase: dict[str, float] = {}
    by_addr: dict[str, float] = {}
    timeouts = 0
    timeout_after: dict[str, int] = {}
    for t in timings:
        phase = str(t.get("phase"))
        secs = float(t.get("seconds") or 0.0)
        by_phase[phase] = by_phase.get(phase, 0.0) + secs
        if phase == "timeout":
            timeouts += 1
            after = t.get("after")
            key = str(after) if after else "snapshot"
            timeout_after[key] = timeout_after.get(key, 0) + 1
        else:
            addr = str(t.get("addr"))
            by_addr[addr] = by_addr.get(addr, 0.0) + secs
    slowest = sorted(by_addr.items(), key=lambda kv: -kv[1])[:10]
    return {
        "phase_seconds": {
            k: round(v, 1) for k, v in sorted(by_phase.items(), key=lambda kv: -kv[1])
        },
        "slowest_tokens": [{"addr": a, "seconds": round(s, 1)} for a, s in slowest],
        "token_timeouts": timeouts,
        "timeout_after_phase": timeout_after,
        "tokens_timed": len(by_addr),
    }


async def amain() -> int:
    ap = argparse.ArgumentParser(description="FENRIR market scout")
    ap.add_argument(
        "--chains", nargs="+", default=["solana", "robinhood"], choices=[c.value for c in Chain]
    )
    ap.add_argument("--limit", type=int, default=25, help="max boosted tokens per chain")
    ap.add_argument(
        "--extra-limit",
        type=int,
        default=12,
        help="max tokens per extra source per chain (gecko_new, gecko_trending, ds_profile)",
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_GROUPS),
        choices=list(SOURCE_GROUPS),
        help="source groups to poll",
    )
    ap.add_argument("--min-score", type=float, default=60.0)
    args = ap.parse_args()

    ds = DexScreenerProvider(timeout_seconds=15)
    gt = GeckoTerminalProvider(timeout_seconds=15)
    gp = GoPlusProvider(timeout_seconds=10)
    perceptor = PerceptorProvider()
    local_safety = RobinhoodSafetyProvider()
    accel = AccelTracker(AccelTracker.default_state_path())
    engine = FilterEngine()
    scorer = ScoringEngine()
    tagger = PlaybookTagger()
    all_cands: list[dict] = []
    by_source: dict[str, int] = {}
    timings: list[dict] = []
    degraded: str | None = None
    try:
        # Run-level deadline: the cron cadence is 10 minutes, so a wedged run
        # must emit partial results instead of hanging until it gets killed.
        async with asyncio.timeout(RUN_TIMEOUT_SECONDS):
            for c in args.chains:
                cands, bs = await scout_chain(
                    Chain(c),
                    ds,
                    gt,
                    gp,
                    engine,
                    scorer,
                    tagger,
                    args.sources,
                    args.limit,
                    args.extra_limit,
                    args.min_score,
                    perceptor,
                    accel,
                    timings,
                    local_safety=local_safety,
                )
                all_cands.extend(cands)
                for k, v in bs.items():
                    by_source[k] = by_source.get(k, 0) + v
    except TimeoutError:
        degraded = f"run deadline {RUN_TIMEOUT_SECONDS:.0f}s exceeded — partial results"
    finally:
        await ds.close()
        await gt.close()
        await gp.close()
        await perceptor.close()
        accel.save()

    all_cands.sort(key=lambda c: -c["score"]["overall"])
    timing = summarize_timings(timings)
    print(
        f"scout timing: scanned={sum(by_source.values())} "
        f"phases={timing['phase_seconds']} timeouts={timing['token_timeouts']} "
        f"timeout_after={timing['timeout_after_phase']} "
        f"slowest={[t['addr'][:10] + '=' + str(t['seconds']) + 's' for t in timing['slowest_tokens'][:5]]}",
        file=sys.stderr,
    )
    out: dict = {
        "ts": time.time(),
        "scanned": sum(by_source.values()),
        "by_source": by_source,
        "candidates": all_cands,
        "timing": timing,
    }
    if degraded:
        out["degraded"] = degraded
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
