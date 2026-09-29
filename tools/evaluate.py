#!/usr/bin/env python3
"""FENRIR token evaluator — one-shot verdict for a token address.

Reuses the bot's own discovery stack instead of hand-rolled checks:
  DexScreenerProvider.fetch_snapshot -> TokenSnapshot (market data)
  GoPlusProvider (EVM: eth/bnb/base) or RugCheck (solana) -> SafetySignals
  FilterEngine  -> low_cap_alpha / mid_cap_momentum / high_cap pass-fail
  ScoringEngine -> 0-100 breakdown (momentum/safety/liquidity/holder/community/risk)

Usage:
  python tools/evaluate.py <token_address> [--chain solana|ethereum|bnb|base|robinhood] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")

from fenrir.discovery.chains.solana import (
    RUGCHECK_SUMMARY,
    enrich_jupiter_holders,
    map_rugcheck_summary,
)
from fenrir.discovery.filters import FilterEngine, FilterName
from fenrir.discovery.lp_vault import (
    check_lp_platform_vault,
    load_known_vaults,
    resolve_lp_mint,
    save_known_vaults,
)
from fenrir.discovery.models import Chain
from fenrir.discovery.playbooks import PLAYBOOK_STRATEGY_IDS, PlaybookTagger
from fenrir.discovery.providers.dexscreener import DexScreenerProvider
from fenrir.discovery.providers.goplus import GoPlusProvider, distribution_metrics
from fenrir.discovery.providers.perceptor import (
    ROBINHOOD_CHAIN_ID,
    PerceptorProvider,
    snapshot_context,
)
from fenrir.discovery.scoring import ScoringEngine
import os


async def _check_lp_vault(snap) -> str | None:
    """Platform-vault LP check.

    Launchpads like StonkFun keep graduated LP in a platform vault instead of
    a recognised locker, so RugCheck reads 0% locked. If the LP mint is
    concentrated in a platform vault wallet, treat LP as locked. Returns a
    note for the report, or None.
    """
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    if not snap.pair_address:
        return None
    lp_mint = await resolve_lp_mint(snap.pair_address)
    if not lp_mint:
        return None
    known = load_known_vaults()
    check = await check_lp_platform_vault(lp_mint, rpc_url, known)
    if check.is_platform_vault and check.holder and check.holder not in known:
        known.add(check.holder)
        save_known_vaults(known)
    if not check.is_platform_vault:
        return None
    snap.safety.lp_locked_or_burned = True
    snap.safety.lp_locked_pct = 100.0
    detail = f"{check.holder_share_pct:.0f}% of LP" if check.holder_share_pct else "LP"
    accts = f", {check.token_account_count} token accounts" if check.token_account_count else ""
    cached = " (cached vault)" if check.cached else ""
    return (
        f"LP held by platform vault {(check.holder or "")[:8]}… — {detail} in "
        f"platform custody{accts}{cached}; treated as locked"
    )


async def enrich_safety(snap, goplus: GoPlusProvider | None) -> list[str]:
    """Attach contract-safety signals. Returns notes about coverage gaps."""
    notes: list[str] = []
    if snap.chain is Chain.SOLANA:
        try:
            import aiohttp

            async with aiohttp.ClientSession(trust_env=True) as s:
                async with s.get(
                    RUGCHECK_SUMMARY.format(mint=snap.token_address),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    if r.status == 200:
                        snap.safety = map_rugcheck_summary(await r.json())
                    else:
                        notes.append(f"RugCheck HTTP {r.status} — safety unknown")
        except Exception as e:  # noqa: BLE001 - fail-open
            notes.append(f"RugCheck unreachable ({type(e).__name__}) — safety unknown")
        # Jupiter holder data: DexScreener snapshots carry no holder info, so
        # without this the holder/distribution filter checks warn-and-pass forever.
        try:
            if not await enrich_jupiter_holders(snap):
                notes.append("Jupiter holder data unavailable — holder checks skipped")
        except Exception:  # noqa: BLE001 - fail-open
            notes.append("Jupiter holder lookup failed — holder checks skipped")
        # Platform-vault LP (e.g. StonkFun graduates): RugCheck reads 0% locked
        # when the platform custodies the LP instead of a known locker.
        try:
            if snap.safety.lp_locked_or_burned is False:
                vault_note = await _check_lp_vault(snap)
                if vault_note:
                    notes.append(vault_note)
        except Exception:  # noqa: BLE001 - fail-open
            pass
    elif snap.chain.is_evm:
        assert goplus is not None, "GoPlusProvider required for EVM safety enrichment"
        sec = await goplus.token_security(snap.chain, snap.token_address)
        if sec is None:
            notes.append("GoPlus has no coverage for this chain — safety unknown")
        else:
            snap.safety = sec.safety
            if sec.holder_count:
                snap.holder_count = sec.holder_count
            # Wallet concentration: drop AMM infrastructure (contract holders —
            # v2/v3 pools, the v4 PoolManager — plus locked supply) and keep
            # EOA-held supply, the actual dump risk. top10 doubles as the
            # concentration/bundle proxy.
            top, top10 = distribution_metrics(sec.holders, {snap.pair_address or ""})
            snap.top_holder_pct = top if top is not None else sec.top_holder_pct
            snap.top10_holder_pct = top10
            snap.dev_wallet_pct = sec.dev_wallet_pct
    return notes


def verdict(score_overall: float, results: dict, snap) -> tuple[str, str]:
    """(label, reason) — FENRIR's read, not financial advice."""
    s = snap.safety
    if s.honeypot:
        return "AVOID", "honeypot: sells are blocked"
    if (s.sell_tax_pct or 0) > 15 or (s.buy_tax_pct or 0) > 15:
        return "AVOID", f"punitive tax (buy {s.buy_tax_pct}% / sell {s.sell_tax_pct}%)"
    if s.mint_disabled is False:
        return "AVOID", "mint authority live — supply can be inflated"
    passed = [k for k, r in results.items() if r.passed]
    if score_overall >= 65 and passed:
        return "WORTH A LOOK", f"scores {score_overall:.0f}/100, fits FENRIR '{passed[0]}' profile"
    if score_overall >= 65:
        return "WORTH A LOOK", f"scores {score_overall:.0f}/100 but fits no FENRIR entry profile"
    if score_overall >= 40:
        return (
            "NEUTRAL",
            f"scores {score_overall:.0f}/100 — nothing disqualifying, nothing compelling",
        )
    return "WEAK", f"scores {score_overall:.0f}/100 — below FENRIR's bar"


def fmt_usd(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}k"
    return f"${v:,.0f}"


def report_text(sym, snap, results, breakdown, notes, tags=None) -> str:
    age = f"{snap.age_minutes:.0f}m" if snap.age_minutes else "?"
    lines = [
        f"{snap.symbol} ({snap.name}) — {snap.chain.value} · {snap.token_address[:10]}…",
        f"Price ${snap.price_usd:.8f} | MCap {fmt_usd(snap.market_cap_usd)} | "
        f"LP {fmt_usd(snap.liquidity_usd)} | Vol24h {fmt_usd(snap.volume_24h_usd)} | Age {age}",
        f"Buys/Sells 1h {snap.txns_1h_buys}/{snap.txns_1h_sells} | "
        f"24h {snap.txns_24h_buys}/{snap.txns_24h_sells} | "
        f"Holders {snap.holder_count if snap.holder_count is not None else '?'}",
        "",
        "FENRIR FILTERS",
    ]
    for name in (
        "low_cap_alpha",
        "mid_cap_momentum",
        "high_cap",
        "degen_launch",
        "volatility_breakout",
        "volume_surge",
        "momentum_transition",
        "curve_ignition",
        "flush_recovery",
        "graduation_watch",
    ):
        r = results[name]
        status = "PASS" if r.passed else "FAIL"
        detail = "" if r.passed else " — " + "; ".join(r.failures[:3])
        lines.append(f"  [{status}] {name}{detail}")
        for w in r.warnings[:2]:
            lines.append(f"         warn: {w}")
    if snap.bond_progress_pct is not None:
        inflow = (
            f" · inflow {snap.bond_inflow_sol:+.1f} SOL" if snap.bond_inflow_sol is not None else ""
        )
        remaining = (
            f" · {snap.bond_sol_remaining:.1f} SOL to graduation"
            if snap.bond_sol_remaining is not None
            else ""
        )
        lines.append(f"  🌊 Bonding curve {snap.bond_progress_pct:.0f}%{inflow}{remaining}")
    b = breakdown
    lines += [
        "",
        f"SCORE {b.overall:.1f}/100  "
        f"(momentum {b.momentum:.0f} · safety {b.safety:.0f} · liquidity {b.liquidity:.0f} · "
        f"holder {b.holder:.0f} · community {b.community:.0f} · risk {b.risk:.0f})",
        "",
        "SAFETY",
    ]
    s = snap.safety

    def yn(v):
        return "?" if v is None else ("yes" if v else "no")

    lines += [
        f"  honeypot: {yn(s.honeypot)} | buy tax: {s.buy_tax_pct if s.buy_tax_pct is not None else '?'}% | "
        f"sell tax: {s.sell_tax_pct if s.sell_tax_pct is not None else '?'}%",
        f"  mint disabled: {yn(s.mint_disabled)} | ownership renounced: {yn(s.ownership_renounced)} | "
        f"blacklist: {yn(s.blacklist_present)}",
    ]
    lp_line = f"  LP locked/burned: {yn(s.lp_locked_or_burned)}"
    if snap.migrated is False:
        lp_line = "  LP locked/burned: n/a (pre-migration)"
    elif s.lp_locked_pct is not None:
        lp_line += f" ({s.lp_locked_pct:.0f}%)"
    lines.append(lp_line)
    if s.risk_flags:
        lines.append(f"  risk flags: {', '.join(s.risk_flags[:5])}")
    for n in notes:
        lines.append(f"  note: {n}")
    label, reason = verdict(b.overall, results, snap)
    lines += ["", f"VERDICT: {label} — {reason}"]
    if tags is not None:
        if tags.matches:
            pb = ", ".join(f"{m.display_name} ({m.strength:.2f})" for m in tags.matches)
            lines += ["", f"PLAYBOOKS: {pb}"]
            if tags.confluent:
                lines.append(
                    f"  ⚡ confluent ({len(tags.sources)} strategies, "
                    f"combined {tags.combined_strength:.2f})"
                )
        else:
            lines += [
                "",
                f"PLAYBOOKS: none of the {len(PLAYBOOK_STRATEGY_IDS)} strategy playbooks fit",
            ]
    return "\n".join(lines)


async def amain() -> int:
    ap = argparse.ArgumentParser(description="FENRIR one-shot token evaluator")
    ap.add_argument("address", help="token contract/mint address")
    ap.add_argument("--chain", choices=[c.value for c in Chain], default=None)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--no-perceptor",
        action="store_true",
        help="skip the Perceptor on-chain scan for Robinhood tokens",
    )
    ap.add_argument(
        "--perceptor-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for a Perceptor verdict (default 300)",
    )
    args = ap.parse_args()

    chain = Chain(args.chain) if args.chain else None
    ds = DexScreenerProvider(timeout_seconds=15)
    gp = GoPlusProvider(timeout_seconds=10)
    try:
        snap = await ds.fetch_snapshot(args.address, chain=chain)
    finally:
        await ds.close()
    if snap is None:
        print("No DexScreener pair found for this address.")
        return 1

    notes = await enrich_safety(snap, gp)
    await gp.close()

    # Solana: live bonding-curve position for the graduation_watch filter.
    if snap.chain is Chain.SOLANA and not args.json:
        print("Bonding curve: reading on-chain state…", flush=True)
    if snap.chain is Chain.SOLANA:
        try:
            from fenrir.discovery.providers.pumpfun import annotate_bond_curve

            await annotate_bond_curve(snap)
        except Exception:  # noqa: BLE001 - fail-open
            pass

    # Robinhood safety net: when GoPlus has nothing, Perceptor's on-chain
    # forensics scan can still verify safety. Manual tool => wait for it.
    perceptor_info: dict | None = None
    if not args.no_perceptor and snap.chain is Chain.ROBINHOOD and snap.safety.is_empty:
        pp = PerceptorProvider()
        try:
            if not args.json:
                print(
                    "Perceptor: scanning on-chain history (up to "
                    f"{args.perceptor_timeout:.0f}s)…",
                    flush=True,
                )
            report = await pp.investigate(
                ROBINHOOD_CHAIN_ID,
                snap.token_address,
                timeout_seconds=args.perceptor_timeout,
                context=snapshot_context(snap),
            )
        finally:
            await pp.close()
        if report is not None:
            snap.safety = report.safety
            perceptor_info = {
                "band": report.band,
                "band_label": report.band_label,
                "headline": report.headline,
                "investigation_id": report.investigation_id,
            }
            notes.append(f"Perceptor verdict: {report.band_label} — {report.headline}")
        else:
            notes.append("Perceptor scan did not complete in time — safety unknown")

    engine = FilterEngine()
    results = {fn.value: engine.evaluate(snap, fn) for fn in FilterName}
    breakdown = ScoringEngine().score(snap)
    tags = PlaybookTagger().tag(snap)

    if args.json:
        print(
            json.dumps(
                {
                    "symbol": snap.symbol,
                    "name": snap.name,
                    "chain": snap.chain.value,
                    "address": snap.token_address,
                    "price_usd": snap.price_usd,
                    "market_cap_usd": snap.market_cap_usd,
                    "liquidity_usd": snap.liquidity_usd,
                    "volume_24h_usd": snap.volume_24h_usd,
                    "bond_progress_pct": snap.bond_progress_pct,
                    "bond_inflow_sol": snap.bond_inflow_sol,
                    "bond_sol_remaining": snap.bond_sol_remaining,
                    "filters": {
                        k: {"passed": r.passed, "failures": r.failures, "warnings": r.warnings}
                        for k, r in results.items()
                    },
                    "playbooks": tags.as_dict(),
                    "score": breakdown.as_dict(),
                    "verdict": verdict(breakdown.overall, results, snap),
                    "perceptor": perceptor_info,
                    "notes": notes,
                },
                indent=1,
            )
        )

    else:
        print(report_text(args.address, snap, results, breakdown, notes, tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
