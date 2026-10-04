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
from fenrir.discovery.lp_vault import VaultCheck, check_pool_lp_vault
from fenrir.discovery.models import Chain
from fenrir.discovery.lp_lock_v4 import inspect_v4_lp_lock
from fenrir.discovery.bundle_check import (
    BundleDeployerReport,
    INCONCLUSIVE_TTL_S,
    check_bundle_and_deployer,
    get_cached_bundle_report,
    save_cached_bundle_report,
)
from fenrir.discovery.solana_forensics import (  # noqa: E402
    INCONCLUSIVE_TTL_S as SOLANA_FORENSICS_INCONCLUSIVE_TTL_S,
    SolanaForensicsReport,
    check_solana_distribution,
    get_cached_forensics,
    save_cached_forensics,
)
from fenrir.discovery.playbooks import PLAYBOOK_STRATEGY_IDS, PlaybookTagger
from fenrir.discovery.providers.dexscreener import DexScreenerProvider
from fenrir.discovery.providers.goplus import GoPlusProvider, distribution_metrics
from fenrir.discovery.providers.robinhood_safety import (
    RobinhoodSafetyProvider,
    enrich_robinhood_safety,
)
from fenrir.discovery.scoring import ScoringEngine
import os

# Outer deadline for the whole platform-vault LP walk (Raydium resolve + RPCs).
# Per-RPC timeouts live inside check_pool_lp_vault; this is the backstop.
LP_VAULT_CHECK_TIMEOUT_SECONDS = 45.0

# Deadlines for the heavy Robinhood-chain safety legs. Each gets its own
# backstop inside the 90s per-token budget (they run concurrently).
V4_LOCK_TIMEOUT_SECONDS = 40.0
BUNDLE_CHECK_TIMEOUT_SECONDS = 60.0
# Solana distribution forensics (direct RPC holder read): cheap enough for a
# tight deadline; the volatility_breakout filter fails closed without it.
SOLANA_FORENSICS_TIMEOUT_SECONDS = 25.0


async def _solana_forensics_notes(snap) -> list[str]:
    """Direct-RPC holder distribution for a Solana snapshot.

    Jupiter's holder enrichment misses many young tokens (the 2026-10-01
    volatility_breakout blowups all cleared with "Top-10 holders %
    unavailable"). This fills snap.top_holder_pct / snap.top10_holder_pct
    from chain data so the concentration caps can bite. Cached 24h per
    token; inconclusive results back off 1h. Mutates snap holders.
    """
    notes: list[str] = []
    if snap.chain is not Chain.SOLANA:
        return notes
    try:
        report: SolanaForensicsReport | None
        cached = get_cached_forensics(snap.token_address)
        if cached is not None:
            if cached.get("_inconclusive"):
                notes.append("holder forensics: recently inconclusive — skipping re-check")
                return notes
            report = SolanaForensicsReport.from_dict(cached)
            notes.append(f"holder forensics: cached — {report.detail}")
        else:
            try:
                report = await asyncio.wait_for(
                    check_solana_distribution(snap.token_address),
                    timeout=SOLANA_FORENSICS_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                report = None
            if report is not None:
                save_cached_forensics(snap.token_address, report.as_dict())
                notes.append(f"holder forensics: {report.detail}")
            else:
                # Back off: don't burn RPC budget on this token every tick.
                save_cached_forensics(
                    snap.token_address,
                    {"_inconclusive": True},
                    ttl_seconds=SOLANA_FORENSICS_INCONCLUSIVE_TTL_S,
                )
                notes.append("holder forensics inconclusive — distribution unknown")
        if report is not None:
            # Fill what Jupiter missed; never override provider data we have.
            if snap.top_holder_pct is None:
                snap.top_holder_pct = report.top_holder_pct
            if snap.top10_holder_pct is None:
                snap.top10_holder_pct = report.top10_holder_pct
    except Exception:  # noqa: BLE001 - fail-open
        notes.append("holder forensics failed — distribution unknown")
    return notes


async def _v4_lock_notes(snap) -> list[str]:
    """Robinhood v4 LP-lock verification (ATM lesson, 2026-09-30).

    GoPlus rarely reports LP lock state on this chain, so an unknown lock
    used to sail through the young-coin filters straight into an LP pull.
    Verify on-chain via the v4 PositionManager: burned/locker-held position
    NFTs = locked; EOA-held = pullable. Mutates snap.safety.
    """
    notes: list[str] = []
    if not (
        snap.chain is Chain.ROBINHOOD
        and snap.safety.lp_locked_or_burned is None
        and snap.pair_address
        and len(snap.pair_address) == 66
    ):
        return notes
    try:
        v4lock = await asyncio.wait_for(
            inspect_v4_lp_lock(snap.pair_address, age_minutes=snap.age_minutes),
            timeout=V4_LOCK_TIMEOUT_SECONDS,
        )
    except Exception:  # noqa: BLE001 - fail-open (includes TimeoutError)
        return notes
    if v4lock.locked is not None:
        snap.safety.lp_locked_or_burned = v4lock.locked
        notes.append(f"v4 LP lock: {v4lock.detail}")
    return notes


async def _bundle_notes(snap) -> list[str]:
    """Bundle + deployer-cluster check (2026-09-30, Bubblemaps bands).

    Coordinated launch buys and deployer-linked supply, verified on-chain.
    Cached 24h per token. Mutates snap.safety / snap.bundle_* fields.
    """
    notes: list[str] = []
    if not (snap.chain is Chain.ROBINHOOD and snap.pair_address and len(snap.pair_address) == 66):
        return notes
    try:
        report: BundleDeployerReport | None
        cached = get_cached_bundle_report(snap.token_address)
        if cached is not None:
            if cached.get("_inconclusive"):
                notes.append("bundle/deployer: recently inconclusive — skipping re-check")
                return notes
            report = BundleDeployerReport.from_dict(cached)
            notes.append(f"bundle/deployer: cached — {report.detail}")
        else:
            try:
                report = await asyncio.wait_for(
                    check_bundle_and_deployer(
                        snap.token_address,
                        snap.pair_address,
                        age_minutes=snap.age_minutes,
                    ),
                    timeout=BUNDLE_CHECK_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                report = None
            if report is not None:
                save_cached_bundle_report(snap.token_address, report.as_dict())
                notes.append(f"bundle/deployer: {report.detail}")
            else:
                # Back off: don't burn a 60s RPC walk on this token every tick.
                save_cached_bundle_report(
                    snap.token_address,
                    {"_inconclusive": True},
                    ttl_seconds=INCONCLUSIVE_TTL_S,
                )
                notes.append("bundle/deployer check inconclusive — metrics unknown")
        if report is not None:
            s = snap.safety
            s.bundled_supply_pct = report.bundled_supply_pct
            s.largest_cluster_pct = report.largest_cluster_pct
            s.cluster_count = report.cluster_count
            s.deployer_cluster_pct = report.deployer_cluster_pct
            s.deployer_holding_pct = report.deployer_holding_pct
            s.deployer_distributed_wallets = report.deployer_distributed_wallets
            s.deployer_funder_is_serial_launcher = report.deployer_funder_is_serial_launcher
            if snap.bundle_pct is None:
                snap.bundle_pct = report.bundled_supply_pct
            if snap.insider_pct is None:
                snap.insider_pct = report.deployer_cluster_pct
            snap.bundle_report = report.as_dict()
    except Exception:  # noqa: BLE001 - fail-open
        notes.append("bundle/deployer check failed — metrics unknown")
    return notes


async def _check_lp_vault(snap) -> VaultCheck | None:
    """Platform-vault LP check.

    Launchpads like StonkFun keep graduated LP in a platform vault instead of
    a recognised locker, so RugCheck reads 0% locked. Returns the VaultCheck
    (or None when it can't run); the caller applies it to ``snap.safety``
    only when RugCheck left LP as unlocked. Pure: never mutates the snapshot.
    """
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    if not snap.pair_address:
        return None
    try:
        # Outer deadline on top of the per-RPC timeouts inside: the vault walk
        # must never eat a large slice of the token's 90s budget.
        check = await asyncio.wait_for(
            check_pool_lp_vault(snap.pair_address, rpc_url),
            timeout=LP_VAULT_CHECK_TIMEOUT_SECONDS,
        )
    except Exception:  # noqa: BLE001 - fail-open (includes TimeoutError)
        return None
    return check


def _vault_note(check: VaultCheck) -> str:
    detail = f"{check.holder_share_pct:.0f}% of LP" if check.holder_share_pct else "LP"
    accts = f", {check.token_account_count} token accounts" if check.token_account_count else ""
    cached = " (cached vault)" if check.cached else ""
    return (
        f"LP held by platform vault {(check.holder or "")[:8]}… — {detail} in "
        f"platform custody{accts}{cached}; treated as locked"
    )


async def _enrich_rugcheck(snap) -> list[str]:
    """RugCheck safety summary for a Solana snapshot. Mutates snap.safety."""
    notes: list[str] = []
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
    return notes


async def _enrich_jupiter_notes(snap) -> list[str]:
    """Jupiter holder data for a Solana snapshot. Mutates snap holders."""
    notes: list[str] = []
    try:
        # DexScreener snapshots carry no holder info, so without this the
        # holder/distribution filter checks warn-and-pass forever.
        if not await enrich_jupiter_holders(snap):
            notes.append("Jupiter holder data unavailable — holder checks skipped")
    except Exception:  # noqa: BLE001 - fail-open
        notes.append("Jupiter holder lookup failed — holder checks skipped")
    return notes


async def enrich_safety(snap, goplus: GoPlusProvider | None) -> list[str]:
    """Attach contract-safety signals. Returns notes about coverage gaps."""
    notes: list[str] = []
    if snap.chain is Chain.SOLANA:
        # RugCheck, Jupiter holders, the platform-vault LP walk, and the
        # direct-RPC holder forensics are independent — run them concurrently
        # instead of sequentially.
        # The vault result is applied after RugCheck: it only matters when
        # RugCheck left LP as unlocked (same condition as before).
        rug_task = asyncio.create_task(_enrich_rugcheck(snap))
        jup_task = asyncio.create_task(_enrich_jupiter_notes(snap))
        vault_task = asyncio.create_task(_check_lp_vault(snap))
        forensics_task = asyncio.create_task(_solana_forensics_notes(snap))
        notes.extend(await rug_task)
        notes.extend(await jup_task)
        notes.extend(await forensics_task)
        vault_check = await vault_task
        if snap.safety.lp_locked_or_burned is False and vault_check is not None:
            if vault_check.burned:
                snap.safety.lp_locked_or_burned = True
                snap.safety.lp_locked_pct = 100.0
                notes.append("LP supply fully burned (0 outstanding) — treated as locked")
            elif vault_check.is_platform_vault:
                snap.safety.lp_locked_or_burned = True
                snap.safety.lp_locked_pct = 100.0
                notes.append(_vault_note(vault_check))
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
        # Robinhood v4 LP-lock verification + bundle/deployer check are
        # independent — run them concurrently instead of sequentially.
        v4_task = asyncio.create_task(_v4_lock_notes(snap))
        bundle_task = asyncio.create_task(_bundle_notes(snap))
        notes.extend(await v4_task)
        notes.extend(await bundle_task)
    # Chart patterns (both chains): hourly GeckoTerminal candles -> tags.
    # Independent of the safety legs above — runs for every snapshot.
    notes.extend(await _chart_pattern_notes(snap))
    return notes


async def _chart_pattern_notes(snap) -> list[str]:
    """Detect chart patterns from hourly candles; attach tags to the snapshot.

    Bullish patterns join the playbook tags as entry confluence. Bearish
    patterns are caution notes only — they never auto-fail a gate (hit rates
    get quantified in the user_cases loop first).
    """
    notes: list[str] = []
    try:
        from fenrir.discovery.chart_patterns import attach_chart_patterns

        pats = await asyncio.wait_for(attach_chart_patterns(snap), timeout=30.0)
    except Exception:  # noqa: BLE001 - fail-open
        return notes
    for p in pats:
        if p.direction == "bearish":
            notes.append(f"chart caution: {p.display_name} ({p.strength:.0%}) — {p.rationale}")
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
    br = snap.bundle_report or {}
    if br.get("bundled_supply_pct") is not None or br.get("deployer_address"):
        lines += ["", "BUNDLE / DEPLOYER"]

        def pct(v):
            return "?" if v is None else f"{v:.1f}%"

        lines.append(
            f"  bundled supply: {pct(br.get('bundled_supply_pct'))} | "
            f"clusters: {br.get('cluster_count', 0)} | "
            f"largest cluster: {pct(br.get('largest_cluster_pct'))}"
        )
        if br.get("launch_window_partial"):
            lines.append(
                "  launch window partial: metrics may understate (Initialize predates scan range)"
            )
        dep = br.get("deployer_address")
        if dep:
            serial = br.get("deployer_funder_is_serial_launcher")
            lines.append(
                f"  deployer: {dep[:10]}… | holding: {pct(br.get('deployer_holding_pct'))} | "
                f"distributed to {br.get('deployer_distributed_wallets', '?')} wallets | "
                f"serial launcher: {yn(serial)}"
            )
            lines.append(f"  deployer-linked supply: {pct(br.get('deployer_cluster_pct'))}")
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
        help="skip the on-chain safety read for Robinhood tokens (legacy name)",
    )
    ap.add_argument(
        "--perceptor-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for the on-chain safety read (default 300; legacy name)",
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

    if (
        not args.json
        and snap.chain is Chain.ROBINHOOD
        and snap.pair_address
        and len(snap.pair_address) == 66
    ):
        print("Bundle/deployer: checking on-chain distribution (up to ~60s)…", flush=True)
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

    # Robinhood safety net: when GoPlus has nothing, the local on-chain
    # safety reader can still verify safety. Manual tool => wait for it.
    safety_info: dict | None = None
    if not args.no_perceptor and snap.chain is Chain.ROBINHOOD and snap.safety.is_empty:
        local = RobinhoodSafetyProvider()
        try:
            if not args.json:
                print(
                    "Robinhood safety: reading on-chain (up to " f"{args.perceptor_timeout:.0f}s)…",
                    flush=True,
                )
            report = await asyncio.wait_for(
                enrich_robinhood_safety(snap, local),
                timeout=args.perceptor_timeout,
            )
        except Exception:  # noqa: BLE001 - fail-open (includes TimeoutError)
            report = None
        if report is not None:
            safety_info = {
                "band": report.band,
                "band_label": report.band_label,
                "headline": report.headline,
                "investigation_id": report.investigation_id,
            }
            notes.append(f"Safety verdict: {report.band_label} — {report.headline}")
        else:
            notes.append("On-chain safety read did not complete — safety unknown")

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
                    "pair_address": snap.pair_address,
                    "dex_id": snap.dex_id,
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
                    "safety_report": safety_info,
                    "bundle": snap.bundle_report,
                    "notes": notes,
                    "flow_1h": {"buys": snap.txns_1h_buys, "sells": snap.txns_1h_sells},
                    "flow_24h": {"buys": snap.txns_24h_buys, "sells": snap.txns_24h_sells},
                    "safety": {
                        "honeypot": snap.safety.honeypot,
                        "buy_tax_pct": snap.safety.buy_tax_pct,
                        "sell_tax_pct": snap.safety.sell_tax_pct,
                        "mint_disabled": snap.safety.mint_disabled,
                        "blacklist_present": snap.safety.blacklist_present,
                        "lp_locked_pct": snap.safety.lp_locked_pct,
                    },
                    "dexscreener_url": (
                        f"https://dexscreener.com/{snap.chain.value}/{snap.token_address}"
                    ),
                },
                indent=1,
            )
        )

    else:
        print(report_text(args.address, snap, results, breakdown, notes, tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
