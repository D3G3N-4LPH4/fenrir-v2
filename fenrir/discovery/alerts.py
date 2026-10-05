"""Telegram alert formatting for scout candidates.

Produces a scannable, Markdown-formatted alert block per candidate. The
contract address is wrapped in a code span so it is tap-to-copy on mobile.
All user-derived text (symbol, name, source) is escaped for Telegram's
legacy Markdown parse mode; the address itself is hex and needs no escaping.
"""

from __future__ import annotations

from typing import Any


def fmt_usd(v: Any) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "?"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v / 1_000:.1f}k"
    return f"${v:,.0f}"


def escape_md(text: Any) -> str:
    """Escape Telegram legacy-Markdown special chars in free text."""
    s = str(text or "")
    for ch in ("\\", "_", "*", "[", "]", "`"):
        s = s.replace(ch, "\\" + ch)
    return s


def _score(cand: dict) -> float:
    s = cand.get("score", 0)
    if isinstance(s, dict):
        s = s.get("overall", 0)
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _age_str(age_min: Any) -> str:
    try:
        m = float(age_min)
    except (TypeError, ValueError):
        return "?"
    if m >= 120:
        return f"{m / 60:.1f}h old"
    return f"{m:.0f}m old"


def _fmt_price_precise(v: Any) -> str:
    """Compact price formatting that survives sub-cent memecoin prices."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "?"
    if v <= 0:
        return "$0"
    if v >= 1:
        return f"${v:,.4f}".rstrip("0").rstrip(".")
    # small prices: 3 significant figures
    import math

    digits = max(0, 3 - int(math.floor(math.log10(v))) - 1)
    return f"${v:.{digits}f}"


def build_exit_ladder(price_usd: Any) -> list[tuple[int, str]] | None:
    """Take-profit ladder from the alert-time price.

    Gate-tracker review (2026-10-03): 36% of alerts peaked >=+50% after
    clearance but holding killed the basket (median -96%). Every alert
    carries its exits so the manual trader has the ladder at entry time.
    """
    try:
        p = float(price_usd)
    except (TypeError, ValueError):
        return None
    if p <= 0:
        return None
    return [(pct, _fmt_price_precise(p * (1 + pct / 100.0))) for pct in (25, 50, 100)]


def _tier_line(tier: Any) -> str | None:
    t = str(tier or "")
    if t == "ignition":
        return "\u26a1 early ignition \u2014 pre-momentum entry"
    if t == "late":
        return "\u231b late entry \u2014 move mostly done, logged not alerted"
    return None


def format_scout_alert(cand: dict) -> str:
    """Render one candidate dict (as emitted by scout.py/channel_poll.py) as a
    Telegram Markdown alert."""
    symbol = escape_md(cand.get("symbol") or "?")
    name = escape_md(cand.get("name") or "")
    chain = str(cand.get("chain") or "?").title()
    source = str(cand.get("source") or "")
    address = cand.get("address") or ""
    score = _score(cand)
    filters = ", ".join(cand.get("passed_filters") or [])

    lines = []
    is_misfit = bool(cand.get("misfit"))
    header = f"\U0001f3af *{symbol}*"
    if name and name.lower() != symbol.lower().replace("\\", ""):
        header += f" \u2014 {name}"
    if is_misfit:
        header += " \u2014 gate-rejected, tracked"
    lines.append(header)

    subtitle = chain
    if source and not source.startswith("boosted"):
        subtitle += f" \u00b7 via {escape_md(source)}"
    if cand.get("dex_paid"):
        subtitle += " \u00b7 \U0001f4b0 DEX paid"
    lines.append(subtitle)
    lines.append("")

    score_line = f"\u2b50 {score:.1f}/100"
    if filters:
        score_line += f" \u00b7 {escape_md(filters)}"
    lines.append(score_line)
    tier = _tier_line(cand.get("entry_tier"))
    if tier:
        lines.append(tier)

    pb = cand.get("playbooks") or {}
    entries = pb.get("playbooks") or []
    if entries:
        parts = [
            f"{escape_md(e.get('display_name') or e.get('strategy_id') or '?')} "
            f"({float(e.get('strength', 0)):.2f})"
            for e in entries
        ]
        # 2026-10-03: the ⚡confluent conviction marker is gone. Gate-tracker
        # review showed confluent playbooks hit 17% vs 28% without confluence —
        # it reads as conviction but predicts nothing. Playbooks still list
        # for context; they no longer imply an edge.
        lines.append("\U0001f4d6 " + ", ".join(parts))
    cc = cand.get("caller_confluence")
    if cc:
        lines.append("\U0001f465 caller confluence: " + " + ".join(escape_md(str(x)) for x in cc))
    wb = cand.get("wallet_buys")
    if wb:
        labels = [str(w.get("label") or "?") for w in wb]
        wline = "\U0001f45b wallet buy: " + " + ".join(escape_md(x) for x in labels)
        spent = [w.get("sol_spent") for w in wb if w.get("sol_spent")]
        if spent:
            try:
                wline += f" (~{sum(float(s) for s in spent):.2f} SOL)"
            except (TypeError, ValueError):
                pass
        lines.append(wline)
    lines.append("")

    lines.append(
        f"\U0001f4b0 mcap {fmt_usd(cand.get('market_cap_usd'))} \u00b7 "
        f"liq {fmt_usd(cand.get('liquidity_usd'))} \u00b7 "
        f"24h vol {fmt_usd(cand.get('volume_24h_usd'))}"
    )
    buys, sells = cand.get("buys_1h"), cand.get("sells_1h")
    if buys is None and sells is None:
        flow = "1h flow n/a"
    else:
        flow = f"1h {buys if buys is not None else '?'} buys / {sells if sells is not None else '?'} sells"
    move = cand.get("price_change_1h_pct")
    if move is not None:
        try:
            flow += f" \u00b7 {float(move):+.0f}% 1h"
        except (TypeError, ValueError):
            pass
    lines.append(f"\U0001f4ca {flow}")
    lines.append(f"\u23f1\ufe0f {_age_str(cand.get('age_minutes'))}")
    # ATH distance (2026-10-04): the best context for whether a runner is
    # extended or basing. Fail-open — absent when candles were unavailable.
    ath = cand.get("ath") or {}
    if ath.get("ath_price"):
        try:
            drop = float(ath["drop_pct"])
            hrs = float(ath.get("hours_ago", 0))
            age_s = f"{hrs:.0f}h" if hrs < 48 else f"{hrs / 24:.1f}d"
            lines.append(
                f"\U0001f4c9 ATH ${float(ath['ath_price']):.6g} " f"({drop:+.0f}% / {age_s} ago)"
            )
        except (TypeError, ValueError):
            pass
    # Security block (2026-10-04, Phanes-style): concentration + bundle
    # exposure inline instead of buried in the score.
    sec_bits = []
    top10 = cand.get("top10_holder_pct")
    if top10 is not None:
        try:
            sec_bits.append(f"Top 10 {float(top10):.0f}%")
        except (TypeError, ValueError):
            pass
    holders = cand.get("holder_count")
    if holders is not None:
        sec_bits.append(f"{int(holders):,} holders")
    bundle = cand.get("largest_cluster_pct")
    if bundle is not None:
        try:
            sec_bits.append(f"insider clust {float(bundle):.1f}%")
        except (TypeError, ValueError):
            pass
    if sec_bits:
        lines.append("\U0001f512 " + " \u00b7 ".join(sec_bits))
    # Socials row (2026-10-04): X / TG / web from DexScreener's info object.
    socials = []
    if cand.get("twitter"):
        socials.append(f"[X]({cand['twitter']})")
    if cand.get("telegram"):
        socials.append(f"[TG]({cand['telegram']})")
    if cand.get("website"):
        socials.append(f"[Web]({cand['website']})")
    if socials:
        lines.append("\U0001f517 " + " \u00b7 ".join(socials))
    if not is_misfit:
        ladder = build_exit_ladder(cand.get("price_usd"))
        if ladder:
            targets = " \u00b7 ".join(f"+{pct}% {price}" for pct, price in ladder)
            lines.append(f"\U0001f3af Exits \u2014 {targets}")
            lines.append("move stop to entry at +25%")
    # Bonding-curve position for pre-graduation pump.fun tokens.
    bond = cand.get("bond_progress_pct")
    if bond is not None:
        try:
            curve_line = f"\U0001f30a Bonding curve {float(bond):.0f}%"
            remaining = cand.get("bond_sol_remaining")
            if remaining is not None:
                curve_line += f" \u00b7 {float(remaining):.0f} SOL to graduation"
            inflow = cand.get("bond_inflow_sol")
            if inflow is not None:
                curve_line += f" (\u25b2{float(inflow):.1f} SOL)"
            lines.append(curve_line)
        except (TypeError, ValueError):
            pass
    lines.append("")

    if address:
        lines.append("\U0001f4cb Contract \u2014 tap to copy:")
        lines.append(f"`{address}`")
        lines.append("")
        lines.append(
            f"\U0001f517 [DexScreener](https://dexscreener.com/"
            f"{str(cand.get('chain') or '').lower()}/{address})"
        )

    if cand.get("safety_unknown"):
        lines.append("")
        lines.append("\u26a0\ufe0f safety not verifiable on this chain")

    safety = cand.get("safety") or {}
    if safety.get("status") == "complete" and safety.get("band_label"):
        lines.append("")
        pline = f"\U0001f50d Safety: *{escape_md(str(safety['band_label']))}*"
        if safety.get("headline"):
            pline += f" \u2014 {escape_md(str(safety['headline']))}"
        lines.append(pline)

    return "\n".join(lines).strip() + "\n"
