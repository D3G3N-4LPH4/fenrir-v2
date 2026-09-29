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
    header = f"\U0001f3af *{symbol}*"
    if name and name.lower() != symbol.lower().replace("\\", ""):
        header += f" \u2014 {name}"
    lines.append(header)

    subtitle = chain
    if source and not source.startswith("boosted"):
        subtitle += f" \u00b7 via {escape_md(source)}"
    lines.append(subtitle)
    lines.append("")

    score_line = f"\u2b50 {score:.1f}/100"
    if filters:
        score_line += f" \u00b7 {escape_md(filters)}"
    lines.append(score_line)

    pb = cand.get("playbooks") or {}
    entries = pb.get("playbooks") or []
    if entries:
        parts = [
            f"{escape_md(e.get('display_name') or e.get('strategy_id') or '?')} "
            f"({float(e.get('strength', 0)):.2f})"
            for e in entries
        ]
        pb_line = "\U0001f4d6 " + ", ".join(parts)
        if pb.get("confluent"):
            pb_line += " \u26a1confluent"
        lines.append(pb_line)
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

    perc = cand.get("perceptor") or {}
    if perc.get("status") == "complete" and perc.get("band_label"):
        lines.append("")
        pline = f"\U0001f50d Perceptor: *{escape_md(str(perc['band_label']))}*"
        if perc.get("headline"):
            pline += f" \u2014 {escape_md(str(perc['headline']))}"
        lines.append(pline)
    elif perc.get("investigation_id"):
        lines.append("")
        lines.append("\U0001f50d Perceptor on-chain scan running\u2026")

    return "\n".join(lines).strip() + "\n"


_BAND_EMOJI = {"low": "\U0001f7e2", "medium": "\U0001f7e1", "high": "\U0001f534"}


def format_perceptor_verdict(address: str, context: dict | None, report) -> str:
    """Render a landed Perceptor verdict as a Telegram Markdown follow-up.

    ``context`` is the token dict stored with the investigation
    (symbol/name/chain/dexscreener); ``report`` is a PerceptorReport.
    """
    ctx = context or {}
    symbol = escape_md(ctx.get("symbol") or "?")
    name = escape_md(ctx.get("name") or "")
    chain = str(ctx.get("chain") or "robinhood").title()
    dexscreener = ctx.get("dexscreener") or (f"https://dexscreener.com/robinhood/{address}")
    band = str(report.band or "").lower()
    emoji = _BAND_EMOJI.get(band, "\U0001f52c")  # microscope fallback
    band_label = escape_md(report.band_label or report.band or "unknown")

    lines = []
    header = f"{emoji} *{symbol}*"
    if name and name.lower() != symbol.lower().replace("\\", ""):
        header += f" \u2014 {name}"
    lines.append(header)
    lines.append(f"Perceptor verdict \u00b7 {escape_md(chain)}")
    lines.append("")

    verdict_line = f"*{band_label}*"
    if report.headline:
        verdict_line += f" \u2014 {escape_md(report.headline)}"
    lines.append(verdict_line)
    lines.append("")

    s = report.safety
    facts = []
    if s.lp_locked_or_burned:
        facts.append(
            "\U0001f512 LP locked"
            + (f" ({s.lp_locked_pct:.0f}%)" if s.lp_locked_pct is not None else "")
        )
    elif s.lp_locked_pct is not None:
        facts.append(f"\U0001f512 LP locked {s.lp_locked_pct:.0f}%")
    if s.buy_tax_pct is not None or s.sell_tax_pct is not None:
        facts.append(f"tax {s.buy_tax_pct or 0:.0f}%/{s.sell_tax_pct or 0:.0f}% buy/sell")
    if s.honeypot is False:
        facts.append("\u2705 selling works \u2014 not a honeypot")
    elif s.honeypot is True:
        facts.append("\U0001f6d1 honeypot \u2014 selling fails")
    if s.mint_disabled is True:
        facts.append("mint disabled")
    elif s.mint_disabled is False:
        facts.append("\u26a0\ufe0f mint live")
    if s.blacklist_present:
        facts.append("\u26a0\ufe0f blacklist enabled")
    if s.ownership_renounced is True:
        facts.append("ownership renounced")
    if facts:
        lines.append(" \u00b7 ".join(facts))
        lines.append("")

    if s.risk_flags:
        seen = {str(report.headline or "").strip().lower()}
        flags = [
            f
            for f in s.risk_flags
            if str(f).strip().lower() not in seen and seen.add(str(f).strip().lower()) is None
        ][:4]
        if flags:
            lines.append("\u26a0\ufe0f " + escape_md("; ".join(flags)))
            lines.append("")

    lines.append("\U0001f4cb Contract \u2014 tap to copy:")
    lines.append(f"`{address}`")
    lines.append("")
    links = f"\U0001f517 [DexScreener]({dexscreener})"
    if report.investigation_id:
        links += (
            " \u00b7 [Perceptor report]"
            f"(https://www.perceptor.info/?investigation={report.investigation_id})"
        )
    lines.append(links)
    return "\n".join(lines).strip() + "\n"
