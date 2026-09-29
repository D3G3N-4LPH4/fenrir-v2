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

    return "\n".join(lines).strip() + "\n"
