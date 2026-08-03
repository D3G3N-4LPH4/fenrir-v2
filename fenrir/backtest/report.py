#!/usr/bin/env python3
"""
FENRIR - Backtest report formatting (Phase 6.3)

Turns a ``PortfolioResult`` into a compact, readable text report: a per-strategy metrics
table, the combined line, and the confluent-vs-non-confluent comparison that answers
whether multi-strategy agreement paid off. Pure string building — no I/O.
"""

from __future__ import annotations

from fenrir.backtest.models import BacktestMetrics
from fenrir.backtest.portfolio import PortfolioResult

_HEADER = f"{'strategy':<18}{'trades':>7}{'win%':>7}{'exp%':>8}{'PF':>7}{'sharpe':>8}{'maxDD%':>8}"


def _row(label: str, m: BacktestMetrics) -> str:
    return (
        f"{label:<18}{m.trades:>7}{m.win_rate * 100:>6.1f}%"
        f"{m.expectancy_pct:>8.2f}{m.profit_factor:>7.2f}{m.sharpe:>8.2f}{m.max_drawdown_pct:>8.2f}"
    )


def format_report(result: PortfolioResult) -> str:
    """Render a portfolio backtest result as a text report."""
    lines: list[str] = []
    lines.append("=" * len(_HEADER))
    lines.append("FENRIR BACKTEST REPORT")
    lines.append("=" * len(_HEADER))
    lines.append(_HEADER)
    lines.append("-" * len(_HEADER))

    for sid in sorted(result.per_strategy):
        res = result.per_strategy[sid]
        lines.append(_row(sid, res.metrics))
        lines.append(f"  {'':<16}entered {res.samples_entered}/{res.samples_evaluated} evaluated")

    lines.append("-" * len(_HEADER))
    lines.append(_row("COMBINED", result.combined_metrics))

    # Confluence comparison — the Phase 5/6 payoff question.
    lines.append("")
    lines.append("Confluence (multi-strategy agreement):")
    lines.append(f"  confluent tokens: {len(result.confluent_tokens)}")
    lines.append(_row("  confluent", result.confluent_metrics))
    lines.append(_row("  non-confluent", result.non_confluent_metrics))
    if result.confluent_metrics.trades and result.non_confluent_metrics.trades:
        edge = result.confluent_metrics.expectancy_pct - result.non_confluent_metrics.expectancy_pct
        verdict = "helped" if edge > 0 else "did not help"
        lines.append(f"  confluence edge (expectancy): {edge:+.2f}% — {verdict}")
    else:
        lines.append("  confluence edge: n/a (need trades in both buckets)")

    lines.append("=" * len(_HEADER))
    return "\n".join(lines)
