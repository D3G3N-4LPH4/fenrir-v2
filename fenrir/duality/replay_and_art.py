"""
replay_and_art.py — FENRIR v2 Incident Replay + OpenPipe ART Export
====================================================================
Two responsibilities:

1. IncidentReplayer — given a live trade's event record (written by the
   live ExecutionRecorder), reconstructs the exact sequence of inputs
   Claude saw and re-runs the strategy through backtest mode. Tells you
   exactly what went wrong and why.

2. ARTExporter — converts labeled BacktestRecorder events into the
   OpenPipe ART format for RL-based fine-tuning. Produces:
     - preference pairs (good decision vs bad decision on similar context)
     - reward signals from outcome labels
     - full prompt/response jsonl for supervised fine-tuning baseline

Usage:
    # After a backtest run:
    exporter = ARTExporter(recorder)
    exporter.export("./art_data/run_2025_01/")

    # After a live incident:
    replayer = IncidentReplayer(live_event_log="./logs/live_events.jsonl")
    await replayer.replay_incident(incident_id="...", feed=hist_feed, router=sim_router)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .backtest_impls import BacktestRecorder, SimulatedRouter, HistoricalDataFeed

logger = logging.getLogger("fenrir.art")


# ---------------------------------------------------------------------------
# ART data structures
# ---------------------------------------------------------------------------

@dataclass
class ARTSample:
    """
    A single training sample for OpenPipe ART.
    Contains the full prompt/response pair + a reward signal.
    """
    sample_id:    str
    mint:         str
    prompt:       str         # the exact prompt Claude received
    response:     str         # Claude's exact response
    action:       str         # BUY / SELL / HOLD / EXIT
    reward:       float       # continuous reward: -1.0 to +1.0
    label:        str         # GOOD_BUY / BAD_BUY / etc.
    pnl_pct:      float
    hold_secs:    float
    model:        str
    timestamp:    str

    def to_openrouter_format(self) -> dict:
        """Format expected by OpenPipe ART fine-tuning API."""
        return {
            "messages": [
                {"role": "user",      "content": self.prompt},
                {"role": "assistant", "content": self.response},
            ],
            "reward":     self.reward,
            "metadata": {
                "sample_id":  self.sample_id,
                "mint":       self.mint,
                "action":     self.action,
                "label":      self.label,
                "pnl_pct":    self.pnl_pct,
                "hold_secs":  self.hold_secs,
                "model":      self.model,
                "timestamp":  self.timestamp,
            },
        }


@dataclass
class PreferencePair:
    """
    A preference pair for DPO (Direct Preference Optimization) training.
    'chosen' is the response that led to a better outcome.
    'rejected' is the response that led to a worse outcome.
    Both must be on similar context (same token, similar market conditions).
    """
    sample_id:      str
    prompt:         str
    chosen:         str    # response from GOOD_* labeled decision
    rejected:       str    # response from BAD_* labeled decision
    chosen_reward:  float
    rejected_reward: float
    context_summary: str   # brief description of market conditions

    def to_openrouter_format(self) -> dict:
        return {
            "messages": [{"role": "user", "content": self.prompt}],
            "chosen":   [{"role": "assistant", "content": self.chosen}],
            "rejected": [{"role": "assistant", "content": self.rejected}],
            "metadata": {
                "sample_id":      self.sample_id,
                "chosen_reward":  self.chosen_reward,
                "rejected_reward": self.rejected_reward,
                "context":        self.context_summary,
            },
        }


# ---------------------------------------------------------------------------
# ART Exporter
# ---------------------------------------------------------------------------

# Reward map — converts string labels to continuous reward signal
REWARD_MAP = {
    "GOOD_BUY":       +1.0,
    "OK_BUY":         +0.4,
    "BREAK_EVEN_BUY": +0.1,
    "BAD_BUY":        -1.0,
    "GOOD_SELL":      +0.8,
    "GOOD_HOLD":      +0.6,
    "OK_HOLD":        +0.2,
    "BAD_HOLD":       -0.7,
    "LOSS_SELL":      +0.3,   # sold at a loss but at least stopped it
    "EARLY_SELL":     -0.3,   # left money on table
    "UNKNOWN":        +0.0,
}


class ARTExporter:
    """
    Converts a BacktestRecorder's labeled event log into training data
    for OpenPipe ART fine-tuning.

    Outputs:
      art_samples.jsonl      — all individual samples with rewards
      preference_pairs.jsonl — DPO pairs where available
      sft_data.jsonl         — supervised fine-tuning data (good samples only)
      summary.json           — statistics about the export
    """

    # Only export samples where Claude's full prompt/response was captured
    MIN_PROMPT_LEN = 100

    def __init__(self, recorder: BacktestRecorder):
        self._events = recorder.get_events()

    def export(self, output_dir: str | Path) -> dict:
        """
        Export all training data. Returns summary statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        samples       = self._build_samples()
        pairs         = self._build_preference_pairs(samples)
        sft_samples   = [s for s in samples if s.reward > 0.5]

        # Write art_samples.jsonl
        self._write_jsonl(
            output_dir / "art_samples.jsonl",
            [s.to_openrouter_format() for s in samples],
        )

        # Write preference_pairs.jsonl
        self._write_jsonl(
            output_dir / "preference_pairs.jsonl",
            [p.to_openrouter_format() for p in pairs],
        )

        # Write sft_data.jsonl (OpenAI fine-tune format compatible)
        self._write_jsonl(
            output_dir / "sft_data.jsonl",
            [
                {
                    "messages": [
                        {"role": "user",      "content": s.prompt},
                        {"role": "assistant", "content": s.response},
                    ]
                }
                for s in sft_samples
            ],
        )

        # Label distribution
        label_counts: dict[str, int] = {}
        for s in samples:
            label_counts[s.label] = label_counts.get(s.label, 0) + 1

        avg_reward   = sum(s.reward for s in samples) / max(len(samples), 1)
        avg_pnl      = sum(s.pnl_pct for s in samples) / max(len(samples), 1)

        summary = {
            "total_events":      len(self._events),
            "exportable_samples": len(samples),
            "preference_pairs":  len(pairs),
            "sft_samples":       len(sft_samples),
            "avg_reward":        round(avg_reward, 3),
            "avg_pnl_pct":       round(avg_pnl, 2),
            "label_distribution": label_counts,
            "output_dir":        str(output_dir),
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"ART export: {len(samples)} samples, {len(pairs)} pairs → {output_dir}"
        )
        self._print_summary(summary)
        return summary

    def _build_samples(self) -> list[ARTSample]:
        samples = []
        for i, event in enumerate(self._events):
            dec     = event.get("decision", {})
            outcome = event.get("outcome")

            # Skip events where Claude's prompt wasn't captured
            prompt   = dec.get("raw_prompt", "")
            response = dec.get("raw_response", "")
            if len(prompt) < self.MIN_PROMPT_LEN or not response:
                continue

            # Skip events without outcome (open positions at end of backtest)
            if not outcome:
                continue

            label  = outcome.get("label", "UNKNOWN")
            reward = REWARD_MAP.get(label, 0.0)

            # Scale reward by PnL magnitude (large wins/losses matter more)
            pnl_pct   = outcome.get("pnl_pct", 0.0)
            magnitude = min(abs(pnl_pct) / 50, 1.0)   # normalize to 0-1 at 50% PnL
            reward    = reward * (0.5 + 0.5 * magnitude)
            reward    = max(-1.0, min(1.0, reward))

            samples.append(ARTSample(
                sample_id=f"fenrir_{i:06d}",
                mint=event.get("mint", ""),
                prompt=prompt,
                response=response,
                action=dec.get("action", ""),
                reward=round(reward, 4),
                label=label,
                pnl_pct=round(pnl_pct, 2),
                hold_secs=round(outcome.get("hold_secs", 0), 1),
                model=dec.get("model", ""),
                timestamp=dec.get("timestamp", ""),
            ))

        return samples

    def _build_preference_pairs(
        self, samples: list[ARTSample]
    ) -> list[PreferencePair]:
        """
        Group samples by mint (same token = similar market context),
        then pair good decisions against bad decisions.
        """
        by_mint: dict[str, list[ARTSample]] = {}
        for s in samples:
            by_mint.setdefault(s.mint, []).append(s)

        pairs = []
        for mint, mint_samples in by_mint.items():
            good = sorted(
                [s for s in mint_samples if s.reward > 0.4],
                key=lambda s: s.reward, reverse=True
            )
            bad = sorted(
                [s for s in mint_samples if s.reward < -0.2],
                key=lambda s: s.reward
            )

            for g, b in zip(good, bad):
                # Use the bad sample's prompt as the shared context
                # (both saw a similar situation but decided differently)
                pairs.append(PreferencePair(
                    sample_id=f"pair_{g.sample_id}_{b.sample_id}",
                    prompt=b.prompt,
                    chosen=g.response,
                    rejected=b.response,
                    chosen_reward=g.reward,
                    rejected_reward=b.reward,
                    context_summary=(
                        f"Token {mint[:8]} | "
                        f"chosen={g.action}(PnL={g.pnl_pct:+.1f}%) "
                        f"rejected={b.action}(PnL={b.pnl_pct:+.1f}%)"
                    ),
                ))

        return pairs

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        logger.debug(f"Wrote {len(records)} records → {path}")

    @staticmethod
    def _print_summary(summary: dict) -> None:
        print("\n" + "=" * 55)
        print("FENRIR ART Export Summary")
        print("=" * 55)
        print(f"  Events recorded:     {summary['total_events']}")
        print(f"  Exportable samples:  {summary['exportable_samples']}")
        print(f"  Preference pairs:    {summary['preference_pairs']}")
        print(f"  SFT samples (good):  {summary['sft_samples']}")
        print(f"  Avg reward:          {summary['avg_reward']:+.3f}")
        print(f"  Avg PnL:             {summary['avg_pnl_pct']:+.2f}%")
        print("\n  Label distribution:")
        for label, count in sorted(
            summary["label_distribution"].items(),
            key=lambda x: -x[1]
        ):
            bar = "█" * min(count, 30)
            print(f"    {label:<20} {bar} {count}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# Incident Replayer
# ---------------------------------------------------------------------------

class IncidentReplayer:
    """
    Given a live event log (JSONL written by the live ExecutionRecorder),
    reconstructs exactly what Claude saw during an incident and re-runs
    the strategy through backtest mode to diagnose the failure.

    Workflow:
      1. Load the live event log
      2. Find the incident (by mint + timestamp or loss threshold)
      3. Load historical candle data for the same time window
      4. Re-run strategy with BacktestContextProvider seeded from
         the live snapshots captured in the event log
      5. Compare Claude's live decision vs. what it would decide on replay
      6. Generate an ART sample from the incident

    This closes the loop:
      live incident → replay → labeled sample → ART training → better model
    """

    def __init__(self, live_event_log: str | Path):
        self.log_path    = Path(live_event_log)
        self._live_events: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.log_path.exists():
            raise FileNotFoundError(f"Live event log not found: {self.log_path}")
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._live_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed log line: {line[:80]}")
        logger.info(f"Loaded {len(self._live_events)} live events from {self.log_path}")

    def find_losses(self, min_loss_pct: float = -10.0) -> list[dict]:
        """
        Find events where the trade resulted in a loss exceeding the threshold.
        Only works if outcomes were backfilled; live events need price lookup.
        """
        return [
            e for e in self._live_events
            if e.get("outcome") and e["outcome"].get("pnl_pct", 0) <= min_loss_pct
        ]

    def find_by_mint(self, mint: str) -> list[dict]:
        return [e for e in self._live_events if e.get("mint") == mint]

    def find_by_id(self, incident_id: str) -> dict | None:
        for e in self._live_events:
            if e.get("decision", {}).get("sample_id") == incident_id:
                return e
        return None

    def generate_incident_report(self, event: dict) -> str:
        """
        Generates a human-readable incident report for a single trade event.
        Shows exactly what Claude saw, what it decided, and what happened.
        """
        dec     = event.get("decision", {})
        order   = event.get("order") or {}
        outcome = event.get("outcome") or {}
        context = event.get("context") or {}
        mint    = event.get("mint", "unknown")

        lines = [
            "=" * 60,
            f"FENRIR INCIDENT REPORT",
            f"Mint:      {mint}",
            f"Timestamp: {dec.get('timestamp', 'unknown')}",
            "=" * 60,
            "",
            "--- CLAUDE'S DECISION ---",
            f"Action:     {dec.get('action')}",
            f"Confidence: {dec.get('confidence', 0):.0%}",
            f"Reasoning:  {dec.get('reasoning', '')}",
            f"Model:      {dec.get('model', 'unknown')}",
            f"Tokens:     {dec.get('tokens_used', 0)}",
            f"Latency:    {dec.get('latency_ms', 0):.0f}ms",
            "",
            "--- ORDER EXECUTION ---",
        ]

        if order:
            lines += [
                f"Success:    {order.get('success')}",
                f"Side:       {order.get('side')}",
                f"SOL:        {order.get('sol_amount', 0):.4f}",
                f"Price:      {order.get('effective_price', 0):.8f}",
                f"Slippage:   {order.get('slippage_pct', 0):.2%}",
                f"Latency:    {order.get('latency_ms', 0):.0f}ms",
            ]
        else:
            lines.append("No order placed (HOLD decision)")

        lines += ["", "--- OUTCOME ---"]
        if outcome:
            label = outcome.get("label", "UNKNOWN")
            pnl   = outcome.get("pnl_pct", 0)
            hold  = outcome.get("hold_secs", 0)
            reward = REWARD_MAP.get(label, 0)
            lines += [
                f"Label:      {label}",
                f"PnL:        {pnl:+.2f}%",
                f"Hold time:  {hold:.0f}s ({hold/60:.1f}min)",
                f"ART reward: {reward:+.2f}",
            ]
        else:
            lines.append("No outcome recorded (position still open or data missing)")

        lines += ["", "--- MARKET CONTEXT AT DECISION TIME ---"]
        oc = context.get("on_chain", {})
        if oc:
            lines += [
                f"Price:      {oc.get('price_sol', 0):.8f} SOL",
                f"MCap:       {oc.get('market_cap_sol', 0):.1f} SOL",
                f"Liquidity:  {oc.get('liquidity_sol', 0):.1f} SOL",
                f"Holders:    {oc.get('holder_count', 0)}",
                f"B/S ratio:  {oc.get('buy_sell_ratio', 0):.2f}",
                f"Bonding:    {oc.get('bonding_progress', 0):.0%}",
            ]
        soc = context.get("social", {})
        if soc:
            lines += [
                f"Sentiment:  {soc.get('sentiment_score', 0):+.2f}",
                f"Warnings:   {soc.get('bear_warnings', [])}",
            ]

        lines += [
            "",
            "--- EXACT PROMPT SENT TO CLAUDE ---",
            dec.get("raw_prompt", "[not captured]"),
            "",
            "--- CLAUDE'S EXACT RESPONSE ---",
            dec.get("raw_response", "[not captured]"),
            "=" * 60,
        ]
        return "\n".join(lines)

    def export_incident_as_art(self, event: dict) -> ARTSample | None:
        """
        Convert a single live incident into an ART training sample.
        Only works if outcome is available (requires price lookup or
        retrospective labeling).
        """
        dec     = event.get("decision", {})
        outcome = event.get("outcome")
        prompt  = dec.get("raw_prompt", "")
        response = dec.get("raw_response", "")

        if not outcome or len(prompt) < 50:
            return None

        label  = outcome.get("label", "UNKNOWN")
        pnl    = outcome.get("pnl_pct", 0.0)
        reward = REWARD_MAP.get(label, 0.0)

        return ARTSample(
            sample_id=f"incident_{event.get('mint', '')[:8]}_{dec.get('timestamp', '')[:10]}",
            mint=event.get("mint", ""),
            prompt=prompt,
            response=response,
            action=dec.get("action", ""),
            reward=round(reward, 4),
            label=label,
            pnl_pct=round(pnl, 2),
            hold_secs=round(outcome.get("hold_secs", 0), 1),
            model=dec.get("model", ""),
            timestamp=dec.get("timestamp", ""),
        )

    def print_all_incidents(self, min_loss_pct: float = -10.0) -> None:
        losses = self.find_losses(min_loss_pct)
        if not losses:
            print(f"No losses worse than {min_loss_pct:.0f}% found in log.")
            return
        print(f"\nFound {len(losses)} incidents with loss < {min_loss_pct:.0f}%\n")
        for event in losses:
            print(self.generate_incident_report(event))
            print()


# ---------------------------------------------------------------------------
# Live execution recorder (the production counterpart)
# ---------------------------------------------------------------------------

class LiveExecutionRecorder:
    """
    Live implementation of ExecutionRecorder.
    Writes to an append-only JSONL file — one line per decision event.
    This is what feeds IncidentReplayer after a bad trade.

    Every live Claude decision must call recorder.record() — if it doesn't,
    you lose the data needed to diagnose failures and generate ART samples.
    """

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.log_path, "a")   # append mode — never overwrites
        logger.info(f"LiveExecutionRecorder writing to {self.log_path}")

    async def record(
        self,
        mint:     str,
        decision: "ClaudeDecision",
        order:    "OrderResult | None",
        context:  dict,
        outcome:  dict | None = None,
    ) -> None:
        event = {
            "mint":     mint,
            "decision": decision.to_dict(),
            "order":    order.to_dict() if order else None,
            "context":  context,
            "outcome":  outcome,   # None at decision time; backfilled later
        }
        self._f.write(json.dumps(event) + "\n")

    async def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.flush()
        self._f.close()
