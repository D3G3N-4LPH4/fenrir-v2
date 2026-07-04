#!/usr/bin/env python3
"""
FENRIR Strategy: Narrative / Cult Tracker (Swing Trading)

Tracks social meta-narratives and buys secondary tokens when the
primary leader of a narrative pumps hard. Liquidity naturally rotates
from the leader into the "beta plays" in the same narrative cluster.

Entry logic:
  - Token must match an active narrative (AI agents, dog breeds, political, etc.)
  - The narrative's primary token must be in an uptrend (1h price change > threshold)
  - Secondary token: lower market cap than leader, same narrative tag
  - Market cap: $50k to $500k (early beta, before rotation hits)
  - Momentum building: 1h volume > $50k and accelerating

Exit logic:
  - Exit completely when the primary leader starts losing its 1h uptrend
  - Trailing stop: 20% (narrative can vanish in hours)
  - Hard take profit at 3x (narrative pumps rarely last longer)
  - Max hold: 12 hours

Risk: LOWER than sniping, but entirely dependent on social sentiment.
Hype cycles on Solana vanish within 24 hours.

Note: Narrative detection uses token metadata (name/symbol/description)
and is enhanced by the AI brain which is prompted to identify narrative
matches and leader/beta relationships.

Conforms to the ``TradingStrategy`` ABC (registers in STRATEGY_REGISTRY) while
retaining the richer ``evaluate_token`` / ``NarrativeSignal`` machinery, which
gates on the DexScreener ``MarketData`` produced by ``fenrir.filters``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fenrir.config import BotConfig
from fenrir.strategies.base import TradeParams, TradingStrategy

logger = logging.getLogger("FENRIR.NarrativeTracker")

# Known narrative keyword clusters
# AI brain will extend these dynamically based on current meta
NARRATIVE_CLUSTERS: dict[str, list[str]] = {
    "ai_agents": [
        "ai",
        "agent",
        "gpt",
        "claude",
        "llm",
        "neural",
        "brain",
        "agi",
        "openai",
        "devin",
        "manus",
        "cognition",
    ],
    "dog_breeds": [
        "doge",
        "shib",
        "floki",
        "bonk",
        "dogwifhat",
        "wif",
        "pitbull",
        "husky",
        "corgi",
        "poodle",
        "labrador",
        "retriever",
    ],
    "political": [
        "trump",
        "maga",
        "biden",
        "kamala",
        "pepe",
        "frog",
        "patriot",
        "freedom",
        "america",
        "usa",
    ],
    "cats": [
        "cat",
        "nyan",
        "popcat",
        "michi",
        "mew",
        "meow",
        "kitty",
        "feline",
        "purrfect",
    ],
    "gaming": [
        "gamer",
        "pixel",
        "arcade",
        "mario",
        "pokemon",
        "gg",
        "noob",
        "pwned",
        "loot",
        "quest",
    ],
    "anime": [
        "anime",
        "waifu",
        "naruto",
        "goku",
        "senpai",
        "kawaii",
        "otaku",
        "manga",
        "shonen",
    ],
}


def detect_narrative(token_name: str, token_symbol: str) -> str | None:
    """
    Detect which narrative cluster a token belongs to.
    Returns the cluster name or None if no match.
    """
    search_text = f"{token_name} {token_symbol}".lower()
    search_text = re.sub(r"[^a-z0-9 ]", " ", search_text)
    words = set(search_text.split())

    for cluster_name, keywords in NARRATIVE_CLUSTERS.items():
        if any(kw in search_text or kw in words for kw in keywords):
            return cluster_name

    return None


@dataclass
class NarrativeConfig:
    """Tunable parameters for the narrative tracker strategy."""

    # Primary leader must be pumping this hard (1h %)
    min_leader_pump_1h_pct: float = 50.0
    # Beta play market cap range
    min_market_cap_usd: float = 50_000.0
    max_market_cap_usd: float = 500_000.0
    # Momentum building in beta token
    min_volume_1h_usd: float = 50_000.0
    min_txns_1h: int = 75
    # Minimum liquidity
    min_liquidity_usd: float = 25_000.0
    # Token age window (hours)
    min_age_hours: float = 0.5
    max_age_hours: float = 48.0
    # Trailing stop (narrative can die fast)
    trailing_stop_pct: float = 20.0
    # Hard take profit (3x)
    take_profit_pct: float = 200.0
    # Hard stop
    stop_loss_pct: float = 25.0
    # Max hold (hours)
    max_hold_hours: float = 12.0
    # AI confidence threshold
    ai_min_confidence: float = 0.58
    # Daily budget
    daily_budget_sol: float = 0.0


@dataclass
class NarrativeSignal:
    """Signal for a narrative beta play opportunity."""

    token_address: str
    pair_address: str
    token_name: str
    token_symbol: str
    narrative: str
    age_hours: float
    market_cap_usd: float
    liquidity_usd: float
    volume_1h_usd: float
    txns_1h: int
    price_change_1h_pct: float
    price_usd: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def narrative_momentum_score(self) -> float:
        """0-1 score for how strong the narrative momentum is."""
        vol_score = min(1.0, self.volume_1h_usd / 200_000.0)
        txn_score = min(1.0, self.txns_1h / 300.0)
        momentum_score = min(1.0, max(0.0, self.price_change_1h_pct / 100.0))
        return (vol_score * 0.4) + (txn_score * 0.3) + (momentum_score * 0.3)


class NarrativeTrackerStrategy(TradingStrategy):
    """
    Narrative and cult tracking swing strategy.

    Identifies tokens belonging to active meta-narratives and enters
    as beta plays when the narrative leader is pumping. Uses AI to
    validate narrative match and assess rotation timing.
    """

    strategy_id = "narrative_tracker"
    display_name = "Narrative / Cult Tracker"
    description = (
        "Swing-trades narrative beta plays: buys lower-cap tokens in an active "
        "meta-narrative (AI agents, dog breeds, etc.) when the narrative leader "
        "is pumping and liquidity is rotating. Exits fast if the leader fades."
    )

    budget_sol = 1.0
    max_concurrent_positions = 3

    def __init__(self, config: BotConfig) -> None:
        super().__init__()
        self.config = config
        self.params = NarrativeConfig()
        # Track active narratives and their leader performance
        self._narrative_leaders: dict[str, dict[str, Any]] = {}

        self._params = TradeParams(
            buy_amount_sol=config.buy_amount_sol,
            max_slippage_bps=config.max_slippage_bps,
            stop_loss_pct=self.params.stop_loss_pct,
            take_profit_pct=self.params.take_profit_pct,
            trailing_stop_pct=self.params.trailing_stop_pct,
            max_position_age_minutes=int(self.params.max_hold_hours * 60),
            priority_fee_lamports=config.priority_fee_lamports,
            ai_min_confidence=self.params.ai_min_confidence,
            ai_temperature=config.ai_temperature,
            ai_entry_timeout=config.ai_entry_timeout_seconds,
        )

    def update_narrative_leader(self, narrative: str, leader_data: dict[str, Any]) -> None:
        """
        Update the known state of a narrative leader.
        Called externally when a leader token is detected pumping.
        """
        self._narrative_leaders[narrative] = {
            "updated_at": datetime.now(UTC),
            "price_change_1h_pct": leader_data.get("price_change_1h_pct", 0.0),
            "volume_1h_usd": leader_data.get("volume_1h_usd", 0.0),
            "market_cap_usd": leader_data.get("market_cap_usd", 0.0),
        }
        logger.info(
            f"Narrative leader updated: {narrative} | "
            f"1h={leader_data.get('price_change_1h_pct', 0):+.0f}%"
        )

    def _is_leader_pumping(self, narrative: str) -> bool:
        """Check if the narrative leader is currently in an uptrend."""
        leader = self._narrative_leaders.get(narrative)
        if not leader:
            return False
        change = leader.get("price_change_1h_pct", 0.0)
        return bool(change >= self.params.min_leader_pump_1h_pct)

    # ── ABC interface ──────────────────────────────────────────────────

    async def should_evaluate(self, token_data: dict) -> bool:
        """
        Cheap pre-filter on token_data only: does the name/symbol match a
        known narrative cluster? The market-condition gating (mcap, volume,
        liquidity, age) happens in ``evaluate_token``.
        """
        name = token_data.get("name", "")
        symbol = token_data.get("symbol", "")
        return detect_narrative(name, symbol) is not None

    def get_ai_context(self) -> str:
        return (
            "# STRATEGY CONTEXT: NARRATIVE BETA PLAY\n"
            "You are evaluating a lower-cap token as a 'beta play' within an "
            "active social meta-narrative, entered as liquidity rotates from "
            "the narrative leader.\n"
            "Key evaluation tasks for this strategy:\n"
            "- Is the narrative genuinely active on Solana right now?\n"
            "- Is this a real beta play or just a superficial keyword match?\n"
            "- Has liquidity rotation from the leader already started?\n"
            "- Is social sentiment (X/Telegram) supporting the narrative?\n"
            "- What is the realistic exit window before the hype dies?\n"
            "- Narrative plays live and die on social momentum — be critical\n"
            "- Time horizon: hours; exit fast if the leader loses its uptrend\n"
        )

    def get_trade_params(self) -> TradeParams:
        return self._params

    # ── Rich signal machinery (used by the market-data stage) ──────────

    def evaluate_token(
        self,
        token_data: dict[str, Any],
        market_data: Any | None = None,
    ) -> NarrativeSignal | None:
        if not self.state.active or market_data is None:
            return None

        token_address = token_data.get("token_address", "")
        token_name = token_data.get("name", "")
        token_symbol = token_data.get("symbol", "")

        # Detect narrative
        narrative = detect_narrative(token_name, token_symbol)
        if not narrative:
            return None  # Not in any known narrative cluster

        age_minutes = getattr(market_data, "age_minutes", 0.0)
        age_hours = age_minutes / 60.0
        mcap = getattr(market_data, "market_cap_usd", 0.0)
        liq = getattr(market_data, "liquidity_usd", 0.0)
        vol_1h = getattr(market_data, "volume_1h_usd", 0.0)
        txns_1h = getattr(market_data, "txns_1h_total", 0)
        price_usd = getattr(market_data, "price_usd", 0.0)
        change_1h = getattr(market_data, "price_change_1h_pct", 0.0)
        pair_address = getattr(market_data, "pair_address", "") or ""

        # Age window — silent skip
        if not (self.params.min_age_hours <= age_hours <= self.params.max_age_hours):
            return None

        failures = []

        # Check if narrative leader is pumping.
        # If we don't have leader data, let the AI assess it.
        if not self._is_leader_pumping(narrative):
            logger.debug(
                f"Narrative {narrative}: leader not currently pumping — allowing AI to assess"
            )

        if mcap < self.params.min_market_cap_usd:
            failures.append(f"MCap ${mcap:,.0f} < min ${self.params.min_market_cap_usd:,.0f}")
        if mcap > self.params.max_market_cap_usd:
            failures.append(f"MCap ${mcap:,.0f} > max ${self.params.max_market_cap_usd:,.0f}")
        if liq < self.params.min_liquidity_usd:
            failures.append(f"LP ${liq:,.0f} < min ${self.params.min_liquidity_usd:,.0f}")
        if vol_1h < self.params.min_volume_1h_usd:
            failures.append(f"Vol(1h) ${vol_1h:,.0f} < min ${self.params.min_volume_1h_usd:,.0f}")
        if txns_1h < self.params.min_txns_1h:
            failures.append(f"Txns(1h) {txns_1h} < min {self.params.min_txns_1h}")

        if failures:
            logger.debug(
                f"Narrative reject {token_address[:8]}... [{narrative}]: {' | '.join(failures)}"
            )
            return None

        signal = NarrativeSignal(
            token_address=token_address,
            pair_address=pair_address,
            token_name=token_name,
            token_symbol=token_symbol,
            narrative=narrative,
            age_hours=age_hours,
            market_cap_usd=mcap,
            liquidity_usd=liq,
            volume_1h_usd=vol_1h,
            txns_1h=txns_1h,
            price_change_1h_pct=change_1h,
            price_usd=price_usd,
            metadata={
                "strategy": self.strategy_id,
                "stop_loss_pct": self.params.stop_loss_pct,
                "take_profit_pct": self.params.take_profit_pct,
                "trailing_stop_pct": self.params.trailing_stop_pct,
                "max_hold_hours": self.params.max_hold_hours,
                "ai_min_confidence": self.params.ai_min_confidence,
                "leader_pumping": self._is_leader_pumping(narrative),
                "leader_data": self._narrative_leaders.get(narrative, {}),
            },
        )

        logger.info(
            f"Narrative SIGNAL {token_address[:8]}... | "
            f"narrative={narrative} name='{token_name}' "
            f"mcap=${mcap:,.0f} 1h={change_1h:+.1f}% "
            f"momentum={signal.narrative_momentum_score:.2f}"
        )
        return signal

    def build_ai_context(self, signal: NarrativeSignal) -> str:
        """Per-signal context injected into the AI prompt for this candidate."""
        leader_data = signal.metadata.get("leader_data", {})
        leader_str = (
            f"Leader 1h pump: {leader_data.get('price_change_1h_pct', 0.0):+.0f}%"
            if leader_data
            else "Leader status: unknown — AI should assess current narrative strength"
        )

        return "\n".join(
            [
                "=== NARRATIVE BETA PLAY EVALUATION ===",
                f"Strategy: {self.display_name}",
                f"Token: {signal.token_address}",
                f"Name: {signal.token_name} (${signal.token_symbol})",
                f"Narrative cluster: {signal.narrative.upper().replace('_', ' ')}",
                f"Token age: {signal.age_hours:.1f} hours",
                f"Market cap: ${signal.market_cap_usd:,.0f} (beta play)",
                f"Liquidity: ${signal.liquidity_usd:,.0f}",
                f"1-hour volume: ${signal.volume_1h_usd:,.0f}",
                f"1-hour transactions: {signal.txns_1h}",
                f"1-hour price change: {signal.price_change_1h_pct:+.1f}%",
                f"Narrative momentum score: {signal.narrative_momentum_score:.2f}/1.00",
                f"{leader_str}",
                "",
                "EXIT PLAN (swing):",
                f"  Take profit: +{signal.metadata['take_profit_pct']:.0f}% (3x)",
                f"  Trailing stop: {signal.metadata['trailing_stop_pct']:.0f}% "
                f"(EXIT FAST if leader loses 1h uptrend)",
                f"  Hard stop: -{signal.metadata['stop_loss_pct']:.0f}%",
                f"  Max hold: {signal.metadata['max_hold_hours']:.0f} hours",
                "",
                "AI EVALUATION TASKS:",
                f"  1. Is the '{signal.narrative}' narrative currently active on Solana?",
                "  2. Is this token a genuine beta play or just a keyword match?",
                "  3. Has liquidity rotation from the leader already started?",
                "  4. Is social sentiment (X/Telegram) supporting this narrative?",
                "  5. What is the realistic exit window before hype dies?",
                "",
                "DECISION: BUY or SKIP with confidence 0.0-1.0.",
                "Narrative plays live and die on social momentum. Be critical.",
            ]
        )
