#!/usr/bin/env python3
"""
FENRIR - Claude Brain (Autonomous AI Decision Engine)

The autonomous Claude Brain that sits between token detection and trade execution.
Claude evaluates every token with full context — recent decisions, portfolio state,
session performance — and outputs BUY/SKIP decisions with reasoning.

Also evaluates exit timing, with the ability to override mechanical triggers
when it detects momentum or risk patterns the rules can't capture.

Usage:
    brain = ClaudeBrain(config, logger)
    await brain.initialize()

    # Entry evaluation
    should_buy, analysis, amount = await brain.evaluate_entry(token_data, positions)

    # Exit evaluation (called when mechanical trigger fires, or on cadence)
    action, reason = await brain.evaluate_exit(token_addr, position, trigger)
"""

import asyncio
from datetime import datetime

from fenrir.ai.decision_engine import (
    AIDecision,
    AITradingAnalyst,
    TokenAnalysis,
    TokenMetadata,
)
from fenrir.ai.memory import AISessionMemory, DecisionRecord


class ClaudeBrain:
    """
    Autonomous Claude Brain for FENRIR.

    Wraps AITradingAnalyst with:
    - Session memory (conversation context across decisions)
    - Timeout fallback (graceful degradation to rule-based)
    - Performance comparison tracking (AI vs rules)
    - Dynamic exit evaluation with override capability
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.analyst: AITradingAnalyst | None = None
        self.memory = AISessionMemory(max_size=config.ai_memory_size)
        self.enabled = config.ai_analysis_enabled and bool(config.ai_api_key)

        # Per-position exit evaluation timestamps
        self._last_exit_eval: dict[str, datetime] = {}

        # Performance stats
        self.stats = {
            "ai_entries_evaluated": 0,
            "ai_entries_bought": 0,
            "ai_entries_skipped": 0,
            "ai_timeouts": 0,
            "ai_errors": 0,
            "rule_fallbacks": 0,
            "ai_exits_evaluated": 0,
            "ai_exits_overridden": 0,
            "ai_avg_response_ms": 0.0,
            "_response_times": [],
        }

    async def initialize(self) -> None:
        """Initialize the AI analyst session. Safe to call even if AI is disabled."""
        if not self.enabled:
            self.logger.info(
                "🧠 AI Brain: DISABLED "
                "(set ai_analysis_enabled=True and provide ai_api_key to enable)"
            )
            return

        self.analyst = AITradingAnalyst(
            api_key=self.config.ai_api_key,
            model=self.config.ai_model,
            temperature=self.config.ai_temperature,
            timeout_seconds=int(self.config.ai_entry_timeout_seconds) + 2,
        )
        await self.analyst.initialize()
        self.logger.info(
            f"🧠 AI Brain: ONLINE (model={self.config.ai_model}, "
            f"entry_timeout={self.config.ai_entry_timeout_seconds}s, "
            f"min_confidence={self.config.ai_min_confidence_to_buy})"
        )

    # ──────────────────────────────────────────────────────────────
    #  ENTRY EVALUATION
    # ──────────────────────────────────────────────────────────────

    async def evaluate_entry(
        self,
        token_data: dict,
        positions: dict,
    ) -> tuple[bool, TokenAnalysis | None, float | None]:
        """
        Evaluate whether to buy a newly detected token.

        Args:
            token_data: Dict from PumpFunMonitor with token info
            positions: Dict of open positions from PositionManager

        Returns:
            (should_buy, analysis, buy_amount_override)
            - should_buy: True if AI recommends buying
            - analysis: Full TokenAnalysis (None if AI disabled)
            - buy_amount_override: AI-suggested amount in SOL (None to use config default)

        Fallback behavior:
            - AI disabled → (True, None, None) — auto-buy as before
            - AI timeout → depends on ai_fallback_to_rules config
            - AI error → depends on ai_fallback_to_rules config
        """
        if not self.enabled or not self.analyst:
            return (True, None, None)

        self.stats["ai_entries_evaluated"] += 1
        entry_timeout = self.config.ai_entry_timeout_seconds
        min_confidence = self.config.ai_min_confidence_to_buy
        fallback = self.config.ai_fallback_to_rules
        dynamic_sizing = self.config.ai_dynamic_position_sizing

        try:
            # Build TokenMetadata from the detection dict
            metadata = self._build_token_metadata(token_data)

            # Build conversation context
            memory_context = self.memory.build_context_block()
            portfolio_context = self.memory.build_portfolio_context(positions)
            risk_context = self.memory.get_risk_appetite_adjustment()

            # Call AI with timeout
            start_ms = _now_ms()
            analysis = await asyncio.wait_for(
                self.analyst.analyze_token_launch_with_context(
                    token_metadata=metadata,
                    memory_context=memory_context,
                    portfolio_context=portfolio_context,
                    risk_context=risk_context,
                ),
                timeout=entry_timeout,
            )
            elapsed_ms = _now_ms() - start_ms
            self._record_response_time(elapsed_ms)

            # Record decision in memory
            record = DecisionRecord(
                timestamp=datetime.now(),
                token_mint=token_data["token_address"],
                token_symbol=token_data.get("symbol", "???"),
                token_name=token_data.get("name", "Unknown"),
                decision=analysis.decision.value.upper(),
                confidence=analysis.confidence,
                risk_score=analysis.risk_score,
                reasoning_summary=analysis.reasoning[:100],
                red_flags=analysis.red_flags or [],
                green_flags=analysis.green_flags or [],
            )

            # Determine if we should buy
            should_buy = (
                analysis.decision in (AIDecision.STRONG_BUY, AIDecision.BUY)
                and analysis.confidence >= min_confidence
            )
            record.was_bought = should_buy
            self.memory.record_decision(record)

            # Log
            symbol = token_data.get("symbol", "???")
            self.logger.info(
                f"🧠 AI BRAIN: {analysis.decision.value.upper()} ${symbol} "
                f"(conf={analysis.confidence:.0%}, risk={analysis.risk_score:.1f}/10, "
                f"{elapsed_ms:.0f}ms)"
            )
            if analysis.reasoning:
                self.logger.info(f"   Reasoning: {analysis.reasoning[:120]}")
            if analysis.red_flags:
                self.logger.info(f"   Red flags: {', '.join(analysis.red_flags[:3])}")
            if analysis.green_flags:
                self.logger.info(f"   Green flags: {', '.join(analysis.green_flags[:3])}")

            if should_buy:
                self.stats["ai_entries_bought"] += 1
                buy_amount = None
                if dynamic_sizing and analysis.suggested_buy_amount_sol:
                    buy_amount = analysis.suggested_buy_amount_sol
                return (True, analysis, buy_amount)
            else:
                self.stats["ai_entries_skipped"] += 1
                return (False, analysis, None)

        except TimeoutError:
            self.stats["ai_timeouts"] += 1
            symbol = token_data.get("symbol", "???")
            self.logger.warning(f"🧠 AI BRAIN: TIMEOUT ({entry_timeout}s) for ${symbol}")
            if fallback:
                self.stats["rule_fallbacks"] += 1
                self.logger.info("   Falling back to rule-based auto-buy")
                return (True, None, None)
            return (False, None, None)

        except Exception as e:
            self.stats["ai_errors"] += 1
            self.logger.error("AI Brain entry evaluation", e)
            if fallback:
                self.stats["rule_fallbacks"] += 1
                return (True, None, None)
            return (False, None, None)

    # ──────────────────────────────────────────────────────────────
    #  EXIT EVALUATION
    # ──────────────────────────────────────────────────────────────

    async def evaluate_exit(
        self,
        token_address: str,
        position,
        mechanical_trigger: str | None = None,
    ) -> tuple[str, str | None]:
        """
        Evaluate whether to exit a position.

        Args:
            token_address: The token mint address
            position: Position object with price/pnl data
            mechanical_trigger: If set, a rule-based trigger fired (e.g. "Take Profit: +102%")

        Returns:
            (action, reason) where action is one of:
            - "HOLD" — keep the position open
            - "EXIT" — close the position
            - "OVERRIDE_HOLD" — AI overrides a mechanical trigger to keep holding

        Cadence:
            - When mechanical_trigger is set: always evaluates (gives AI override chance)
            - When no trigger: only evaluates if ai_exit_eval_interval_seconds has passed
        """
        if not self.enabled or not self.analyst:
            if mechanical_trigger:
                return ("EXIT", mechanical_trigger)
            return ("HOLD", None)

        # Cadence check: skip if evaluated too recently (unless trigger fired)
        now = datetime.now()
        exit_interval = self.config.ai_exit_eval_interval_seconds
        last_eval = self._last_exit_eval.get(token_address)
        if (
            not mechanical_trigger
            and last_eval
            and (now - last_eval).total_seconds() < exit_interval
        ):
            return ("HOLD", None)

        self._last_exit_eval[token_address] = now
        self.stats["ai_exits_evaluated"] += 1

        exit_timeout = self.config.ai_exit_timeout_seconds

        try:
            pnl_pct = position.get_pnl_percent()
            entry_price = position.entry_price
            current_price = position.current_price
            peak_price = position.peak_price
            entry_time = position.entry_time
            hold_minutes = int((now - entry_time).total_seconds() / 60)

            memory_context = self.memory.build_context_block()

            result = await asyncio.wait_for(
                self.analyst.evaluate_exit_strategy_with_context(
                    token_mint=token_address,
                    entry_price=entry_price,
                    current_price=current_price,
                    peak_price=peak_price,
                    hold_time_minutes=hold_minutes,
                    current_pnl_pct=pnl_pct,
                    mechanical_trigger=mechanical_trigger,
                    memory_context=memory_context,
                ),
                timeout=exit_timeout,
            )

            action = result.get("action", "HOLD").upper()
            reasoning = result.get("reasoning", "")

            # Normalize action names
            if action == "TAKE_PROFIT":
                action = "EXIT"
                reasoning = f"Take profit: {reasoning}"

            # Safety floor: never override stop-loss if drawdown is extreme
            stop_loss_pct = self.config.stop_loss_pct
            hard_floor = stop_loss_pct * 1.5
            if mechanical_trigger and action == "HOLD" and pnl_pct <= -hard_floor:
                self.logger.warning(
                    f"🧠 AI wanted to OVERRIDE but hard floor hit "
                    f"(pnl={pnl_pct:.1f}% <= -{hard_floor:.0f}%). Forcing EXIT."
                )
                return ("EXIT", f"{mechanical_trigger} (hard floor override)")

            # Log the decision
            addr_short = token_address[:8] + "..."
            if mechanical_trigger and action == "HOLD":
                self.stats["ai_exits_overridden"] += 1
                self.logger.info(
                    f"🧠 AI EXIT OVERRIDE: Holding {addr_short} "
                    f"despite '{mechanical_trigger}' — {reasoning[:80]}"
                )
                return ("OVERRIDE_HOLD", f"AI override: {reasoning}")

            if action == "EXIT":
                self.logger.info(f"🧠 AI EXIT: {addr_short} — {reasoning[:80]}")
                return ("EXIT", reasoning)

            return ("HOLD", None)

        except TimeoutError:
            self.logger.warning(
                f"🧠 AI exit eval TIMEOUT ({exit_timeout}s) for {token_address[:8]}..."
            )
            if mechanical_trigger:
                return ("EXIT", mechanical_trigger)
            return ("HOLD", None)

        except Exception as e:
            self.logger.error("AI Brain exit evaluation", e)
            if mechanical_trigger:
                return ("EXIT", mechanical_trigger)
            return ("HOLD", None)

    # ──────────────────────────────────────────────────────────────
    #  OUTCOME TRACKING
    # ──────────────────────────────────────────────────────────────

    def record_trade_outcome(
        self,
        token_address: str,
        pnl_pct: float,
        exit_reason: str,
        hold_time_minutes: int,
        pnl_sol: float = 0.0,
    ) -> None:
        """Record the actual outcome for a previously analyzed token."""
        self.memory.update_outcome(token_address, pnl_pct, exit_reason, hold_time_minutes, pnl_sol)
        if self.analyst:
            self.analyst.track_prediction_outcome(token_address, pnl_pct, hold_time_minutes)
        # Clean up per-position exit eval timestamp (position is closed)
        self._last_exit_eval.pop(token_address, None)

    def get_performance_report(self) -> dict:
        """Get combined performance report: brain stats + AI analyst accuracy."""
        report = {k: v for k, v in self.stats.items() if k != "_response_times"}
        if self.analyst:
            report["ai_analyst_report"] = self.analyst.get_ai_performance_report()
        report["session_memory"] = self.memory.get_session_stats()
        return report

    # ──────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_metadata_field(value: str, max_length: int = 200) -> str:
        """Sanitize user-controlled metadata to prevent prompt injection."""
        if not value:
            return value
        # Truncate to prevent flooding
        value = value[:max_length]
        # Strip control characters and known prompt-injection delimiters
        value = "".join(ch for ch in value if ch.isprintable() or ch in (" ", "\n"))
        # Remove markdown heading syntax that could be misinterpreted
        value = value.replace("#", "").replace("```", "")
        return value.strip()

    def _build_token_metadata(self, token_data: dict) -> TokenMetadata:
        """Convert PumpFunMonitor token_data dict to TokenMetadata for AI analysis."""
        curve_state = token_data.get("bonding_curve_state")
        return TokenMetadata(
            token_mint=token_data["token_address"],
            name=self._sanitize_metadata_field(token_data.get("name", "Unknown"), max_length=100),
            symbol=self._sanitize_metadata_field(token_data.get("symbol", "???"), max_length=20),
            description=self._sanitize_metadata_field(
                token_data.get("description"), max_length=500
            ),
            bonding_curve_state=curve_state,
            initial_liquidity_sol=token_data.get("initial_liquidity_sol", 0),
            current_market_cap_sol=token_data.get("market_cap_sol", 0),
            creator_address=token_data.get("creator"),
        )

    def _record_response_time(self, ms: float) -> None:
        """Track response times for averaging."""
        times = self.stats["_response_times"]
        times.append(ms)
        if len(times) > 100:
            times.pop(0)
        self.stats["ai_avg_response_ms"] = sum(times) / len(times)

    async def close(self) -> None:
        """Shutdown the brain. Closes HTTP sessions."""
        if self.analyst:
            await self.analyst.close()
            self.analyst = None
        self.logger.info("🧠 AI Brain: OFFLINE")


def _now_ms() -> float:
    """Current time in milliseconds."""
    return datetime.now().timestamp() * 1000
