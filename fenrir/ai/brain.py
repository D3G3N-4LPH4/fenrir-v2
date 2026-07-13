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
from collections import deque
from datetime import datetime
from typing import cast

from fenrir.ai.agent_panel import MultiAgentPanel, PanelResult
from fenrir.ai.context_builder import apply_exit_plan_to_position, build_batched_exit_context
from fenrir.ai.decision_engine import (
    AIDecision,
    AITradingAnalyst,
    TokenAnalysis,
    TokenMetadata,
)
from fenrir.ai.ensemble_scorer import EnsembleResult, EnsembleScorer
from fenrir.ai.local_backend import LocalAITradingAnalyst
from fenrir.ai.memory import AISessionMemory, DecisionRecord


def _is_data_poor_launch(token_data: dict) -> bool:
    """True for a bare fresh pump.fun launch (no momentum/social data yet).

    Enriched candidates carry a scanner ``tier``, a ``smart_money_tier``, an
    explicit ``source``, or market/holder data; fresh launches carry none. The
    panel's momentum/narrative lenses can only score the enriched kind, so the
    caller gates launches on the risk/safety lens alone.
    """
    return not (
        token_data.get("tier")
        or token_data.get("smart_money_tier")
        or token_data.get("source") in ("scanner", "smart_money")
        or token_data.get("holder_count")
        or token_data.get("market_cap_usd")
    )


class ClaudeBrain:
    """
    Autonomous Claude Brain for FENRIR.

    Wraps AITradingAnalyst with:
    - Session memory (conversation context across decisions)
    - Timeout fallback (graceful degradation to rule-based)
    - Performance comparison tracking (AI vs rules)
    - Dynamic exit evaluation with override capability
    """

    def __init__(self, config, logger, breaker=None, db_path: str = "fenrir_trades.db", audit=None):
        self.config = config
        self.logger = logger
        self._breaker = breaker
        self._db_path = db_path
        # Optional AuditChain — when ai_memory_resume is set, session memory is
        # rebuilt as a projection of this log on startup (§1 harness kernel).
        self._audit = audit
        self.analyst: AITradingAnalyst | None = None
        # Second-opinion gate: the 2-model EnsembleScorer or the role-specialized
        # MultiAgentPanel (config.ai_multi_agent_enabled). Both expose the same
        # score()/should_trade/position_multiplier/conviction surface.
        self._ensemble_scorer: EnsembleScorer | MultiAgentPanel | None = None
        self.memory = AISessionMemory(max_size=config.ai_memory_size)
        self.enabled = config.ai_analysis_enabled and bool(config.ai_api_key)

        # Per-position exit evaluation timestamps
        self._last_exit_eval: dict[str, datetime] = {}

        # Performance stats
        self.stats: dict[str, float] = {
            "ai_entries_evaluated": 0,
            "ai_entries_bought": 0,
            "ai_entries_skipped": 0,
            "ai_timeouts": 0,
            "ai_errors": 0,
            "rule_fallbacks": 0,
            "ai_exits_evaluated": 0,
            "ai_exits_overridden": 0,
            "ai_avg_response_ms": 0.0,
        }
        # Bounded rolling window of response times (O(1) appends) for averaging.
        self._response_times: deque[float] = deque(maxlen=100)

    async def initialize(self) -> None:
        """Initialize the AI analyst session. Safe to call even if AI is disabled."""
        if not self.enabled:
            self.logger.info(
                "🧠 AI Brain: DISABLED "
                "(set ai_analysis_enabled=True and provide ai_api_key to enable)"
            )
            return

        # §1: optionally rebuild session memory as a projection of the audit log.
        # Off by default — preserves the deliberate reset-on-restart behavior;
        # when enabled (and the audit session_id is reused across restarts) this
        # restores recent decisions/outcomes so the brain isn't blind after a crash.
        if getattr(self.config, "ai_memory_resume", False) and self._audit is not None:
            try:
                restored = AISessionMemory.from_audit_chain(
                    self._audit, max_size=self.config.ai_memory_size
                )
                if restored.decisions:
                    self.memory = restored
                    self.logger.info(
                        f"🧠 AI Brain: restored {len(restored.decisions)} decision(s) "
                        "from audit chain (session memory resume)"
                    )
            except Exception as e:
                self.logger.warning(
                    f"🧠 AI Brain: memory resume failed ({e}); starting with empty memory"
                )

        if getattr(self.config, "ai_local_model_enabled", False):
            self.analyst = LocalAITradingAnalyst(
                base_url=self.config.ai_local_model_url,
                model_name=self.config.ai_local_model_name,
                temperature=self.config.ai_temperature,
                timeout_seconds=int(self.config.ai_entry_timeout_seconds) + 2,
            )
            await self.analyst.initialize()

            # Health check on startup
            healthy, status_msg = await self.analyst.health_check()
            if not healthy:
                self.logger.warning(f"🦙 Local model health check FAILED: {status_msg}")
                self.logger.warning(
                    "   Falling back to cloud API. Fix local server to use abliterated model."
                )
                # Fall back to cloud analyst
                self.analyst = AITradingAnalyst(
                    api_key=self.config.ai_api_key,
                    model=self.config.ai_model,
                    temperature=self.config.ai_temperature,
                    timeout_seconds=int(self.config.ai_entry_timeout_seconds) + 2,
                    breaker=self._breaker,
                    db_path=self._db_path,
                    fallback_models=self.config.ai_model_fallbacks,
                )
                await self.analyst.initialize()
                self.logger.info(
                    f"🧠 AI Brain: ONLINE (cloud fallback, model={self.config.ai_model})"
                )
            else:
                self.logger.info(
                    f"🦙 AI Brain: LOCAL ONLINE (model={self.config.ai_local_model_name}, "
                    f"url={self.config.ai_local_model_url})"
                )
        else:
            self.analyst = AITradingAnalyst(
                api_key=self.config.ai_api_key,
                model=self.config.ai_model,
                temperature=self.config.ai_temperature,
                timeout_seconds=int(self.config.ai_entry_timeout_seconds) + 2,
                breaker=self._breaker,
                db_path=self._db_path,
                fallback_models=self.config.ai_model_fallbacks,
            )
            await self.analyst.initialize()
            self.logger.info(
                f"🧠 AI Brain: ONLINE (model={self.config.ai_model}, "
                f"entry_timeout={self.config.ai_entry_timeout_seconds}s, "
                f"min_confidence={self.config.ai_min_confidence_to_buy})"
            )

        # Second-opinion gate on BUY decisions: role-specialized multi-agent panel
        # (opt-in) or the 2-model ensemble. Both are drop-in for the gate.
        if self.enabled:
            if getattr(self.config, "ai_multi_agent_enabled", False):
                panel = MultiAgentPanel(
                    api_key=self.config.ai_api_key,
                    model=self.config.ai_model,
                )
                await panel.initialize()
                self._ensemble_scorer = panel
                self.logger.info("🔬 Second opinion: multi-agent panel (risk/momentum/narrative)")
            else:
                sol_threshold = getattr(self.config, "ensemble_sol_threshold", 0.5)
                scorer = EnsembleScorer(
                    api_key=self.config.ai_api_key,
                    sol_threshold=sol_threshold,
                )
                await scorer.initialize()
                self._ensemble_scorer = scorer

    # ──────────────────────────────────────────────────────────────
    #  ENTRY EVALUATION
    # ──────────────────────────────────────────────────────────────

    async def evaluate_entry(
        self,
        token_data: dict,
        positions: dict,
        strategy_context: str | None = None,
        historical_context: str | None = None,
    ) -> tuple[bool, TokenAnalysis | None, float | None]:
        """
        Evaluate whether to buy a newly detected token.

        Args:
            token_data: Dict from PumpFunMonitor with token info
            positions: Dict of open positions from PositionManager
            strategy_context: Optional strategy-specific instructions for AI
            historical_context: Optional historical performance context

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

            # Merge optional market-signal / strategy / historical context.
            extra_blocks = []
            market_signal = self._build_market_signal_context(token_data)
            if market_signal:
                extra_blocks.append(market_signal)
            if strategy_context:
                extra_blocks.append(strategy_context)
            if historical_context:
                extra_blocks.append(historical_context)
            if extra_blocks and memory_context:
                memory_context = "\n\n".join(extra_blocks) + "\n\n" + memory_context
            elif extra_blocks:
                memory_context = "\n\n".join(extra_blocks)

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

            # G0DM0D3: ensemble gate — independent second opinion
            symbol = token_data.get("symbol", "???")
            buy_amount = None
            if should_buy and self._ensemble_scorer:
                try:
                    ensemble_ctx = self._build_ensemble_context(token_data, analysis)
                    ensemble: EnsembleResult | PanelResult
                    # Fresh launches have no momentum/social data for the panel's
                    # momentum/narrative lenses — running them would reject every
                    # snipe. Gate launches on the risk/safety lens only (veto).
                    scorer = self._ensemble_scorer
                    if isinstance(scorer, MultiAgentPanel) and _is_data_poor_launch(token_data):
                        ensemble = await scorer.score(
                            context=ensemble_ctx,
                            sol_amount=self.config.buy_amount_sol,
                            veto_only=True,
                        )
                    elif isinstance(scorer, MultiAgentPanel) and token_data.get("tier") in (
                        "mid",
                        "large",
                    ):
                        # Established swing candidates score more moderately than
                        # launch snipes — relax the per-lens BUY bar for them.
                        ensemble = await scorer.score(
                            context=ensemble_ctx,
                            sol_amount=self.config.buy_amount_sol,
                            buy_threshold=self.config.ai_established_buy_threshold,
                        )
                    else:
                        ensemble = await scorer.score(
                            context=ensemble_ctx,
                            sol_amount=self.config.buy_amount_sol,
                        )
                    # Per-lens breakdown (panel: risk/momentum/narrative; ensemble:
                    # per-model) so rejections/sizing are explainable in the logs.
                    detail = (
                        ensemble.summary()
                        if hasattr(ensemble, "summary")
                        else ensemble.conviction.value
                    )
                    if not ensemble.should_trade:
                        should_buy = False
                        self.logger.info(f"🔬 2nd opinion REJECTED ${symbol}: {detail}")
                    elif ensemble.position_multiplier < 1.0:
                        buy_amount = self.config.buy_amount_sol * ensemble.position_multiplier
                        self.logger.info(
                            f"🔬 2nd opinion PARTIAL ${symbol} "
                            f"→ {ensemble.position_multiplier:.0%}: {detail}"
                        )
                    else:
                        self.logger.info(f"🔬 2nd opinion PASSED ${symbol}: {detail}")
                except Exception as ens_err:
                    self.logger.warning(f"🔬 Ensemble scorer error: {ens_err} — skipping gate")

            record.was_bought = should_buy
            self.memory.record_decision(record)

            # Log
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
                if dynamic_sizing and analysis.suggested_buy_amount_sol and buy_amount is None:
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
        report: dict[str, object] = dict(self.stats)
        if self.analyst:
            report["ai_analyst_report"] = self.analyst.get_ai_performance_report()
        report["session_memory"] = self.memory.get_session_stats()
        return report

    # ──────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_metadata_field(value: str | None, max_length: int = 200) -> str | None:
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

    def _build_market_signal_context(self, token_data: dict) -> str | None:
        """Concise DexScreener-momentum + RugCheck-risk block for the AI prompt.

        Best-effort: emits only the lines whose signals are present in token_data
        (populated by the bot from the market filter + security gate). Returns
        None when neither is available so no empty block is injected.
        """
        lines: list[str] = []

        vol_5m = token_data.get("dex_volume_5m_usd")
        if vol_5m is not None:
            parts = [f"5m vol ${vol_5m:,.0f}"]
            buys = token_data.get("dex_txns_5m_buys")
            sells = token_data.get("dex_txns_5m_sells")
            if buys is not None and sells is not None:
                parts.append(f"buys/sells {buys}/{sells}")
            bp = token_data.get("dex_buy_pressure_5m")
            if bp is not None:
                parts.append(f"{bp * 100:.0f}% buy pressure")
            ch_1h = token_data.get("dex_price_change_1h_pct")
            if ch_1h is not None:
                parts.append(f"1h {ch_1h:+.1f}%")
            liq = token_data.get("dex_liquidity_usd")
            if liq is not None:
                parts.append(f"liq ${liq:,.0f}")
            lines.append("DexScreener momentum: " + ", ".join(parts))

        score = token_data.get("rugcheck_score")
        if score is not None:
            risks = token_data.get("rugcheck_risks") or []
            risk_names = (
                ", ".join(str(r.get("name", "?")) for r in risks) if risks else "none flagged"
            )
            lines.append(f"RugCheck risk {score}/100 (lower = safer); flags: {risk_names}")

        wallet = token_data.get("smart_money_wallet")
        if wallet:
            tier = token_data.get("smart_money_tier", "B")
            sol = token_data.get("smart_money_sol") or 0.0
            spent = f", ~{sol:.2f} SOL" if sol > 0 else ""
            lines.append(
                f"Smart-money: {tier}-tier tracked wallet {wallet[:6]}… just bought this{spent}"
            )

        return "\n".join(lines) if lines else None

    def _build_token_metadata(self, token_data: dict) -> TokenMetadata:
        """Convert PumpFunMonitor token_data dict to TokenMetadata for AI analysis."""
        curve_state = token_data.get("bonding_curve_state")
        return TokenMetadata(
            token_mint=token_data["token_address"],
            name=cast(
                str,
                self._sanitize_metadata_field(token_data.get("name", "Unknown"), max_length=100),
            ),
            symbol=cast(
                str, self._sanitize_metadata_field(token_data.get("symbol", "???"), max_length=20)
            ),
            description=self._sanitize_metadata_field(
                token_data.get("description"), max_length=500
            ),
            bonding_curve_state=curve_state,
            initial_liquidity_sol=token_data.get("initial_liquidity_sol", 0),
            current_market_cap_sol=token_data.get("market_cap_sol", 0),
            creator_address=token_data.get("creator"),
            # Scanner-surfaced (mid/large-cap) tokens carry real enrichment —
            # populate it so the prompt's metrics/socials aren't blank zeros.
            holder_count=token_data.get("holder_count", 0) or 0,
            website=self._sanitize_metadata_field(token_data.get("website"), max_length=200),
            twitter=self._sanitize_metadata_field(token_data.get("twitter"), max_length=200),
            telegram=self._sanitize_metadata_field(token_data.get("telegram"), max_length=200),
            tier=token_data.get("tier"),
        )

    def _record_response_time(self, ms: float) -> None:
        """Track response times for averaging (bounded deque, O(1) appends)."""
        times = self._response_times
        times.append(ms)
        self.stats["ai_avg_response_ms"] = sum(times) / len(times) if times else 0.0

    def _build_ensemble_context(self, token_data: dict, analysis: TokenAnalysis) -> str:
        """Build a compact context string for EnsembleScorer's independent scoring."""
        lines = [
            f"Token: {token_data.get('name', 'Unknown')} ({token_data.get('symbol', '???')})",
        ]
        tier = token_data.get("tier")
        if tier in ("mid", "large"):
            # Tell the panel's risk lens this is an ESTABLISHED AMM token, not a
            # fresh pump.fun launch — otherwise it scores it as a likely memecoin
            # rug (safety ~15) and vetoes every scanner candidate.
            mcap_usd = token_data.get("market_cap_usd") or 0
            liq_usd = token_data.get("liquidity_usd") or 0
            holders = token_data.get("holder_count") or 0
            lines.append(
                f"Type: ESTABLISHED {tier.upper()}-CAP trading on an AMM (NOT a fresh launch). "
                f"~${mcap_usd:,.0f} mcap, ~${liq_usd:,.0f} liquidity, {holders:,} holders. "
                "Assess safety by liquidity depth, holder base and track record — a high mcap "
                "is normal here and is NOT itself a rug signal."
            )
            # Momentum/flow signals (Jupiter stats) so the momentum lens scores real
            # data instead of defaulting low on a data-poor context.
            mom = []
            pc1, pc24 = token_data.get("price_change_1h"), token_data.get("price_change_24h")
            if pc1 is not None:
                mom.append(f"1h price {pc1:+.1f}%")
            if pc24 is not None:
                mom.append(f"24h price {pc24:+.1f}%")
            vol = token_data.get("volume_24h_usd")
            if vol:
                mom.append(f"24h volume ~${vol:,.0f}")
            nb, ns = token_data.get("num_buys_24h"), token_data.get("num_sells_24h")
            if nb is not None and ns is not None:
                mom.append(f"24h buys/sells {nb:,}/{ns:,}")
            osl = token_data.get("organic_score_label")
            if osl:
                mom.append(f"organic score: {osl}")
            thp = token_data.get("top_holders_pct")
            if thp is not None:
                mom.append(f"top holders {thp:.0f}%")
            if mom:
                lines.append("Momentum/flow: " + ", ".join(mom))
        lines += [
            f"Liquidity: {token_data.get('initial_liquidity_sol', 0):.2f} SOL",
            f"Market cap: {token_data.get('market_cap_sol', 0):.2f} SOL",
        ]
        if token_data.get("creator"):
            lines.append(f"Creator: {token_data['creator']}")
        if analysis.reasoning:
            lines.append(f"Primary analysis: {analysis.reasoning[:200]}")
        if analysis.red_flags:
            lines.append(f"Red flags: {', '.join(analysis.red_flags[:3])}")
        if analysis.green_flags:
            lines.append(f"Green flags: {', '.join(analysis.green_flags[:3])}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    #  BATCHED EXIT EVALUATION (Nocturne pattern)
    # ──────────────────────────────────────────────────────────────

    async def evaluate_exits_batched(
        self,
        positions: dict,
        portfolio_summary: dict,
        wallet_balance_sol: float = 0.0,
        triggered_exits: dict[str, str] | None = None,
    ) -> list[dict]:
        """
        Evaluate exit actions for ALL open positions in a single LLM call.

        Implements Nocturne's batched-asset pattern: instead of one round-trip
        per position, the full portfolio state is packed into one structured
        context and evaluated at once. Positions still in AI-imposed cooldown
        are automatically skipped.

        Args:
            positions:          Dict of token_address → Position objects
            portfolio_summary:  From PositionManager.get_portfolio_summary()
            wallet_balance_sol: Available wallet balance in SOL
            triggered_exits:    Dict of token_address → mechanical trigger string

        Returns:
            List of exit_decision dicts (one per active position):
            [{"token_address": "...", "action": "HOLD|EXIT|...", "exit_plan": "...", ...}]

        Side effects:
            Calls apply_exit_plan_to_position() on each position so the AI's
            continuation contract is persisted for the next evaluation cycle.
        """
        if not self.enabled or not self.analyst:
            # AI disabled — defer all to mechanical triggers
            return [
                {
                    "token_address": addr,
                    "action": "EXIT" if triggered_exits and addr in triggered_exits else "HOLD",
                    "reasoning": "AI disabled",
                    "urgency": 0.0,
                    "exit_plan": "",
                }
                for addr in positions
            ]

        exit_timeout = self.config.ai_exit_timeout_seconds

        try:
            memory_block = self.memory.build_context_block()
            context_json, active_addresses = build_batched_exit_context(
                positions=positions,
                portfolio_summary=portfolio_summary,
                wallet_balance_sol=wallet_balance_sol,
                session_memory_block=memory_block,
                triggered_exits=triggered_exits or {},
            )

            if not active_addresses:
                return []

            result = await asyncio.wait_for(
                self.analyst.evaluate_exits_batched(
                    context_json=context_json,
                    active_addresses=active_addresses,
                ),
                timeout=exit_timeout,
            )

            decisions: list[dict] = result.get("exit_decisions", [])

            # Persist AI continuation contracts onto Position objects
            for decision in decisions:
                addr = decision.get("token_address")
                exit_plan = decision.get("exit_plan", "")
                if addr and addr in positions and exit_plan:
                    apply_exit_plan_to_position(positions[addr], exit_plan)

            self.stats["ai_exits_evaluated"] += len(decisions)
            return decisions

        except TimeoutError:
            self.stats["ai_timeouts"] += 1
            self.logger.warning(
                f"🧠 AI Brain: batched exit eval TIMEOUT ({exit_timeout}s) "
                f"for {len(positions)} positions"
            )
            return [
                {
                    "token_address": addr,
                    "action": "EXIT" if triggered_exits and addr in triggered_exits else "HOLD",
                    "reasoning": "AI timeout",
                    "urgency": 0.0,
                    "exit_plan": "",
                }
                for addr in positions
            ]

        except Exception as e:
            self.stats["ai_errors"] += 1
            self.logger.error("AI Brain batched exit evaluation", e)
            return [
                {
                    "token_address": addr,
                    "action": "EXIT" if triggered_exits and addr in triggered_exits else "HOLD",
                    "reasoning": "AI error",
                    "urgency": 0.0,
                    "exit_plan": "",
                }
                for addr in positions
            ]

    async def close(self) -> None:
        """Shutdown the brain. Closes HTTP sessions."""
        if self.analyst:
            await self.analyst.close()
            self.analyst = None
        if self._ensemble_scorer:
            await self._ensemble_scorer.close()
            self._ensemble_scorer = None
        self.logger.info("🧠 AI Brain: OFFLINE")


def _now_ms() -> float:
    """Current time in milliseconds."""
    return datetime.now().timestamp() * 1000
