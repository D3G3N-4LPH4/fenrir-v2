#!/usr/bin/env python3
"""
FENRIR - AI Decision Engine

Integrate LLM reasoning into trading decisions for superior alpha.

The AI analyzes:
- Token metadata (name, symbol, description)
- Social signals (Twitter mentions, Telegram activity)
- Holder distribution (whale concentration)
- Historical patterns (similar tokens)
- On-chain metrics (liquidity, volume, velocity)
- Market conditions (overall sentiment, volatility)

Then provides:
- BUY/SKIP recommendation with confidence score
- Reasoning explanation
- Risk assessment
- Optimal exit strategy

Models supported:
- Anthropic Claude (via OpenRouter)
- OpenAI GPT-4 (via OpenRouter)
- Local models (via Ollama)
"""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import aiohttp

from fenrir.protocol.pumpfun import BondingCurveState
from fenrir.ai.structured_output import (
    ENTRY_ANALYSIS_SCHEMA,
    EXIT_ANALYSIS_SCHEMA,
    BATCHED_EXIT_SCHEMA,
    build_response_format,
    parse_or_sanitize,
)
from fenrir.ai.provider_resilience import ProviderResilientCaller

logger = logging.getLogger(__name__)


class AIDecision(Enum):
    """AI trading decisions."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    SKIP = "skip"
    AVOID = "avoid"


@dataclass
class TokenAnalysis:
    """AI analysis of a token launch."""

    decision: AIDecision
    confidence: float  # 0.0-1.0
    reasoning: str
    risk_score: float  # 0.0-10.0 (higher = riskier)

    # Recommended parameters
    suggested_buy_amount_sol: float | None = None
    suggested_stop_loss_pct: float | None = None
    suggested_take_profit_pct: float | None = None

    # Analysis details
    social_score: float | None = None
    liquidity_score: float | None = None
    holder_score: float | None = None
    timing_score: float | None = None

    # Red flags
    red_flags: list[str] = field(default_factory=list)
    green_flags: list[str] = field(default_factory=list)

    # Metadata
    analyzed_at: datetime | None = None
    model_used: str | None = None

    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.now()


@dataclass
class TokenMetadata:
    """Token launch metadata for AI analysis."""

    token_mint: str
    name: str
    symbol: str
    description: str | None = None

    # URLs
    website: str | None = None
    twitter: str | None = None
    telegram: str | None = None
    discord: str | None = None

    # On-chain data
    bonding_curve_state: BondingCurveState | None = None
    initial_liquidity_sol: float = 0.0
    current_market_cap_sol: float = 0.0
    holder_count: int = 0
    top_10_holder_pct: float = 0.0

    # Creator info
    creator_address: str | None = None
    creator_previous_launches: int = 0
    creator_success_rate: float = 0.0

    # Social signals
    twitter_followers: int = 0
    telegram_members: int = 0
    launch_announcement_engagement: float = 0.0


class AITradingAnalyst:
    """
    AI-powered trading analyst using LLMs.
    Provides intelligent buy/skip decisions with reasoning.
    """

    OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"

    # Cheap model for sanitizer fallback (called only on primary parse failures)
    SANITIZE_MODEL = "openai/gpt-4o-mini"

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
        temperature: float = 0.3,  # Lower = more conservative
        timeout_seconds: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout_seconds

        self.session: aiohttp.ClientSession | None = None
        # Created in initialize() once the session exists.
        # Persists for the analyst lifetime, carrying structured-output / tool
        # degradation state across calls (Nocturne allow_structured pattern).
        self._caller: ProviderResilientCaller | None = None

        # Track AI performance (bounded to prevent unbounded growth)
        self.predictions: deque = deque(maxlen=200)

    async def initialize(self):
        """Initialize HTTP session and provider-resilient caller."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        if not self._caller:
            self._caller = ProviderResilientCaller(
                api_key=self.api_key,
                session=self.session,
                url=self.OPENROUTER_API,
            )

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def analyze_token_launch(
        self, token_metadata: TokenMetadata, market_conditions: dict | None = None
    ) -> TokenAnalysis:
        """
        Analyze a token launch and provide trading decision.

        Args:
            token_metadata: All available data about the token
            market_conditions: Overall market sentiment, volatility, etc.

        Returns:
            TokenAnalysis with decision and reasoning
        """
        if not self.session:
            await self.initialize()

        # Build analysis prompt
        prompt = self._build_analysis_prompt(token_metadata, market_conditions)

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        analysis = self._parse_llm_response(response, token_metadata)

        # Track prediction for later evaluation
        self.predictions.append(
            {
                "token_mint": token_metadata.token_mint,
                "analysis": analysis,
                "actual_performance": None,  # Fill in later
            }
        )

        return analysis

    def _build_analysis_prompt(self, token: TokenMetadata, market_conditions: dict | None) -> str:
        """
        Build a comprehensive prompt for the LLM.
        This is where the magic happens.
        """
        # Calculate derived metrics
        liquidity_ratio = (
            token.initial_liquidity_sol / token.current_market_cap_sol
            if token.current_market_cap_sol > 0
            else 0
        )

        prompt = f"""You are an expert memecoin trading analyst specializing in pump.fun token launches on Solana. Analyze the following token and provide a trading decision.

# TOKEN INFORMATION
Name: {token.name}
Symbol: {token.symbol}
Mint Address: {token.token_mint}
Description: {token.description or "Not provided"}

# ON-CHAIN METRICS
Initial Liquidity: {token.initial_liquidity_sol:.2f} SOL
Current Market Cap: {token.current_market_cap_sol:.2f} SOL
Liquidity Ratio: {liquidity_ratio:.2%}
Holder Count: {token.holder_count}
Top 10 Holders Control: {token.top_10_holder_pct:.1f}%

# BONDING CURVE STATUS"""

        if token.bonding_curve_state:
            bc = token.bonding_curve_state
            prompt += f"""
Current Price: ${bc.get_price():.10f}
Migration Progress: {bc.get_migration_progress():.1f}%
Completed: {"Yes" if bc.complete else "No"}
"""

        prompt += f"""
# CREATOR INFORMATION
Address: {token.creator_address or "Unknown"}
Previous Launches: {token.creator_previous_launches}
Success Rate: {token.creator_success_rate:.1%}

# SOCIAL SIGNALS
Twitter: {token.twitter or "Not found"}
Twitter Followers: {token.twitter_followers}
Telegram: {token.telegram or "Not found"}
Telegram Members: {token.telegram_members}
Launch Engagement: {token.launch_announcement_engagement:.1f}/10

# MARKET CONDITIONS"""

        if market_conditions:
            prompt += f"""
Overall Sentiment: {market_conditions.get('sentiment', 'neutral')}
Market Volatility: {market_conditions.get('volatility', 'normal')}
Recent Launch Success Rate: {market_conditions.get('recent_success_rate', 0):.1%}
"""
        else:
            prompt += "\nNot provided"

        prompt += """

# YOUR TASK
Analyze this token launch and provide:

1. **DECISION**: One of: STRONG_BUY, BUY, SKIP, or AVOID
2. **CONFIDENCE**: 0.0 to 1.0 (how confident are you?)
3. **RISK_SCORE**: 0.0 to 10.0 (higher = riskier)
4. **REASONING**: Detailed explanation of your decision
5. **RED_FLAGS**: List any concerning signals
6. **GREEN_FLAGS**: List any positive signals
7. **RECOMMENDED_PARAMETERS**: Suggested buy amount (SOL), stop loss (%), take profit (%)

# EVALUATION CRITERIA
- **Liquidity**: Is there enough liquidity to exit? (5+ SOL good, 2- SOL risky)
- **Holder Distribution**: Too concentrated = rug risk. Aim for <30% in top 10.
- **Social Presence**: Real community or botted? Quality > quantity.
- **Creator History**: Serial ruggers = AVOID. Proven track = GREEN FLAG.
- **Market Cap**: Too high too fast = FOMO trap. Early entry better.
- **Timing**: Is this a good entry point or already pumped?
- **Name/Symbol**: Derivative/copycat names often underperform.

# RESPONSE FORMAT
Respond ONLY with valid JSON in this exact format:
```json
{
  "decision": "BUY",
  "confidence": 0.75,
  "risk_score": 6.5,
  "reasoning": "Your detailed analysis here...",
  "red_flags": ["List", "of", "concerns"],
  "green_flags": ["List", "of", "positives"],
  "suggested_buy_amount_sol": 0.1,
  "suggested_stop_loss_pct": 25,
  "suggested_take_profit_pct": 100,
  "social_score": 7.0,
  "liquidity_score": 8.0,
  "holder_score": 6.5,
  "timing_score": 9.0
}
```

Remember: You're trading REAL money. Be conservative. Most memecoins go to zero. Only recommend BUY when conviction is high.
"""

        return prompt

    async def _call_llm_structured(
        self,
        system: str,
        user: str,
        response_format: dict | None = None,
        tools: list | None = None,
    ) -> dict:
        """
        Call OpenRouter and return the raw message dict.

        Uses ProviderResilientCaller so structured-output and tool failures
        degrade automatically without bubbling exceptions.
        Replaces the fragile _call_llm() → response.find("{") pattern.
        """
        if not self._caller:
            await self.initialize()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": 2000,
        }
        try:
            data = await self._caller.post(
                payload=payload,
                response_format=response_format,
                tools=tools,
            )
            if not data.get("choices"):
                raise ValueError("No choices in response")
            return data["choices"][0]["message"]
        except Exception as exc:
            logger.error("LLM API error: %s", exc)
            return {"content": self._get_conservative_default()}

    # Kept for backwards-compat with any callers not yet migrated.
    async def _call_llm(self, prompt: str) -> str:
        """Legacy shim: call _call_llm_structured and return content string."""
        msg = await self._call_llm_structured(
            system=(
                "You are an expert memecoin analyst. "
                "You provide honest, data-driven assessments. "
                "You err on the side of caution. "
                "You respond ONLY with valid JSON."
            ),
            user=prompt,
        )
        return msg.get("content") or self._get_conservative_default()

    def _parse_entry_data(self, data: dict, token: TokenMetadata) -> TokenAnalysis:
        """Build a TokenAnalysis from a validated dict (post schema parse)."""
        decision_map = {
            "STRONG_BUY": AIDecision.STRONG_BUY,
            "BUY": AIDecision.BUY,
            "SKIP": AIDecision.SKIP,
            "AVOID": AIDecision.AVOID,
        }
        decision = decision_map.get(
            str(data.get("decision", "SKIP")).upper(), AIDecision.SKIP
        )
        return TokenAnalysis(
            decision=decision,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", "No reasoning provided"),
            risk_score=float(data.get("risk_score", 5.0)),
            suggested_buy_amount_sol=data.get("suggested_buy_amount_sol"),
            suggested_stop_loss_pct=data.get("suggested_stop_loss_pct"),
            suggested_take_profit_pct=data.get("suggested_take_profit_pct"),
            social_score=data.get("social_score"),
            liquidity_score=data.get("liquidity_score"),
            holder_score=data.get("holder_score"),
            timing_score=data.get("timing_score"),
            red_flags=data.get("red_flags", []),
            green_flags=data.get("green_flags", []),
            model_used=self.model,
        )

    def _conservative_analysis(self, reason: str = "AI analysis failed") -> TokenAnalysis:
        """Return a SKIP analysis used as the safe fallback."""
        red_flags = [reason] if reason == "AI analysis failed" else ["AI analysis failed", reason]
        return TokenAnalysis(
            decision=AIDecision.SKIP,
            confidence=0.0,
            reasoning=reason,
            risk_score=10.0,
            red_flags=red_flags,
            model_used=self.model,
        )

    # Kept for backwards-compat with callers that pass a raw string.
    def _parse_llm_response(self, response: str, token: TokenMetadata) -> TokenAnalysis:
        """Legacy shim: extract JSON from a string and build TokenAnalysis."""
        from fenrir.ai.structured_output import extract_json
        data = extract_json(response)
        if not isinstance(data, dict):
            logger.error("Failed to parse LLM response (no JSON dict found)")
            return self._conservative_analysis("Failed to parse LLM response")
        return self._parse_entry_data(data, token)

    def _get_conservative_default(self) -> str:
        """Return a conservative default response."""
        return json.dumps(
            {
                "decision": "SKIP",
                "confidence": 0.0,
                "risk_score": 10.0,
                "reasoning": "Unable to perform AI analysis. Defaulting to SKIP for safety.",
                "red_flags": ["AI analysis unavailable"],
                "green_flags": [],
                "suggested_buy_amount_sol": 0.01,
                "suggested_stop_loss_pct": 50,
                "suggested_take_profit_pct": 50,
            }
        )

    async def evaluate_exit_strategy(
        self,
        token_mint: str,
        entry_price: float,
        current_price: float,
        hold_time_minutes: int,
        current_pnl_pct: float,
    ) -> dict:
        """
        Use AI to determine optimal exit strategy.
        Should we hold, take profit, or cut losses?
        """
        prompt = f"""You are evaluating an open position in a memecoin.

# POSITION DETAILS
Token: {token_mint}
Entry Price: ${entry_price:.10f}
Current Price: ${current_price:.10f}
Hold Time: {hold_time_minutes} minutes
Current P&L: {current_pnl_pct:+.1f}%

# YOUR TASK
Recommend: HOLD, TAKE_PROFIT, or EXIT
Provide reasoning.

Respond with JSON:
{{
  "action": "HOLD|TAKE_PROFIT|EXIT",
  "reasoning": "Why?",
  "urgency": 0.0-1.0
}}
"""

        response = await self._call_llm(prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except Exception:
            return {
                "action": "HOLD",
                "reasoning": "Unable to get AI recommendation",
                "urgency": 0.5,
            }

    async def analyze_token_launch_with_context(
        self,
        token_metadata: TokenMetadata,
        memory_context: str = "",
        portfolio_context: str = "",
        risk_context: str = "",
        market_conditions: dict | None = None,
    ) -> TokenAnalysis:
        """
        Enhanced analysis with session memory context.

        Uses structured output (json_schema) as the primary path and
        parse_or_sanitize() as the fallback — replacing the fragile
        find("{") approach. Provider capability degradation is handled
        transparently by ProviderResilientCaller.
        """
        if not self._caller:
            await self.initialize()

        # Build base analysis prompt
        base_prompt = self._build_analysis_prompt(token_metadata, market_conditions)

        # Inject context blocks before "# YOUR TASK"
        context_sections = []
        if memory_context:
            context_sections.append(memory_context)
        if portfolio_context:
            context_sections.append(portfolio_context)
        if risk_context:
            context_sections.append(f"# RISK CONTEXT\n{risk_context}")

        context_injection = "\n\n".join(context_sections)
        if context_injection and "# YOUR TASK" in base_prompt:
            enhanced_prompt = base_prompt.replace(
                "# YOUR TASK", f"{context_injection}\n\n# YOUR TASK"
            )
        else:
            enhanced_prompt = base_prompt

        system = (
            "You are an expert memecoin analyst. "
            "You provide honest, data-driven assessments. "
            "You err on the side of caution. "
            "You respond ONLY with valid JSON."
        )
        message = await self._call_llm_structured(
            system=system,
            user=enhanced_prompt,
            response_format=build_response_format("entry_analysis", ENTRY_ANALYSIS_SCHEMA),
        )

        data = await parse_or_sanitize(
            message=message,
            schema=ENTRY_ANALYSIS_SCHEMA,
            schema_name="entry_analysis",
            session=self.session,
            api_key=self.api_key,
            sanitize_model=self.SANITIZE_MODEL,
        )

        if not isinstance(data, dict):
            analysis = self._conservative_analysis("entry parse+sanitize failed")
        else:
            analysis = self._parse_entry_data(data, token_metadata)

        self.predictions.append(
            {
                "token_mint": token_metadata.token_mint,
                "analysis": analysis,
                "actual_performance": None,
                "had_context": True,
            }
        )
        return analysis

    async def evaluate_exit_strategy_with_context(
        self,
        token_mint: str,
        entry_price: float,
        current_price: float,
        peak_price: float,
        hold_time_minutes: int,
        current_pnl_pct: float,
        mechanical_trigger: str | None = None,
        memory_context: str = "",
    ) -> dict:
        """
        Enhanced exit evaluation with session memory and mechanical trigger context.

        Uses EXIT_ANALYSIS_SCHEMA for structured output and parse_or_sanitize()
        fallback. Includes exit_plan in the schema so Claude can encode its own
        continuation contract (Nocturne cooldown pattern).

        If a mechanical trigger has fired, Claude may choose OVERRIDE_HOLD to
        keep the position open with an explanation. Hard stop-loss floor is
        enforced by ClaudeBrain.evaluate_exit(), not here.
        """
        if not self._caller:
            await self.initialize()

        drawdown_pct = ((peak_price - current_price) / peak_price * 100) if peak_price > 0 else 0.0

        trigger_section = ""
        if mechanical_trigger:
            trigger_section = f"""
# MECHANICAL TRIGGER FIRED
A rule-based exit trigger has activated: {mechanical_trigger}
You may OVERRIDE this trigger and recommend HOLD if you believe the token
has strong momentum or the trigger is premature. Set action to OVERRIDE_HOLD.
If you override, explain clearly in exit_plan why the position should stay open
and include a cooldown_until timestamp if appropriate.
"""

        prompt = f"""You are evaluating an open memecoin position.

# POSITION DETAILS
Token: {token_mint}
Entry Price: ${entry_price:.10f}
Current Price: ${current_price:.10f}
Peak Price: ${peak_price:.10f}
Drawdown from Peak: {drawdown_pct:.1f}%
Hold Time: {hold_time_minutes} minutes
Current P&L: {current_pnl_pct:+.1f}%
{trigger_section}
{memory_context}

# YOUR TASK
Recommend one of: HOLD | TAKE_PROFIT | EXIT | OVERRIDE_HOLD

exit_plan MUST encode:
  1. Your hold conditions (what would make you exit on the next cycle)
  2. Optional: "cooldown_until: <ISO timestamp UTC>" if re-evaluation should
     be suppressed for a period. Example:
     "Hold while drawdown < 40%. cooldown_until: 2025-10-19T16:00Z"
"""

        system = (
            "You are an expert memecoin analyst evaluating exit timing. "
            "Respond ONLY with valid JSON matching the provided schema."
        )
        message = await self._call_llm_structured(
            system=system,
            user=prompt,
            response_format=build_response_format("exit_analysis", EXIT_ANALYSIS_SCHEMA),
        )

        data = await parse_or_sanitize(
            message=message,
            schema=EXIT_ANALYSIS_SCHEMA,
            schema_name="exit_analysis",
            session=self.session,
            api_key=self.api_key,
            sanitize_model=self.SANITIZE_MODEL,
        )

        if not isinstance(data, dict):
            if mechanical_trigger:
                return {
                    "action": "EXIT",
                    "reasoning": f"AI parse failure; deferring to trigger: {mechanical_trigger}",
                    "urgency": 0.7,
                    "exit_plan": "",
                }
            return {"action": "HOLD", "reasoning": "AI parse failure", "urgency": 0.5, "exit_plan": ""}

        return data

    async def evaluate_exits_batched(
        self,
        context_json: str,
        active_addresses: list[str],
    ) -> dict:
        """
        Evaluate exit actions for ALL open positions in a single LLM call.

        This is Nocturne's batched-asset pattern applied to FENRIR exits:
        instead of one LLM round-trip per position, the full portfolio state
        is packed into one structured context and evaluated at once.

        Benefits:
          - Cross-position awareness: Claude can trade off risk across the book
          - 1/N API calls vs per-position exit evaluation
          - exit_plan replay: prior AI continuation contracts are injected so
            Claude can check whether its own conditions have been invalidated

        Args:
            context_json:     JSON string from context_builder.build_batched_exit_context()
            active_addresses: Ordered list of token addresses in the context
                              (positions in AI cooldown are pre-filtered out)

        Returns:
            {
                "reasoning": "<global portfolio reasoning>",
                "exit_decisions": [
                    {
                        "token_address": "...",
                        "action": "HOLD|EXIT|TAKE_PROFIT|OVERRIDE_HOLD",
                        "reasoning": "...",
                        "urgency": 0.0-1.0,
                        "exit_plan": "<continuation contract, may include cooldown_until>",
                    },
                    ...
                ]
            }
            On total failure returns a safe all-HOLD dict.
        """
        if not self._caller:
            await self.initialize()

        # Build a schema locked to the exact token addresses in this batch
        batch_schema = {
            **BATCHED_EXIT_SCHEMA,
            "properties": {
                **BATCHED_EXIT_SCHEMA["properties"],
                "exit_decisions": {
                    **BATCHED_EXIT_SCHEMA["properties"]["exit_decisions"],
                    "items": {
                        **BATCHED_EXIT_SCHEMA["properties"]["exit_decisions"]["items"],
                        "properties": {
                            **BATCHED_EXIT_SCHEMA["properties"]["exit_decisions"]["items"]["properties"],
                            "token_address": {
                                "type": "string",
                                "enum": active_addresses,
                            },
                        },
                    },
                },
            },
        }

        system = (
            "You are FENRIR's portfolio risk manager evaluating exit timing for "
            "all open memecoin positions. Analyze cross-position risk together. "
            "For each position, write an exit_plan that encodes your hold conditions "
            "and any self-imposed cooldown: 'cooldown_until: <ISO timestamp UTC>'. "
            "Honour prior_ai_exit_plan unless its conditions are clearly invalidated. "
            "Respond ONLY with valid JSON matching the provided schema."
        )

        message = await self._call_llm_structured(
            system=system,
            user=context_json,
            response_format=build_response_format("batched_exit", batch_schema),
        )

        data = await parse_or_sanitize(
            message=message,
            schema=batch_schema,
            schema_name="batched_exit",
            session=self.session,
            api_key=self.api_key,
            sanitize_model=self.SANITIZE_MODEL,
        )

        if not isinstance(data, dict) or not data.get("exit_decisions"):
            logger.warning(
                "evaluate_exits_batched: parse+sanitize failed; "
                "defaulting all %d positions to HOLD",
                len(active_addresses),
            )
            return {
                "reasoning": "batch parse failure",
                "exit_decisions": [
                    {
                        "token_address": addr,
                        "action": "HOLD",
                        "reasoning": "batch parse failure",
                        "urgency": 0.0,
                        "exit_plan": "",
                    }
                    for addr in active_addresses
                ],
            }

        return data

    def track_prediction_outcome(
        self, token_mint: str, actual_pnl_pct: float, hold_time_minutes: int
    ):
        """
        Track actual outcome of an AI prediction.
        Used to evaluate AI performance over time.
        """
        for pred in self.predictions:
            if pred["token_mint"] == token_mint:
                pred["actual_performance"] = {
                    "pnl_pct": actual_pnl_pct,
                    "hold_time_minutes": hold_time_minutes,
                    "timestamp": datetime.now().isoformat(),
                }
                break

    def get_ai_performance_report(self) -> dict:
        """
        Generate report on AI decision quality.
        How accurate were the predictions?
        """
        evaluated = [p for p in self.predictions if p["actual_performance"]]

        if not evaluated:
            return {"total_predictions": len(self.predictions), "evaluated": 0, "accuracy": 0.0}

        # Calculate metrics
        strong_buys = [p for p in evaluated if p["analysis"].decision == AIDecision.STRONG_BUY]
        buys = [p for p in evaluated if p["analysis"].decision == AIDecision.BUY]
        skips = [p for p in evaluated if p["analysis"].decision == AIDecision.SKIP]

        # Were the STRONG_BUY recommendations profitable?
        strong_buy_success = 0
        if strong_buys:
            profitable_strong_buys = [
                p for p in strong_buys if p["actual_performance"]["pnl_pct"] > 0
            ]
            strong_buy_success = len(profitable_strong_buys) / len(strong_buys)

        # Average return on BUY recommendations
        avg_buy_return = 0.0
        if buys:
            avg_buy_return = sum(p["actual_performance"]["pnl_pct"] for p in buys) / len(buys)

        return {
            "total_predictions": len(self.predictions),
            "evaluated": len(evaluated),
            "strong_buy_count": len(strong_buys),
            "strong_buy_success_rate": strong_buy_success,
            "buy_count": len(buys),
            "avg_buy_return_pct": avg_buy_return,
            "skip_count": len(skips),
        }


# ═══════════════════════════════════════════════════════════════════════════
#                              EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════


async def example_usage():
    """Demonstrate AI trading analyst."""
    print("🐺 FENRIR - AI Trading Analyst")
    print("=" * 70)

    # Example token metadata
    TokenMetadata(
        token_mint="ABC123TOKEN",  # noqa: S106
        name="Wolf Finance",
        symbol="WOLF",
        description="The most alpha memecoin on Solana",
        twitter="https://twitter.com/wolf_finance",
        telegram="https://t.me/wolf_finance",
        initial_liquidity_sol=10.0,
        current_market_cap_sol=50.0,
        holder_count=250,
        top_10_holder_pct=25.0,
        creator_address="CREATOR123",
        creator_previous_launches=3,
        creator_success_rate=0.67,
        twitter_followers=1500,
        telegram_members=800,
    )

    print("\n💡 AI Decision Engine Ready")
    print("   Set OPENROUTER_API_KEY to enable")


if __name__ == "__main__":
    asyncio.run(example_usage())
