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

        # Track AI performance (bounded to prevent unbounded growth)
        self.predictions: deque = deque(maxlen=200)

    async def initialize(self):
        """Initialize HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

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

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return response."""
        if not self.session:
            await self.initialize()

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert memecoin analyst. You provide honest, data-driven assessments. You err on the side of caution. You respond ONLY with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 2000,
        }

        try:
            async with self.session.post(
                self.OPENROUTER_API, headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")

                data = await response.json()

                if "choices" not in data or not data["choices"]:
                    raise Exception("No response from LLM")

                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error("LLM API error: %s", e)
            # Return conservative default
            return self._get_conservative_default()

    def _parse_llm_response(self, response: str, token: TokenMetadata) -> TokenAnalysis:
        """Parse LLM response into TokenAnalysis object."""
        try:
            # Extract JSON from response (may be wrapped in markdown)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Map decision string to enum
            decision_map = {
                "STRONG_BUY": AIDecision.STRONG_BUY,
                "BUY": AIDecision.BUY,
                "SKIP": AIDecision.SKIP,
                "AVOID": AIDecision.AVOID,
            }

            decision = decision_map.get(data.get("decision", "SKIP").upper(), AIDecision.SKIP)

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

        except Exception as e:
            logger.error("Failed to parse LLM response: %s", e)
            logger.debug("Raw response: %s", response)

            # Return conservative default
            return TokenAnalysis(
                decision=AIDecision.SKIP,
                confidence=0.0,
                reasoning=f"Failed to parse AI analysis: {str(e)}",
                risk_score=10.0,
                red_flags=["AI analysis failed"],
                model_used=self.model,
            )

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
        Enhanced analysis that includes session memory context.

        Injects recent decision history, portfolio state, and risk warnings
        into the prompt so Claude can reason with full situational awareness.
        """
        if not self.session:
            await self.initialize()

        # Build base analysis prompt
        base_prompt = self._build_analysis_prompt(token_metadata, market_conditions)

        # Build context injection block
        context_sections = []
        if memory_context:
            context_sections.append(memory_context)
        if portfolio_context:
            context_sections.append(portfolio_context)
        if risk_context:
            context_sections.append(f"# RISK CONTEXT\n{risk_context}")

        context_injection = "\n\n".join(context_sections)

        # Inject context before "# YOUR TASK"
        if context_injection and "# YOUR TASK" in base_prompt:
            enhanced_prompt = base_prompt.replace(
                "# YOUR TASK", f"{context_injection}\n\n# YOUR TASK"
            )
        else:
            enhanced_prompt = base_prompt

        # Call LLM and parse response
        response = await self._call_llm(enhanced_prompt)
        analysis = self._parse_llm_response(response, token_metadata)

        # Track prediction
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

        If a mechanical trigger has fired, Claude can choose to OVERRIDE it
        (let the position ride) or confirm the exit.
        """
        drawdown_pct = ((peak_price - current_price) / peak_price * 100) if peak_price > 0 else 0.0

        trigger_section = ""
        if mechanical_trigger:
            trigger_section = f"""
# MECHANICAL TRIGGER FIRED
A rule-based exit trigger has activated: {mechanical_trigger}
You may OVERRIDE this trigger and recommend HOLD if you believe the token
has strong momentum or the trigger is premature.
If you override, explain clearly why the position should stay open.
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
Recommend one of:
- HOLD: Keep the position open
- TAKE_PROFIT: Exit with gains
- EXIT: Close the position (stop loss or risk management)

Respond ONLY with valid JSON:
{{
  "action": "HOLD|TAKE_PROFIT|EXIT",
  "reasoning": "Brief explanation",
  "urgency": 0.0-1.0
}}
"""
        response = await self._call_llm(prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found")
            return json.loads(response[json_start:json_end])
        except Exception:
            # On parse failure, defer to mechanical trigger if present
            if mechanical_trigger:
                return {
                    "action": "EXIT",
                    "reasoning": f"AI parse failure; deferring to trigger: {mechanical_trigger}",
                    "urgency": 0.7,
                }
            return {"action": "HOLD", "reasoning": "AI parse failure", "urgency": 0.5}

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

    # Market conditions

    # Initialize AI analyst (requires API key)
    # analyst = AITradingAnalyst(
    #     api_key="your-openrouter-api-key",
    #     model="anthropic/claude-sonnet-4"
    # )
    #
    # analysis = await analyst.analyze_token_launch(token, market)
    #
    # print(f"\n🤖 AI Analysis:")
    # print(f"   Decision: {analysis.decision.value.upper()}")
    # print(f"   Confidence: {analysis.confidence:.0%}")
    # print(f"   Risk Score: {analysis.risk_score:.1f}/10")
    # print(f"\n   Reasoning: {analysis.reasoning}")
    #
    # if analysis.red_flags:
    #     print(f"\n   🚩 Red Flags:")
    #     for flag in analysis.red_flags:
    #         print(f"      - {flag}")
    #
    # if analysis.green_flags:
    #     print(f"\n   ✅ Green Flags:")
    #     for flag in analysis.green_flags:
    #         print(f"      - {flag}")

    print("\n💡 AI Decision Engine Ready")
    print("   Set OPENROUTER_API_KEY to enable")


if __name__ == "__main__":
    asyncio.run(example_usage())
