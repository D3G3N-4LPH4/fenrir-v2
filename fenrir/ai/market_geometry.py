#!/usr/bin/env python3
"""
FENRIR - Market Geometry Analyzer

Pre-entry analysis pass that derives strategy parameters from each token's
specific market structure, replacing static TRADING_PRESETS with live analysis.

Inspired by OBLITERATUS's InformedAbliterationPipeline — the key insight that
running analysis *before* intervention and auto-configuring parameters based
on what the analysis finds produces far better outcomes than brute-forcing
with fixed settings.

The four geometry axes (mapped from OBLITERATUS to trading):

  OBLITERATUS axis              → FENRIR equivalent
  ──────────────────────────────────────────────────────────────────────────
  Alignment Imprint Detection   → Creator fingerprinting (rug vs legit pattern)
  Concept Cone Geometry         → Is momentum organic or coordinated? (one
                                  mechanism or many concurrent forces?)
  Cross-Layer Alignment         → Liquidity depth across the bonding curve
  Defense Robustness            → Sell wall + bot defense patterns
  ──────────────────────────────────────────────────────────────────────────

Output: a GeometryReport with:
  - Derived TradeParams (position size, slippage, stop, take-profit)
  - Confidence-adjusted AI instructions
  - Plain-language summary injected as strategy_context into ClaudeBrain

Usage:
    # In bot.py _evaluate_and_execute():
    analyzer = MarketGeometryAnalyzer()

    report = analyzer.analyze(token_data, strategy)
    strategy_context = strategy.get_ai_context() + "\\n\\n" + report.ai_context_block

    should_buy, analysis, buy_amount = await self.claude_brain.evaluate_entry(
        token_data,
        strategy_positions,
        strategy_context=strategy_context,
        historical_context=historical_context,
    )

    # Use derived params instead of strategy defaults
    if should_buy:
        params = report.derived_params  # Auto-tuned for this specific token
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

from fenrir.strategies.base import TradeParams

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#                           GEOMETRY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CreatorImprint:
    """
    Creator fingerprint — equivalent to OBLITERATUS's Alignment Imprint Detection.
    Identifies the creator's behavioral pattern from on-chain indicators.
    """

    pattern: str = "unknown"    # "quick_exit", "holder", "dev_whale", "unknown"
    risk_multiplier: float = 1.0  # Higher = riskier creator pattern
    notes: list[str] = field(default_factory=list)


@dataclass
class MomentumGeometry:
    """
    Momentum structure — equivalent to OBLITERATUS's Concept Cone Geometry.
    Is price action driven by one organic mechanism, or coordinated by many?
    """

    is_organic: bool = True
    cone_width: float = 1.0      # Low = concentrated (likely coordinated); High = diffuse (organic)
    momentum_score: float = 0.5  # 0.0 (weak) → 1.0 (strong)
    notes: list[str] = field(default_factory=list)


@dataclass
class LiquidityDepth:
    """
    Liquidity structure across the bonding curve — equivalent to Cross-Layer Alignment.
    How much depth is there and how is it distributed?
    """

    depth_score: float = 0.5     # 0.0 (thin) → 1.0 (deep)
    migration_pct: float = 0.0   # % through bonding curve
    is_thin: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class DefenseRobustness:
    """
    Sell wall and bot defense analysis — equivalent to OBLITERATUS's Ouroboros/robustness eval.
    Are there patterns suggesting the token will resist exits?
    """

    has_sell_walls: bool = False
    has_bot_defense: bool = False
    exit_risk_score: float = 0.0   # 0.0 (easy exit) → 1.0 (hard exit)
    notes: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#                           GEOMETRY REPORT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeometryReport:
    """
    Complete pre-entry analysis report for a single token.
    Contains derived TradeParams and the AI context block.
    """

    # The four geometry axes
    creator_imprint: CreatorImprint = field(default_factory=CreatorImprint)
    momentum_geometry: MomentumGeometry = field(default_factory=MomentumGeometry)
    liquidity_depth: LiquidityDepth = field(default_factory=LiquidityDepth)
    defense_robustness: DefenseRobustness = field(default_factory=DefenseRobustness)

    # Derived composite scores (0.0 → 1.0, higher = better)
    overall_quality_score: float = 0.5
    overall_risk_score: float = 0.5

    # Auto-configured trade parameters
    derived_params: TradeParams | None = None

    # AI context block (injected into ClaudeBrain.evaluate_entry strategy_context)
    ai_context_block: str = ""

    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    token_address: str = ""


# ═══════════════════════════════════════════════════════════════════════════
#                           ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class MarketGeometryAnalyzer:
    """
    Derives strategy parameters from each token's market geometry.

    Equivalent to OBLITERATUS's InformedAbliterationPipeline:
    runs four analysis modules, then auto-configures the downstream
    intervention (TradeParams) based on what each module found.

    All analysis is synchronous and pure-Python — no network calls.
    Fast enough to run on every detected token before AI evaluation.
    """

    def __init__(self, base_params: TradeParams | None = None):
        """
        Args:
            base_params: Fallback params if geometry analysis yields no signal.
                         Typically the active strategy's get_trade_params().
        """
        self.base_params = base_params or TradeParams()
        self._analyses_run: int = 0

    def analyze(
        self,
        token_data: dict,
        strategy=None,
    ) -> GeometryReport:
        """
        Run all four geometry analysis modules and derive TradeParams.

        Args:
            token_data: Dict from PumpFunMonitor (same format as evaluate_entry)
            strategy: Active TradingStrategy (for base params override)

        Returns:
            GeometryReport with derived_params and ai_context_block
        """
        self._analyses_run += 1

        base = strategy.get_trade_params() if strategy else self.base_params
        report = GeometryReport(token_address=token_data.get("token_address", ""))

        # ── Module 1: Creator Imprint ──────────────────────────────────────
        report.creator_imprint = self._analyze_creator(token_data)

        # ── Module 2: Momentum Geometry ───────────────────────────────────
        report.momentum_geometry = self._analyze_momentum(token_data)

        # ── Module 3: Liquidity Depth ─────────────────────────────────────
        report.liquidity_depth = self._analyze_liquidity(token_data)

        # ── Module 4: Defense Robustness ──────────────────────────────────
        report.defense_robustness = self._analyze_defense(token_data)

        # ── Composite Scores ──────────────────────────────────────────────
        report.overall_quality_score = self._compute_quality(report)
        report.overall_risk_score = self._compute_risk(report)

        # ── Derive TradeParams ────────────────────────────────────────────
        report.derived_params = self._derive_params(report, base)

        # ── Build AI Context Block ────────────────────────────────────────
        report.ai_context_block = self._build_ai_context(report, token_data)

        logger.debug(
            f"Geometry [{token_data.get('symbol', '???')}]: "
            f"quality={report.overall_quality_score:.2f} "
            f"risk={report.overall_risk_score:.2f} "
            f"creator={report.creator_imprint.pattern}"
        )

        return report

    # ──────────────────────────────────────────────────────────────────────
    #  MODULE 1 — CREATOR IMPRINT
    # ──────────────────────────────────────────────────────────────────────

    def _analyze_creator(self, token_data: dict) -> CreatorImprint:
        """
        Fingerprint the creator from available on-chain indicators.

        Without historical DB access, we use proxy signals from token_data:
        - Creator address absence → unknown pattern
        - Token name/description patterns → style signals
        - Launch timing and initial liquidity ratio → behavior signals
        """
        imprint = CreatorImprint()
        notes = []

        creator = token_data.get("creator")
        if not creator:
            imprint.pattern = "unknown"
            imprint.risk_multiplier = 1.2
            notes.append("No creator address — elevated risk")
            imprint.notes = notes
            return imprint

        # Liquidity ratio: very low initial liquidity vs mcap = aggressive launch
        liquidity = token_data.get("initial_liquidity_sol", 0)
        mcap = token_data.get("market_cap_sol", 0)
        if mcap > 0:
            liq_ratio = liquidity / mcap
            if liq_ratio < 0.05:
                imprint.pattern = "quick_exit"
                imprint.risk_multiplier = 1.5
                notes.append(f"Very thin liquidity ratio ({liq_ratio:.1%}) — typical quick-exit pattern")
            elif liq_ratio < 0.15:
                imprint.pattern = "dev_whale"
                imprint.risk_multiplier = 1.2
                notes.append(f"Low liquidity ratio ({liq_ratio:.1%}) — possible dev whale concentration")
            else:
                imprint.pattern = "holder"
                imprint.risk_multiplier = 0.85
                notes.append(f"Healthy liquidity ratio ({liq_ratio:.1%}) — looks like a holder mindset")

        # Name/description quality signals
        name = token_data.get("name", "")
        desc = token_data.get("description", "") or ""

        if len(name) < 3 or name.upper() == name:
            notes.append("Generic/lazy token name — correlates with quick-exit creators")
            imprint.risk_multiplier = min(imprint.risk_multiplier * 1.1, 2.0)

        if len(desc) > 100:
            notes.append("Detailed description — creator invested effort (positive signal)")
            imprint.risk_multiplier = max(imprint.risk_multiplier * 0.9, 0.5)

        # Social links as commitment signal
        has_twitter = bool(token_data.get("twitter"))
        has_telegram = bool(token_data.get("telegram"))
        social_count = sum([has_twitter, has_telegram])
        if social_count == 0:
            notes.append("No social links — minimal creator commitment")
            imprint.risk_multiplier = min(imprint.risk_multiplier * 1.15, 2.0)
        elif social_count == 2:
            notes.append("Twitter + Telegram present — creator built community infrastructure")
            imprint.risk_multiplier = max(imprint.risk_multiplier * 0.9, 0.5)

        imprint.notes = notes
        return imprint

    # ──────────────────────────────────────────────────────────────────────
    #  MODULE 2 — MOMENTUM GEOMETRY
    # ──────────────────────────────────────────────────────────────────────

    def _analyze_momentum(self, token_data: dict) -> MomentumGeometry:
        """
        Analyze whether momentum is organic or coordinated.

        Organic: gradual buildup, moderate initial metrics
        Coordinated: unusually high immediate liquidity + low mcap
        (suggests pre-positioned wallets ready to dump on retail)
        """
        geo = MomentumGeometry()
        notes = []

        liquidity = token_data.get("initial_liquidity_sol", 0)
        mcap = token_data.get("market_cap_sol", 0)

        if liquidity <= 0 or mcap <= 0:
            geo.momentum_score = 0.3
            geo.is_organic = True
            geo.cone_width = 1.0
            notes.append("Insufficient data for momentum analysis")
            geo.notes = notes
            return geo

        # Abnormally high liquidity at launch = coordinated pre-positioning
        if liquidity > 20:
            geo.is_organic = False
            geo.cone_width = 0.3  # Narrow cone = concentrated forces
            geo.momentum_score = 0.6  # High momentum but suspicious
            notes.append(f"Very high launch liquidity ({liquidity:.1f} SOL) — likely coordinated")
        elif liquidity > 10:
            geo.is_organic = True  # Could go either way
            geo.cone_width = 0.7
            geo.momentum_score = 0.7
            notes.append(f"Strong launch liquidity ({liquidity:.1f} SOL)")
        elif liquidity >= 3:
            geo.is_organic = True
            geo.cone_width = 1.0
            geo.momentum_score = 0.5
            notes.append(f"Normal launch liquidity ({liquidity:.1f} SOL)")
        else:
            geo.is_organic = True
            geo.cone_width = 1.2
            geo.momentum_score = 0.3
            notes.append(f"Low launch liquidity ({liquidity:.1f} SOL) — thin")

        # Low mcap + high liquidity = pump setup (coordinated)
        if mcap < 10 and liquidity > 5:
            geo.is_organic = False
            geo.cone_width = 0.4
            notes.append("Low mcap + high liquidity ratio = possible coordinated pump")

        geo.notes = notes
        return geo

    # ──────────────────────────────────────────────────────────────────────
    #  MODULE 3 — LIQUIDITY DEPTH
    # ──────────────────────────────────────────────────────────────────────

    def _analyze_liquidity(self, token_data: dict) -> LiquidityDepth:
        """Analyze bonding curve depth and migration progress."""
        depth = LiquidityDepth()
        notes = []

        liquidity = token_data.get("initial_liquidity_sol", 0)
        curve_state = token_data.get("bonding_curve_state")

        if curve_state:
            migration_pct = curve_state.get_migration_progress()
            depth.migration_pct = migration_pct

            if migration_pct > 70:
                depth.depth_score = 0.85
                notes.append(f"Deep curve ({migration_pct:.0f}% migrated) — high liquidity depth")
            elif migration_pct > 40:
                depth.depth_score = 0.6
                notes.append(f"Mid-curve ({migration_pct:.0f}% migrated)")
            else:
                depth.depth_score = 0.35
                notes.append(f"Early curve ({migration_pct:.0f}% migrated) — thin depth")

        # Absolute liquidity floor
        if liquidity < 2:
            depth.is_thin = True
            depth.depth_score = min(depth.depth_score, 0.3)
            notes.append(f"Thin absolute liquidity ({liquidity:.2f} SOL) — exit risk")
        elif liquidity >= 5:
            depth.depth_score = max(depth.depth_score, 0.5)
            notes.append(f"Adequate liquidity ({liquidity:.2f} SOL)")

        depth.notes = notes
        return depth

    # ──────────────────────────────────────────────────────────────────────
    #  MODULE 4 — DEFENSE ROBUSTNESS
    # ──────────────────────────────────────────────────────────────────────

    def _analyze_defense(self, token_data: dict) -> DefenseRobustness:
        """
        Estimate exit difficulty from available signals.

        Without real-time order book data, we use structural proxies:
        - Very low liquidity + high mcap = likely sell walls guarding price
        - Known pump.fun bot patterns in token metadata
        """
        defense = DefenseRobustness()
        notes = []

        liquidity = token_data.get("initial_liquidity_sol", 0)
        mcap = token_data.get("market_cap_sol", 0)

        # High mcap relative to liquidity = price is inflated, exits will be hard
        if mcap > 0 and liquidity > 0:
            price_inflation = mcap / liquidity
            if price_inflation > 20:
                defense.has_sell_walls = True
                defense.exit_risk_score = 0.8
                notes.append(
                    f"High mcap/liquidity ratio ({price_inflation:.1f}x) — "
                    "likely sell walls protecting inflated price"
                )
            elif price_inflation > 10:
                defense.exit_risk_score = 0.5
                notes.append(f"Moderate mcap/liquidity ratio ({price_inflation:.1f}x)")
            else:
                defense.exit_risk_score = 0.2
                notes.append(f"Healthy mcap/liquidity ratio ({price_inflation:.1f}x)")

        # Bot-like naming patterns (common in defensive pump setups)
        name = (token_data.get("name") or "").lower()
        bot_signals = ["safe", "guard", "shield", "protected", "locked"]
        if any(sig in name for sig in bot_signals):
            defense.has_bot_defense = True
            defense.exit_risk_score = min(defense.exit_risk_score + 0.2, 1.0)
            notes.append("Token name contains defense-signaling keywords — elevated exit risk")

        defense.notes = notes
        return defense

    # ──────────────────────────────────────────────────────────────────────
    #  COMPOSITE SCORING
    # ──────────────────────────────────────────────────────────────────────

    def _compute_quality(self, report: GeometryReport) -> float:
        """Weighted composite quality score (0.0 → 1.0, higher = better opportunity)."""
        # Creator imprint: low risk_multiplier = good creator
        creator_score = max(0.0, 1.0 - (report.creator_imprint.risk_multiplier - 0.5) / 1.5)

        # Momentum: organic + strong = best
        momentum_score = report.momentum_geometry.momentum_score
        if not report.momentum_geometry.is_organic:
            momentum_score *= 0.6  # Coordinated momentum is risky

        # Liquidity: deeper is better
        liquidity_score = report.liquidity_depth.depth_score
        if report.liquidity_depth.is_thin:
            liquidity_score *= 0.5

        # Defense: lower exit risk = better opportunity
        defense_score = 1.0 - report.defense_robustness.exit_risk_score

        # Weighted average
        quality = (
            creator_score * 0.30
            + momentum_score * 0.25
            + liquidity_score * 0.25
            + defense_score * 0.20
        )
        return round(min(max(quality, 0.0), 1.0), 3)

    def _compute_risk(self, report: GeometryReport) -> float:
        """Weighted composite risk score (0.0 → 1.0, higher = riskier)."""
        creator_risk = min((report.creator_imprint.risk_multiplier - 0.5) / 1.5, 1.0)
        momentum_risk = 0.7 if not report.momentum_geometry.is_organic else 0.2
        liquidity_risk = 0.8 if report.liquidity_depth.is_thin else 0.2
        defense_risk = report.defense_robustness.exit_risk_score

        risk = (
            creator_risk * 0.30
            + momentum_risk * 0.20
            + liquidity_risk * 0.25
            + defense_risk * 0.25
        )
        return round(min(max(risk, 0.0), 1.0), 3)

    # ──────────────────────────────────────────────────────────────────────
    #  PARAM DERIVATION
    # ──────────────────────────────────────────────────────────────────────

    def _derive_params(self, report: GeometryReport, base: TradeParams) -> TradeParams:
        """
        Auto-configure TradeParams from geometry analysis.

        This is the OBLITERATUS pattern: analysis auto-configures intervention.
        Each axis finding adjusts one or more parameters:

          Creator imprint  → position size (risky creator = smaller)
          Momentum geometry → slippage tolerance (coordinated = tighter)
          Liquidity depth   → stop loss (thin liquidity = tighter stop)
          Defense robustness → trailing stop (hard exit = tighter trail)
        """
        quality = report.overall_quality_score
        risk = report.overall_risk_score

        # ── Position size: scale with quality, shrink with risk ────────────
        size_multiplier = 0.5 + (quality * 1.0) - (risk * 0.5)
        size_multiplier = max(0.3, min(size_multiplier, 1.5))
        derived_buy = round(base.buy_amount_sol * size_multiplier, 4)

        # ── Slippage: coordinated momentum needs tighter tolerance ─────────
        if not report.momentum_geometry.is_organic:
            derived_slippage = max(int(base.max_slippage_bps * 0.7), 200)
        elif report.liquidity_depth.is_thin:
            # Thin liquidity paradoxically needs wider slippage to fill
            derived_slippage = min(int(base.max_slippage_bps * 1.3), 2000)
        else:
            derived_slippage = base.max_slippage_bps

        # ── Stop loss: thin liquidity = tighter stop ───────────────────────
        if report.liquidity_depth.is_thin:
            derived_stop = base.stop_loss_pct * 0.75  # Tighter stop for thin markets
        elif risk > 0.7:
            derived_stop = base.stop_loss_pct * 0.85
        else:
            derived_stop = base.stop_loss_pct

        # ── Trailing stop: hard exit environments need tighter trail ────────
        if report.defense_robustness.exit_risk_score > 0.6:
            derived_trailing = base.trailing_stop_pct * 0.7  # Tighter trail
        elif quality > 0.7:
            derived_trailing = base.trailing_stop_pct * 1.2  # Let winners run
        else:
            derived_trailing = base.trailing_stop_pct

        # ── Take profit: high quality + organic = let it run ─────────────
        if quality > 0.75 and report.momentum_geometry.is_organic:
            derived_tp = base.take_profit_pct * 1.3
        elif risk > 0.7:
            derived_tp = base.take_profit_pct * 0.8  # Take profits earlier on risky tokens
        else:
            derived_tp = base.take_profit_pct

        return TradeParams(
            buy_amount_sol=derived_buy,
            max_slippage_bps=int(derived_slippage),
            stop_loss_pct=round(derived_stop, 1),
            take_profit_pct=round(derived_tp, 1),
            trailing_stop_pct=round(derived_trailing, 1),
            max_position_age_minutes=base.max_position_age_minutes,
            priority_fee_lamports=base.priority_fee_lamports,
            ai_min_confidence=base.ai_min_confidence,
            ai_temperature=base.ai_temperature,
            ai_entry_timeout=base.ai_entry_timeout,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  AI CONTEXT BLOCK
    # ──────────────────────────────────────────────────────────────────────

    def _build_ai_context(self, report: GeometryReport, token_data: dict) -> str:
        """
        Build the strategy_context string injected into ClaudeBrain.evaluate_entry.

        This tells the AI what the geometry analysis found so it can factor
        in structural signals that aren't visible in raw token metadata.
        """
        symbol = token_data.get("symbol", "???")
        lines = [
            "## Pre-Entry Market Geometry Analysis",
            f"Token: ${symbol}",
            f"Overall Quality Score: {report.overall_quality_score:.0%}",
            f"Overall Risk Score: {report.overall_risk_score:.0%}",
            "",
            "### Creator Imprint",
            f"Pattern: {report.creator_imprint.pattern}",
            f"Risk multiplier: {report.creator_imprint.risk_multiplier:.2f}x",
        ]
        for note in report.creator_imprint.notes:
            lines.append(f"- {note}")

        lines += [
            "",
            "### Momentum Geometry",
            f"Organic: {'Yes' if report.momentum_geometry.is_organic else 'No — likely coordinated'}",
            f"Momentum score: {report.momentum_geometry.momentum_score:.0%}",
        ]
        for note in report.momentum_geometry.notes:
            lines.append(f"- {note}")

        lines += [
            "",
            "### Liquidity Depth",
            f"Depth score: {report.liquidity_depth.depth_score:.0%}",
            f"Thin: {'Yes — exit risk elevated' if report.liquidity_depth.is_thin else 'No'}",
        ]
        for note in report.liquidity_depth.notes:
            lines.append(f"- {note}")

        lines += [
            "",
            "### Defense Robustness (Exit Difficulty)",
            f"Exit risk: {report.defense_robustness.exit_risk_score:.0%}",
        ]
        for note in report.defense_robustness.notes:
            lines.append(f"- {note}")

        # Derived params summary
        p = report.derived_params
        if p:
            lines += [
                "",
                "### Auto-Derived Trade Parameters",
                f"Position size: {p.buy_amount_sol:.4f} SOL",
                f"Stop loss: {p.stop_loss_pct:.1f}%",
                f"Take profit: {p.take_profit_pct:.1f}%",
                f"Trailing stop: {p.trailing_stop_pct:.1f}%",
                f"Max slippage: {p.max_slippage_bps} bps",
            ]

        lines += [
            "",
            "Use this geometry analysis to inform your BUY/SKIP decision. "
            "A high risk score or non-organic momentum pattern should raise "
            "your conviction threshold. A high quality score with organic momentum "
            "justifies lower confidence threshold.",
        ]

        return "\n".join(lines)

    def get_stats(self) -> dict:
        return {"analyses_run": self._analyses_run}
