#!/usr/bin/env python3
"""
FENRIR - Transaction Execution Configuration

Per-strategy slippage, priority fee, and MEV protection settings.
Implements the optimal configuration for each trading style:

  Ultra-Early Sniping:  20-40% slippage | Fixed fee 0.01-0.05 SOL | Jito ON
  Fast Momentum:         5-10% slippage  | Dynamic Turbo           | Jito ON
  Established Swing:     1-3%  slippage  | Dynamic Medium          | Jito optional

Key design decisions:
  - Sniping always uses FIXED fees (no historical data for dynamic calculation)
  - Dynamic fees use percentile-based network congestion targeting
  - Jito bundles enabled by default for any trade with slippage > 5%
  - Fee caps prevent runaway costs on congested networks
  - All values overridable via .env for live tuning without code changes

Self-contained: this module maps strategy IDs (including the signal strategies
added in the strategies package) to execution profiles. It is not yet wired
into the trading engine — that reconciliation lands in a later PR.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("FENRIR.TxConfig")

# Lamports per SOL
LAMPORTS_PER_SOL = 1_000_000_000


class FeeMode(Enum):
    """Priority fee calculation mode."""

    FIXED = "fixed"  # Absolute SOL amount — best for sniping
    DYNAMIC = "dynamic"  # Based on live network congestion percentile


class DynamicFeePreset(Enum):
    """Dynamic fee aggressiveness presets."""

    MEDIUM = "medium"  # ~50th percentile — swing trading
    AGGRESSIVE = "aggressive"  # ~75th percentile — momentum trading
    TURBO = "turbo"  # ~90th percentile — fast breakouts


@dataclass
class SlippageConfig:
    """Slippage settings for a trading style."""

    # Slippage tolerance as a percentage (e.g. 20.0 = 20%)
    pct: float
    # Maximum allowed slippage (hard cap, overrides pct if exceeded)
    max_pct: float

    @property
    def bps(self) -> int:
        """BPS representation for Jupiter/Raydium (pct * 100)."""
        return int(self.pct * 100)

    @property
    def max_bps(self) -> int:
        return int(self.max_pct * 100)


@dataclass
class PriorityFeeConfig:
    """Priority fee settings for a trading style."""

    mode: FeeMode = FeeMode.DYNAMIC
    # Fixed fee amount in SOL (used when mode=FIXED)
    fixed_fee_sol: float = 0.005
    # Dynamic fee preset (used when mode=DYNAMIC)
    dynamic_preset: DynamicFeePreset = DynamicFeePreset.AGGRESSIVE
    # Maximum fee cap in SOL (prevents runaway costs)
    max_fee_sol: float = 0.05
    # Minimum fee floor in SOL
    min_fee_sol: float = 0.0005

    @property
    def fixed_fee_lamports(self) -> int:
        return int(self.fixed_fee_sol * LAMPORTS_PER_SOL)

    @property
    def max_fee_lamports(self) -> int:
        return int(self.max_fee_sol * LAMPORTS_PER_SOL)

    @property
    def min_fee_lamports(self) -> int:
        return int(self.min_fee_sol * LAMPORTS_PER_SOL)


@dataclass
class JitoConfig:
    """Jito MEV protection bundle settings."""

    enabled: bool = True
    # Tip paid to Jito validators in SOL
    tip_sol: float = 0.001
    # Jito block engine endpoint
    block_engine_url: str = "https://mainnet.block-engine.jito.wtf/api/v1/bundles"

    @property
    def tip_lamports(self) -> int:
        return int(self.tip_sol * LAMPORTS_PER_SOL)


@dataclass
class TxProfile:
    """
    Complete transaction execution profile for one trading style.
    Bundles slippage + priority fee + MEV protection into one config object.
    """

    name: str
    slippage: SlippageConfig
    priority_fee: PriorityFeeConfig
    jito: JitoConfig
    # Human-readable description
    description: str = ""

    def summary(self) -> str:
        fee_str = (
            f"Fixed {self.priority_fee.fixed_fee_sol:.4f} SOL"
            if self.priority_fee.mode == FeeMode.FIXED
            else f"Dynamic/{self.priority_fee.dynamic_preset.value}"
        )
        jito_str = f"Jito ON ({self.jito.tip_sol:.4f} SOL tip)" if self.jito.enabled else "Jito OFF"
        return f"{self.name}: slippage={self.slippage.pct:.0f}% | fee={fee_str} | {jito_str}"


# ── Pre-built profiles ────────────────────────────────────────────────


def ultra_early_snipe_profile() -> TxProfile:
    """
    Ultra-Early Sniping (new launches, migration snipe).
    20-40% slippage, fixed fee 0.02 SOL, Jito ON.

    Why fixed fee: No historical transaction data exists for a brand-new
    pool. Dynamic fee calculators have no baseline and send minimum fees
    that get ignored during launch congestion.
    """
    return TxProfile(
        name="UltraEarlySnipe",
        description="New launches and pump.fun migrations. Max speed, max fee.",
        slippage=SlippageConfig(pct=25.0, max_pct=40.0),
        priority_fee=PriorityFeeConfig(
            mode=FeeMode.FIXED,
            fixed_fee_sol=0.02,
            max_fee_sol=0.05,
            min_fee_sol=0.01,
        ),
        jito=JitoConfig(enabled=True, tip_sol=0.002),
    )


def fast_momentum_profile() -> TxProfile:
    """
    Fast Momentum Trading (10m-2h old tokens, reversal plays).
    5-10% slippage, dynamic turbo fee, Jito ON.

    Pool has some transaction history so dynamic fee calculation works.
    Turbo targets ~90th percentile of recent successful blocks.
    """
    return TxProfile(
        name="FastMomentum",
        description="Momentum breakouts and reversal plays. Speed + MEV protection.",
        slippage=SlippageConfig(pct=8.0, max_pct=12.0),
        priority_fee=PriorityFeeConfig(
            mode=FeeMode.DYNAMIC,
            dynamic_preset=DynamicFeePreset.TURBO,
            max_fee_sol=0.01,
            min_fee_sol=0.003,
        ),
        jito=JitoConfig(enabled=True, tip_sol=0.001),
    )


def volume_scalp_profile() -> TxProfile:
    """
    Volume Anomaly Scalping (6h+ old, high vol/mcap ratio).
    5-8% slippage, dynamic aggressive fee, Jito ON.
    """
    return TxProfile(
        name="VolumeScalp",
        description="Volume anomaly scalps. Balanced speed and cost.",
        slippage=SlippageConfig(pct=6.0, max_pct=10.0),
        priority_fee=PriorityFeeConfig(
            mode=FeeMode.DYNAMIC,
            dynamic_preset=DynamicFeePreset.AGGRESSIVE,
            max_fee_sol=0.008,
            min_fee_sol=0.003,
        ),
        jito=JitoConfig(enabled=True, tip_sol=0.001),
    )


def swing_trade_profile() -> TxProfile:
    """
    Established Swing Trading (1d+ old, narrative plays).
    1-3% slippage, dynamic medium fee, Jito optional.

    Older tokens with deep liquidity: price is stable enough that low
    slippage rarely causes missed entries. Save fees here.
    """
    return TxProfile(
        name="SwingTrade",
        description="Narrative and swing plays on established tokens. Low cost.",
        slippage=SlippageConfig(pct=2.0, max_pct=5.0),
        priority_fee=PriorityFeeConfig(
            mode=FeeMode.DYNAMIC,
            dynamic_preset=DynamicFeePreset.MEDIUM,
            max_fee_sol=0.002,
            min_fee_sol=0.0005,
        ),
        jito=JitoConfig(enabled=False, tip_sol=0.0005),
    )


# ── Strategy → Profile mapping ────────────────────────────────────────

# Maps strategy IDs to their default transaction profiles
STRATEGY_TX_PROFILES: dict[str, TxProfile] = {
    "migration_snipe": ultra_early_snipe_profile(),
    "sniper": ultra_early_snipe_profile(),  # existing sniper strategy
    "reversal": fast_momentum_profile(),
    "graduation": fast_momentum_profile(),  # existing graduation strategy
    "volume_anomaly": volume_scalp_profile(),
    "narrative_tracker": swing_trade_profile(),
    "default": fast_momentum_profile(),
}


class TxConfigManager:
    """
    Manages transaction execution profiles per strategy.
    Supports runtime overrides via environment variables and
    dynamic fee calculation via Helius priority fee API.
    """

    def __init__(self, rpc_url: str = "") -> None:
        self.rpc_url = rpc_url
        # Deep-copy so per-manager env overrides never mutate the shared
        # module-level profile singletons (would otherwise leak across
        # managers and across test cases).
        self._profiles = copy.deepcopy(STRATEGY_TX_PROFILES)
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to profiles.
        Allows live tuning without code changes.

        Supported env vars:
          SNIPE_SLIPPAGE_PCT        — slippage for snipe profile
          SNIPE_PRIORITY_FEE_SOL    — fixed fee for snipe profile
          MOMENTUM_SLIPPAGE_PCT     — slippage for momentum profile
          SWING_SLIPPAGE_PCT        — slippage for swing profile
          JITO_ENABLED              — enable/disable Jito globally (true/false)
          JITO_TIP_SOL              — Jito tip amount in SOL
        """
        # Snipe profile overrides
        snipe = self._profiles["migration_snipe"]
        if slip := os.getenv("SNIPE_SLIPPAGE_PCT"):
            snipe.slippage.pct = float(slip)
            self._profiles["sniper"].slippage.pct = float(slip)
            logger.info(f"Snipe slippage overridden: {slip}%")
        if fee := os.getenv("SNIPE_PRIORITY_FEE_SOL"):
            snipe.priority_fee.fixed_fee_sol = float(fee)
            self._profiles["sniper"].priority_fee.fixed_fee_sol = float(fee)
            logger.info(f"Snipe priority fee overridden: {fee} SOL")

        # Momentum overrides
        momentum = self._profiles["reversal"]
        if slip := os.getenv("MOMENTUM_SLIPPAGE_PCT"):
            momentum.slippage.pct = float(slip)
            self._profiles["graduation"].slippage.pct = float(slip)
            logger.info(f"Momentum slippage overridden: {slip}%")

        # Swing overrides
        swing = self._profiles["narrative_tracker"]
        if slip := os.getenv("SWING_SLIPPAGE_PCT"):
            swing.slippage.pct = float(slip)
            logger.info(f"Swing slippage overridden: {slip}%")

        # Global Jito overrides
        jito_enabled_str = os.getenv("JITO_ENABLED", "")
        if jito_enabled_str:
            jito_enabled = jito_enabled_str.lower() == "true"
            for profile in self._profiles.values():
                profile.jito.enabled = jito_enabled
            logger.info(f"Jito globally {'enabled' if jito_enabled else 'disabled'}")

        if jito_tip := os.getenv("JITO_TIP_SOL"):
            for profile in self._profiles.values():
                if profile.jito.enabled:
                    profile.jito.tip_sol = float(jito_tip)
            logger.info(f"Jito tip overridden: {jito_tip} SOL")

    def get_profile(self, strategy_id: str) -> TxProfile:
        """Get the transaction profile for a strategy."""
        return self._profiles.get(strategy_id, self._profiles["default"])

    async def get_priority_fee_lamports(
        self,
        strategy_id: str,
        session: Any = None,
    ) -> int:
        """
        Calculate the actual priority fee in lamports for a strategy.

        For FIXED mode: returns the configured fixed amount directly.
        For DYNAMIC mode: queries Helius/Solana for recent fee percentiles
        and returns the appropriate tier, capped at max_fee_sol.
        """
        profile = self.get_profile(strategy_id)
        fee_config = profile.priority_fee

        if fee_config.mode == FeeMode.FIXED:
            return fee_config.fixed_fee_lamports

        # Dynamic fee: query recent prioritization fees
        try:
            fee_lamports = await self._fetch_dynamic_fee(fee_config.dynamic_preset, session)
            # Apply floor and cap
            fee_lamports = max(fee_lamports, fee_config.min_fee_lamports)
            fee_lamports = min(fee_lamports, fee_config.max_fee_lamports)
            return fee_lamports
        except Exception as e:
            logger.warning(f"Dynamic fee fetch failed, using min floor: {e}")
            return fee_config.min_fee_lamports

    async def _fetch_dynamic_fee(
        self,
        preset: DynamicFeePreset,
        session: Any = None,
    ) -> int:
        """
        Fetch recent prioritization fees from Solana RPC and return
        the fee at the target percentile based on preset.

        Percentile targets:
          MEDIUM:     50th percentile
          AGGRESSIVE: 75th percentile
          TURBO:      90th percentile
        """
        percentile_map = {
            DynamicFeePreset.MEDIUM: 50,
            DynamicFeePreset.AGGRESSIVE: 75,
            DynamicFeePreset.TURBO: 90,
        }
        target_pct = percentile_map[preset]

        if not self.rpc_url or session is None:
            # Fallback: hardcoded reasonable values per preset
            fallbacks = {
                DynamicFeePreset.MEDIUM: 100_000,  # 0.0001 SOL
                DynamicFeePreset.AGGRESSIVE: 500_000,  # 0.0005 SOL
                DynamicFeePreset.TURBO: 1_000_000,  # 0.001 SOL
            }
            return fallbacks[preset]

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getRecentPrioritizationFees",
            "params": [[]],
        }

        async with session.post(self.rpc_url, json=payload, timeout=3.0) as resp:
            data = await resp.json()

        fees = data.get("result", [])
        if not fees:
            return 500_000  # Safe fallback

        # Extract fee values and find target percentile
        fee_values = sorted(
            int(entry.get("prioritizationFee", 0))
            for entry in fees
            if entry.get("prioritizationFee", 0) > 0
        )

        if not fee_values:
            return 500_000

        idx = int(len(fee_values) * target_pct / 100)
        idx = min(idx, len(fee_values) - 1)
        return fee_values[idx]

    def log_all_profiles(self) -> None:
        """Log all configured profiles for startup verification."""
        logger.info("Transaction profiles configured:")
        seen = set()
        for strategy_id, profile in self._profiles.items():
            if profile.name not in seen:
                logger.info(f"  [{strategy_id}] {profile.summary()}")
                seen.add(profile.name)

    def get_slippage_bps(self, strategy_id: str) -> int:
        """Get slippage in basis points for a strategy."""
        return self.get_profile(strategy_id).slippage.bps

    def jito_enabled(self, strategy_id: str) -> bool:
        """Check if Jito is enabled for a strategy."""
        return self.get_profile(strategy_id).jito.enabled

    def jito_tip_lamports(self, strategy_id: str) -> int:
        """Get Jito tip in lamports for a strategy."""
        return self.get_profile(strategy_id).jito.tip_lamports
