#!/usr/bin/env python3
"""
FENRIR - Core Configuration

Trading modes, bot configuration, and environment management.
"""

from __future__ import annotations

import importlib  # FIX 1: was used below but never imported
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from fenrir.filters import MarketFilterConfig, SecurityFilterConfig
    from fenrir.trading.tx_config import TxConfigManager


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var; return ``default`` when unset/empty."""
    raw = os.getenv(name, "")
    if raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Parse a float env var; return ``default`` when unset or malformed."""
    raw = os.getenv(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Dependency check (single location for the whole package)
_deps_missing = False
try:
    # Import solana and solders modules dynamically to avoid static analysis
    # issues in environments where the packages aren't installed.
    _solana_rpc = importlib.import_module("solana.rpc.async_api")
    AsyncClient = _solana_rpc.AsyncClient  # type: ignore

    _solders_kp = importlib.import_module("solders.keypair")
    Keypair = _solders_kp.Keypair  # noqa: F401
except ImportError:
    print("Missing dependencies. Install with:")
    print("   pip install solana solders base58 aiohttp python-dotenv websockets")
    _deps_missing = True


_dotenv_loaded = False


def _ensure_dotenv() -> None:
    """Load .env file exactly once, on first call."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv()
        _dotenv_loaded = True


class TradingMode(Enum):
    """Trading modes reflecting different risk appetites."""

    SIMULATION = "simulation"  # Paper trading - no real txs
    CONSERVATIVE = "conservative"  # Small positions, strict stops
    AGGRESSIVE = "aggressive"  # Larger positions, wider stops
    DEGEN = "degen"  # YOLO mode - maximum risk


# ── Mode-specific trading presets ──────────────────────────────────────
TRADING_PRESETS: dict[TradingMode, dict] = {  # type: ignore[type-arg]
    TradingMode.SIMULATION: {
        "buy_amount_sol": 0.1,
        "max_slippage_bps": 500,
        "stop_loss_pct": 25.0,
        "take_profit_pct": 100.0,
        "trailing_stop_pct": 15.0,
        "max_position_age_minutes": 30,
        "min_initial_liquidity_sol": 0.0,
        "max_initial_market_cap_sol": 80.0,
        "priority_fee_lamports": 500_000,
        "ai_entry_timeout_seconds": 12.0,
        "ai_exit_timeout_seconds": 10.0,
        "ai_min_confidence_to_buy": 0.6,
        "ai_temperature": 0.3,
    },
    TradingMode.CONSERVATIVE: {
        "buy_amount_sol": 0.05,
        "max_slippage_bps": 300,
        "stop_loss_pct": 15.0,
        "take_profit_pct": 75.0,
        "trailing_stop_pct": 10.0,
        "max_position_age_minutes": 20,
        "min_initial_liquidity_sol": 5.0,
        "max_initial_market_cap_sol": 60.0,
        "priority_fee_lamports": 500_000,
        "ai_entry_timeout_seconds": 12.0,
        "ai_exit_timeout_seconds": 10.0,
        "ai_min_confidence_to_buy": 0.75,
        "ai_temperature": 0.2,
    },
    TradingMode.AGGRESSIVE: {
        "buy_amount_sol": 0.2,
        "max_slippage_bps": 800,
        "stop_loss_pct": 30.0,
        "take_profit_pct": 200.0,
        "trailing_stop_pct": 20.0,
        "max_position_age_minutes": 45,
        "min_initial_liquidity_sol": 2.0,
        "max_initial_market_cap_sol": 120.0,
        "priority_fee_lamports": 1_000_000,
        "ai_entry_timeout_seconds": 10.0,
        "ai_exit_timeout_seconds": 8.0,
        "ai_min_confidence_to_buy": 0.55,
        "ai_temperature": 0.4,
    },
    TradingMode.DEGEN: {
        "buy_amount_sol": 0.5,
        "max_slippage_bps": 1500,
        "stop_loss_pct": 50.0,
        "take_profit_pct": 500.0,
        "trailing_stop_pct": 30.0,
        "max_position_age_minutes": 60,
        "min_initial_liquidity_sol": 1.0,
        "max_initial_market_cap_sol": 200.0,
        "priority_fee_lamports": 2_000_000,
        "ai_entry_timeout_seconds": 10.0,
        "ai_exit_timeout_seconds": 8.0,
        "ai_min_confidence_to_buy": 0.4,
        "ai_temperature": 0.5,
    },
}


@dataclass
class BotConfig:
    """
    The brain of FENRIR. Every parameter tuned for elegance and control.
    Think of this as the DNA of your trading strategy.
    """

    # Network & Connection
    rpc_url: str = ""
    ws_url: str = ""

    # Wallet Configuration
    private_key: str = ""  # Base58 encoded

    # Pump.fun Program
    pumpfun_program: str = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

    # Trading Parameters
    mode: TradingMode = TradingMode.SIMULATION
    buy_amount_sol: float = 0.1  # SOL per trade
    max_slippage_bps: int = 500  # 5% max slippage

    # Risk Management
    stop_loss_pct: float = 25.0  # Exit if down 25%
    take_profit_pct: float = 100.0  # Exit if up 100%
    trailing_stop_pct: float = 15.0  # Trail by 15% from peak
    max_position_age_minutes: int = 30  # Memecoins pump/dump in minutes

    # Launch Criteria
    min_initial_liquidity_sol: float = 0.0  # Minimum SOL in bonding curve (0 = snipe at creation)
    max_initial_market_cap_sol: float = 80.0  # Don't buy if already too big

    # Execution Settings
    priority_fee_lamports: int = 500_000  # 0.0005 SOL for competitive inclusion
    use_jito: bool = False  # MEV protection via Jito bundles
    jito_tip_lamports: int = 10000  # Tip for Jito validators
    # Use per-strategy transaction profiles (fenrir.trading.tx_config) instead
    # of the flat priority_fee/slippage/jito settings above. Off by default;
    # consumed by the engine once the pipeline-wiring PR lands.
    tx_profiles_enabled: bool = False

    # Monitoring
    websocket_enabled: bool = True  # Real-time vs polling
    poll_interval_seconds: float = 2.0  # If WebSocket fails

    # Strategy selection & budgets
    # Which registered strategies to run (IDs from fenrir.strategies.STRATEGY_REGISTRY).
    # Defaults to the sniper only; the signal strategies are opt-in. Consumed by
    # the bot's strategy loader once the pipeline-wiring PR lands.
    enabled_strategies: list[str] = field(default_factory=lambda: ["sniper"])
    sniper_daily_budget_sol: float = 0.0  # 0 = auto (10 × buy_amount_sol)
    # Absolute cap on net live SOL exposure across ALL strategies — the master
    # safety valve above per-strategy budgets. 0 = disabled. Env: GLOBAL_DAILY_SOL_LIMIT.
    global_daily_sol_limit: float = 0.0

    # Dynamic priority fee: when on, size the compute-unit price from recent
    # on-chain prioritization fees (percentile) for competitive inclusion,
    # clamped to [priority_fee_lamports, max_priority_fee_lamports]. Opt-in so
    # default behavior (the flat priority_fee_lamports) is unchanged.
    dynamic_priority_fee_enabled: bool = False
    max_priority_fee_lamports: int = 5_000_000  # 0.005 SOL ceiling

    # ── Market scanner (multi-tier discovery beyond fresh launches) ────
    # Off by default. When on, periodically pulls Jupiter trending tokens,
    # buckets them into mid/large-cap tiers by USD market cap, and feeds each to
    # the AI (which trades them via the Jupiter buy path). Env: MARKET_SCANNER_ENABLED.
    market_scanner_enabled: bool = False
    scanner_interval_seconds: float = 120.0
    scanner_categories: list[str] = field(default_factory=lambda: ["toptraded", "toporganicscore"])
    scanner_interval_window: str = "24h"  # Jupiter interval: 5m|1h|6h|24h
    mid_cap_min_usd: float = 200_000.0  # mid tier: [mid_min, large_min)
    large_cap_min_usd: float = 1_000_000.0  # large tier: >= large_min
    scanner_min_liquidity_usd: float = 50_000.0
    scanner_min_organic_score: float = 0.0  # 0 = disabled
    scanner_require_verified: bool = False
    scanner_max_candidates_per_cycle: int = 5
    scanner_cooldown_minutes: float = 30.0
    scanner_daily_budget_sol: float = 0.0  # 0 = auto (5 × buy_amount_sol)
    scanner_max_positions: int = 3
    # DexScreener "boosts" as an extra discovery source: memes actively paying for
    # visibility. Off by default; enriched via DexScreener pair data and tiered by
    # mcap like the Jupiter feed. Env: SCANNER_DEX_BOOSTS_ENABLED.
    scanner_dex_boosts_enabled: bool = False
    scanner_dex_timeout_seconds: float = 6.0

    # ── Pre-trade filters (fenrir.filters) ─────────────────────────────
    # Security hard-gate: mint/freeze authority, LP burn, holder concentration.
    security_filter_enabled: bool = False
    security_require_mint_revoked: bool = True
    security_require_freeze_revoked: bool = True
    security_min_lp_burned_pct: float = 90.0
    security_max_top10_holder_pct: float = 30.0
    security_fail_open_on_holder_fetch_error: bool = False
    # RugCheck (rugcheck.xyz) keyless third-party risk score — optional add-on to
    # the security gate. Off by default; rejects tokens above a normalised risk
    # score (0-100, lower = safer) or with a "danger"-level risk.
    rugcheck_enabled: bool = False
    rugcheck_max_score: float = 40.0
    rugcheck_reject_on_danger: bool = True
    rugcheck_fail_open: bool = True
    # Optional Helius key for enriched holder data (falls back to public RPC).
    helius_api_key: str = ""
    # Two-tier DexScreener market-condition filter.
    market_filter_enabled: bool = False
    market_fail_open_on_fetch_error: bool = True

    # Experimental: on-chain pump.fun→Raydium migration feed (WebSocket only).
    # Off by default; the migration parser is not verified offline. Feeds the
    # migration_snipe strategy when enabled.
    migration_feed_enabled: bool = False

    # AI Integration - Claude Brain
    ai_analysis_enabled: bool = True  # Master switch for AI decisions
    ai_api_key: str = ""
    ai_model: str = "anthropic/claude-haiku-4-5"
    ai_provider: str = "openrouter"  # "openrouter" or "anthropic_direct"
    ai_entry_timeout_seconds: float = 12.0  # Max wait for entry analysis
    ai_exit_timeout_seconds: float = 10.0  # Max wait for exit evaluation
    ai_exit_eval_interval_seconds: float = 60.0  # Proactive exit check cadence
    ai_min_confidence_to_buy: float = 0.6  # Minimum confidence for BUY
    ai_memory_size: int = 15  # Rolling decision history size
    ai_memory_resume: bool = False  # Rebuild session memory from the audit chain on startup
    ai_temperature: float = 0.3  # LLM temperature (lower = more conservative)
    ai_fallback_to_rules: bool = True  # Auto-buy on AI failure/timeout?
    ai_dynamic_position_sizing: bool = False  # Let AI set buy amount?

    # Local Model Backend
    ai_local_model_enabled: bool = False
    ai_local_model_url: str = "http://localhost:8000/v1/chat/completions"
    ai_local_model_name: str = "fenrir-brain"

    # Logging — default is INFO; override via LOG_LEVEL env var
    # NOTE: do NOT use os.getenv() here — .env isn't loaded yet at class
    # definition time. The __post_init__ method handles the env override
    # after _ensure_dotenv() has run.
    log_level: str = "INFO"
    log_file: str = "fenrir_bot.log"

    def __post_init__(self) -> None:
        """Fill env-based defaults after dataclass init."""
        _ensure_dotenv()

        if not self.rpc_url:
            self.rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        if not self.ws_url:
            self.ws_url = os.getenv("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com")
        if not self.private_key:
            self.private_key = os.getenv("WALLET_PRIVATE_KEY", "")

        # FIX 2: ai_analysis_enabled controllable from .env
        env_ai_enabled = os.getenv("AI_ANALYSIS_ENABLED", "")
        if env_ai_enabled != "":
            self.ai_analysis_enabled = env_ai_enabled.lower() == "true"

        # FIX 3: support both OpenRouter and direct Anthropic API keys
        if not self.ai_api_key:
            self.ai_api_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv(
                "ANTHROPIC_API_KEY", ""
            )

        # Model is env-overridable so the operator can switch providers/models
        # (e.g. off an out-of-credit BYOK model) without a code change.
        env_ai_model = os.getenv("AI_MODEL", "").strip()
        if env_ai_model:
            self.ai_model = env_ai_model

        if not self.ai_local_model_url:
            self.ai_local_model_url = os.getenv(
                "AI_LOCAL_MODEL_URL", "http://localhost:8000/v1/chat/completions"
            )
        if not self.ai_local_model_name:
            self.ai_local_model_name = os.getenv("AI_LOCAL_MODEL_NAME", "fenrir-brain")
        if not self.ai_local_model_enabled:
            self.ai_local_model_enabled = os.getenv("AI_LOCAL_MODEL_ENABLED", "").lower() == "true"

        # FIX 4: LOG_LEVEL controllable from .env
        # Always defer to env var — .env is now loaded so os.getenv is reliable
        env_log_level = os.getenv("LOG_LEVEL", "").upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if env_log_level in valid_levels:
            self.log_level = env_log_level

        # ── Strategy selection ─────────────────────────────────────────
        env_strategies = os.getenv("ENABLED_STRATEGIES", "")
        if env_strategies.strip():
            self.enabled_strategies = [s.strip() for s in env_strategies.split(",") if s.strip()]

        # ── Pre-trade filters ──────────────────────────────────────────
        self.security_filter_enabled = _env_bool(
            "SECURITY_FILTER_ENABLED", self.security_filter_enabled
        )
        self.security_require_mint_revoked = _env_bool(
            "SECURITY_REQUIRE_MINT_REVOKED", self.security_require_mint_revoked
        )
        self.security_require_freeze_revoked = _env_bool(
            "SECURITY_REQUIRE_FREEZE_REVOKED", self.security_require_freeze_revoked
        )
        self.security_min_lp_burned_pct = _env_float(
            "SECURITY_MIN_LP_BURNED_PCT", self.security_min_lp_burned_pct
        )
        self.security_max_top10_holder_pct = _env_float(
            "SECURITY_MAX_TOP10_HOLDER_PCT", self.security_max_top10_holder_pct
        )
        self.security_fail_open_on_holder_fetch_error = _env_bool(
            "SECURITY_FAIL_OPEN_ON_HOLDER_FETCH_ERROR",
            self.security_fail_open_on_holder_fetch_error,
        )
        self.rugcheck_enabled = _env_bool("RUGCHECK_ENABLED", self.rugcheck_enabled)
        self.rugcheck_max_score = _env_float("RUGCHECK_MAX_SCORE", self.rugcheck_max_score)
        self.rugcheck_reject_on_danger = _env_bool(
            "RUGCHECK_REJECT_ON_DANGER", self.rugcheck_reject_on_danger
        )
        self.rugcheck_fail_open = _env_bool("RUGCHECK_FAIL_OPEN", self.rugcheck_fail_open)
        if not self.helius_api_key:
            self.helius_api_key = os.getenv("HELIUS_API_KEY", "")
        self.market_filter_enabled = _env_bool("MARKET_FILTER_ENABLED", self.market_filter_enabled)
        self.market_fail_open_on_fetch_error = _env_bool(
            "MARKET_FAIL_OPEN_ON_FETCH_ERROR", self.market_fail_open_on_fetch_error
        )
        self.migration_feed_enabled = _env_bool(
            "MIGRATION_FEED_ENABLED", self.migration_feed_enabled
        )

        # ── Execution ──────────────────────────────────────────────────
        self.tx_profiles_enabled = _env_bool("TX_PROFILES_ENABLED", self.tx_profiles_enabled)

        # ── Risk limits ────────────────────────────────────────────────
        env_global = os.getenv("GLOBAL_DAILY_SOL_LIMIT", "")
        if env_global:
            try:
                self.global_daily_sol_limit = float(env_global)
            except ValueError:
                pass

        # ── Execution tuning ───────────────────────────────────────────
        self.dynamic_priority_fee_enabled = _env_bool(
            "DYNAMIC_PRIORITY_FEE_ENABLED", self.dynamic_priority_fee_enabled
        )
        env_maxfee = os.getenv("MAX_PRIORITY_FEE_LAMPORTS", "")
        if env_maxfee:
            try:
                self.max_priority_fee_lamports = int(env_maxfee)
            except ValueError:
                pass

        # ── Market scanner ─────────────────────────────────────────────
        self.market_scanner_enabled = _env_bool(
            "MARKET_SCANNER_ENABLED", self.market_scanner_enabled
        )
        self.scanner_interval_seconds = _env_float(
            "SCANNER_INTERVAL_SECONDS", self.scanner_interval_seconds
        )
        self.mid_cap_min_usd = _env_float("MID_CAP_MIN_USD", self.mid_cap_min_usd)
        self.large_cap_min_usd = _env_float("LARGE_CAP_MIN_USD", self.large_cap_min_usd)
        self.scanner_min_liquidity_usd = _env_float(
            "SCANNER_MIN_LIQUIDITY_USD", self.scanner_min_liquidity_usd
        )
        env_scan_strats = os.getenv("SCANNER_CATEGORIES", "")
        if env_scan_strats:
            self.scanner_categories = [s.strip() for s in env_scan_strats.split(",") if s.strip()]
        self.scanner_dex_boosts_enabled = _env_bool(
            "SCANNER_DEX_BOOSTS_ENABLED", self.scanner_dex_boosts_enabled
        )

    @classmethod
    def from_mode(cls, mode: TradingMode, **overrides) -> BotConfig:  # type: ignore[override]
        """Create a BotConfig pre-tuned for a specific trading mode."""
        preset = TRADING_PRESETS.get(mode, {})  # type: ignore[arg-type]
        merged = {**preset, "mode": mode, **overrides}
        return cls(**merged)

    # ── Component config builders ──────────────────────────────────────
    # Translate the flat BotConfig surface into the self-contained component
    # configs from fenrir.filters / fenrir.trading. Lazy imports keep the
    # config module light and avoid import cycles.

    def build_security_filter_config(self) -> SecurityFilterConfig:
        """Build a SecurityFilterConfig from the security_* fields."""
        from fenrir.filters import SecurityFilterConfig

        return SecurityFilterConfig(
            require_mint_revoked=self.security_require_mint_revoked,
            require_freeze_revoked=self.security_require_freeze_revoked,
            min_lp_burned_pct=self.security_min_lp_burned_pct,
            max_top10_holder_pct=self.security_max_top10_holder_pct,
            fail_open_on_holder_fetch_error=self.security_fail_open_on_holder_fetch_error,
            rugcheck_enabled=self.rugcheck_enabled,
            rugcheck_max_score=self.rugcheck_max_score,
            rugcheck_reject_on_danger=self.rugcheck_reject_on_danger,
            rugcheck_fail_open=self.rugcheck_fail_open,
        )

    def build_market_filter_config(self) -> MarketFilterConfig:
        """Build a MarketFilterConfig from the market_* fields (tier defaults kept)."""
        from fenrir.filters import MarketFilterConfig

        return MarketFilterConfig(
            enabled=self.market_filter_enabled,
            fail_open_on_fetch_error=self.market_fail_open_on_fetch_error,
        )

    def build_tx_config_manager(self) -> TxConfigManager:
        """Build a TxConfigManager bound to this config's RPC URL."""
        from fenrir.trading.tx_config import TxConfigManager

        return TxConfigManager(rpc_url=self.rpc_url)

    def __repr__(self) -> str:
        """Redact sensitive fields to prevent accidental secret leakage in logs."""
        pk_display = "***" if self.private_key else "(empty)"
        api_key_display = "***" if self.ai_api_key else "(empty)"
        return (
            f"BotConfig(mode={self.mode.value}, "
            f"rpc_url='{self.rpc_url[:30]}...', "
            f"private_key='{pk_display}', "
            f"ai_api_key='{api_key_display}', "
            f"buy_amount_sol={self.buy_amount_sol})"
        )

    def validate(self) -> list[str]:
        """Validation that feels like a caring mentor checking your homework."""
        errors = []

        if not self.rpc_url:
            errors.append("RPC URL required - get one from QuickNode or Helius")

        if self.rpc_url and not self.rpc_url.startswith("https://"):
            if not self.rpc_url.startswith("http://127.0.0.1") and not self.rpc_url.startswith(
                "http://localhost"
            ):
                errors.append("RPC URL must use HTTPS (plaintext HTTP leaks wallet data)")

        if self.mode != TradingMode.SIMULATION and not self.private_key:
            errors.append("Private key required for live trading (set WALLET_PRIVATE_KEY)")

        if self.buy_amount_sol <= 0:
            errors.append("Buy amount must be positive")

        if self.stop_loss_pct >= 100:
            errors.append("Stop loss too aggressive - you'd lose everything")

        if self.take_profit_pct <= 0:
            errors.append("Take profit must be positive - you want gains, right?")

        if self.priority_fee_lamports < 0:
            errors.append("Priority fee must be non-negative")

        if self.use_jito and self.jito_tip_lamports <= 0:
            errors.append("Jito tip must be positive when use_jito=True")

        if self.ai_analysis_enabled and not self.ai_api_key:
            errors.append(
                "AI API key required when ai_analysis_enabled=True "
                "(set OPENROUTER_API_KEY or ANTHROPIC_API_KEY)"
            )

        if not (0.0 <= self.ai_temperature <= 2.0):
            errors.append("AI temperature must be between 0.0 and 2.0")

        if not (0.0 <= self.ai_min_confidence_to_buy <= 1.0):
            errors.append("AI min confidence must be between 0.0 and 1.0")

        if self.ai_local_model_enabled and self.ai_analysis_enabled:
            if not self.ai_local_model_url.startswith("http"):
                errors.append("ai_local_model_url must be a valid HTTP URL")
            if not self.ai_local_model_name:
                errors.append("ai_local_model_name must be set when ai_local_model_enabled=True")

        # ── Strategy selection ─────────────────────────────────────────
        if not self.enabled_strategies:
            errors.append("At least one strategy must be enabled (enabled_strategies is empty)")
        else:
            # Lazy import to avoid a config <-> strategies import cycle.
            from fenrir.strategies import STRATEGY_REGISTRY

            unknown = [s for s in self.enabled_strategies if s not in STRATEGY_REGISTRY]
            if unknown:
                available = ", ".join(sorted(STRATEGY_REGISTRY))
                errors.append(
                    f"Unknown strategy id(s): {', '.join(unknown)} (available: {available})"
                )

        # ── Pre-trade filters ──────────────────────────────────────────
        if not (0.0 <= self.security_min_lp_burned_pct <= 100.0):
            errors.append("security_min_lp_burned_pct must be between 0 and 100")

        if not (0.0 <= self.security_max_top10_holder_pct <= 100.0):
            errors.append("security_max_top10_holder_pct must be between 0 and 100")

        if not (0.0 <= self.rugcheck_max_score <= 100.0):
            errors.append("rugcheck_max_score must be between 0 and 100")

        return errors
