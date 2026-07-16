"""
FENRIR Test Suite - Shared Fixtures
"""

import pytest

from fenrir.config import BotConfig, _ensure_dotenv
from fenrir.core.positions import PositionManager
from fenrir.logger import FenrirLogger

# Every env var BotConfig reads in __post_init__. An operator's real .env is loaded
# into the environment, so without this the local suite silently tests THEIR config
# (e.g. BUY_AMOUNT_SOL=0.01 breaking preset assertions) and diverges from CI, which
# has no .env. Cleared for every test so BotConfig() is deterministic; tests that
# want a value set it explicitly via monkeypatch.setenv.
_CONFIG_ENV = [
    "AI_ANALYSIS_ENABLED",
    "AI_ENTRY_TIMEOUT_SECONDS",
    "AI_ESTABLISHED_BUY_THRESHOLD",
    "AI_EVALUATE_ALL_LAUNCHES",
    "AI_FALLBACK_TO_RULES",
    "AI_LOCAL_MODEL_ENABLED",
    "AI_LOCAL_MODEL_NAME",
    "AI_LOCAL_MODEL_URL",
    "AI_MODEL",
    "AI_MODEL_FALLBACKS",
    "AI_MULTI_AGENT_ENABLED",
    "BUY_AMOUNT_SOL",
    "DISCOVERY_CHAINS",
    "DISCOVERY_ENABLED",
    "DISCOVERY_FILTERS",
    "DISCOVERY_INTERVAL_SECONDS",
    "DISCOVERY_MIN_ALERT_SCORE",
    "DISCOVERY_SOLANA_CATEGORIES",
    "ENABLED_STRATEGIES",
    "GLOBAL_DAILY_SOL_LIMIT",
    "HELIUS_API_KEY",
    "LARGE_CAP_MIN_USD",
    "LOG_LEVEL",
    "MARKET_FILTER_ENABLED",
    "MAX_PRIORITY_FEE_LAMPORTS",
    "MID_CAP_MIN_USD",
    "OPENROUTER_API_KEY",
    "RUGCHECK_ENABLED",
    "RUGCHECK_FAIL_OPEN",
    "RUGCHECK_MAX_SCORE",
    "SCANNER_CATEGORIES",
    "SECURITY_FILTER_ENABLED",
    "SMART_MONEY_ENABLED",
    "SMART_MONEY_PRIORITY_WALLETS",
    "SMART_MONEY_WALLETS",
    "SOLANA_RPC_URL",
    "SOLANA_WS_URL",
    "TX_PROFILES_ENABLED",
    "WALLET_PRIVATE_KEY",
]


@pytest.fixture(autouse=True)
def _isolate_config_env(monkeypatch):
    """Make BotConfig() deterministic regardless of the operator's local .env.

    Order matters: BotConfig lazily loads .env on the FIRST construction
    (config._ensure_dotenv, guarded by a module flag). Clearing before that load
    happens is useless — load_dotenv would re-inject every var right back. So force
    the one-time load first, then clear; later _ensure_dotenv calls are no-ops.
    """
    _ensure_dotenv()
    for var in _CONFIG_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def bot_config():
    """Default bot configuration for tests."""
    return BotConfig()


@pytest.fixture
def logger(bot_config):
    """Logger instance for tests."""
    return FenrirLogger(bot_config)


@pytest.fixture
def position_manager(bot_config, logger):
    """Position manager for tests."""
    return PositionManager(bot_config, logger)
