"""
FENRIR Test Suite - Shared Fixtures
"""

import pytest

from fenrir.config import BotConfig
from fenrir.core.positions import PositionManager
from fenrir.logger import FenrirLogger


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
