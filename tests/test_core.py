#!/usr/bin/env python3
"""
FENRIR Trading Bot - Test Suite

Run with: pytest tests/test_core.py -v
Or: python tests/test_core.py
"""

from datetime import datetime

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.core.positions import Position, PositionManager
from fenrir.core.wallet import WalletManager
from fenrir.logger import FenrirLogger


class TestBotConfig:
    """Test configuration validation."""

    def test_default_config_valid(self):
        """Default config should be valid in simulation mode."""
        config = BotConfig()
        errors = config.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_missing_private_key_live_trading(self):
        """Live trading requires private key."""
        config = BotConfig(mode=TradingMode.CONSERVATIVE, private_key="")
        errors = config.validate()
        assert any("Private key" in err for err in errors)

    def test_invalid_stop_loss(self):
        """Stop loss >= 100% should fail."""
        config = BotConfig(stop_loss_pct=100.0)
        errors = config.validate()
        assert any("Stop loss" in err for err in errors)

    def test_invalid_take_profit(self):
        """Negative take profit should fail."""
        config = BotConfig(take_profit_pct=-10.0)
        errors = config.validate()
        assert any("Take profit" in err for err in errors)


class TestPosition:
    """Test position tracking and P&L calculations."""

    def test_position_pnl_profit(self):
        """Test profit calculation."""
        position = Position(
            token_address="TEST123",
            entry_time=datetime.now(),
            entry_price=0.000001,
            amount_tokens=1_000_000,
            amount_sol_invested=1.0,
            peak_price=0.000001,
        )

        position.update_price(0.000002)  # Price doubles

        assert position.get_pnl_percent() == 100.0
        assert position.get_pnl_sol() == 1.0

    def test_position_pnl_loss(self):
        """Test loss calculation."""
        position = Position(
            token_address="TEST123",
            entry_time=datetime.now(),
            entry_price=0.000001,
            amount_tokens=1_000_000,
            amount_sol_invested=1.0,
            peak_price=0.000001,
        )

        position.update_price(0.0000005)  # Price halves

        assert position.get_pnl_percent() == -50.0
        assert position.get_pnl_sol() == -0.5

    def test_take_profit_trigger(self):
        """Test take profit condition."""
        position = Position(
            token_address="TEST123",
            entry_time=datetime.now(),
            entry_price=0.000001,
            amount_tokens=1_000_000,
            amount_sol_invested=1.0,
            peak_price=0.000001,
        )

        position.update_price(0.000003)  # +200%

        assert position.should_take_profit(100.0) is True  # Target 100%
        assert position.should_take_profit(300.0) is False  # Target 300%

    def test_stop_loss_trigger(self):
        """Test stop loss condition."""
        position = Position(
            token_address="TEST123",
            entry_time=datetime.now(),
            entry_price=0.000001,
            amount_tokens=1_000_000,
            amount_sol_invested=1.0,
            peak_price=0.000001,
        )

        position.update_price(0.0000005)  # -50%

        assert position.should_stop_loss(25.0) is True  # 25% stop
        assert position.should_stop_loss(75.0) is False  # 75% stop

    def test_trailing_stop_trigger(self):
        """Test trailing stop from peak."""
        position = Position(
            token_address="TEST123",
            entry_time=datetime.now(),
            entry_price=0.000001,
            amount_tokens=1_000_000,
            amount_sol_invested=1.0,
            peak_price=0.000001,
        )

        # Price goes up
        position.update_price(0.000003)  # Peak at 0.000003
        assert position.peak_price == 0.000003

        # Price drops 20% from peak
        position.update_price(0.0000024)

        assert position.should_trailing_stop(15.0) is True  # 15% trail
        assert position.should_trailing_stop(25.0) is False  # 25% trail


class TestPositionManager:
    """Test portfolio management."""

    def setup_method(self):
        """Setup test environment."""
        config = BotConfig()
        logger = FenrirLogger(config)
        self.manager = PositionManager(config, logger)

    def test_open_position(self):
        """Test opening a position."""
        self.manager.open_position(
            token_address="TOKEN1", entry_price=0.000001, amount_tokens=1_000_000, amount_sol=1.0
        )

        assert len(self.manager.positions) == 1
        assert "TOKEN1" in self.manager.positions

    def test_close_position(self):
        """Test closing a position."""
        self.manager.open_position(
            token_address="TOKEN1", entry_price=0.000001, amount_tokens=1_000_000, amount_sol=1.0
        )

        position = self.manager.close_position("TOKEN1", "Test exit")

        assert len(self.manager.positions) == 0
        assert position is not None
        assert position.token_address == "TOKEN1"

    def test_portfolio_summary(self):
        """Test portfolio summary calculation."""
        self.manager.open_position(
            token_address="TOKEN1", entry_price=0.000001, amount_tokens=1_000_000, amount_sol=1.0
        )
        self.manager.open_position(
            token_address="TOKEN2", entry_price=0.000002, amount_tokens=500_000, amount_sol=1.0
        )

        self.manager.update_prices(
            {
                "TOKEN1": 0.000002,  # +100%
                "TOKEN2": 0.000001,  # -50%
            }
        )

        summary = self.manager.get_portfolio_summary()

        assert summary["num_positions"] == 2
        assert summary["total_invested_sol"] == 2.0
        assert summary["total_pnl_sol"] == pytest.approx(0.5, abs=0.01)
        assert summary["total_pnl_pct"] == pytest.approx(25.0, abs=1.0)

    def test_exit_conditions(self):
        """Test automatic exit signal detection."""
        self.manager.open_position(
            token_address="TOKEN1", entry_price=0.000001, amount_tokens=1_000_000, amount_sol=1.0
        )

        self.manager.update_prices({"TOKEN1": 0.000003})  # +200%

        exits = self.manager.check_exit_conditions()

        assert len(exits) == 1
        assert exits[0][0] == "TOKEN1"
        assert "Take Profit" in exits[0][1]


class TestWalletManager:
    """Test wallet functionality."""

    def test_simulation_wallet(self):
        """Test simulation mode generates throwaway wallet."""
        wallet = WalletManager("", simulation_mode=True)

        address = wallet.get_address()
        assert len(address) > 0
        assert wallet.simulation_mode is True

    def test_invalid_private_key(self):
        """Test invalid private key handling."""
        with pytest.raises(ValueError):
            WalletManager("invalid_key", simulation_mode=False)


def run_tests():
    """Run tests without pytest."""
    print("Running FENRIR Test Suite")
    print("=" * 70)

    test_classes = [TestBotConfig(), TestPosition(), TestPositionManager(), TestWalletManager()]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}")
        print("-" * 70)

        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    if hasattr(test_class, "setup_method"):
                        test_class.setup_method()

                    method = getattr(test_class, method_name)
                    method()

                    print(f"  PASS: {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  FAIL: {method_name}: {str(e)}")

    print("\n" + "=" * 70)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("All tests passed!")
        return 0
    else:
        print(f"{total_tests - passed_tests} tests failed")
        return 1


if __name__ == "__main__":
    try:
        import pytest

        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not found, running simple test suite...")
        exit(run_tests())
