#!/usr/bin/env python3
"""
FENRIR - Trading Engine Tests

Async test suite for TradingEngine: simulation mode, live execution,
wallet balance pre-checks, slippage guards, and position management.

Run with: pytest tests/test_trading_engine.py -v
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.core.positions import Position, PositionManager
from fenrir.protocol.pumpfun import BondingCurveState
from fenrir.trading.engine import LAMPORTS_PER_SOL, TradingEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_TOKEN = "So11111111111111111111111111111111111111199"

FRESH_CURVE = BondingCurveState(
    virtual_token_reserves=1_073_000_000,
    virtual_sol_reserves=30_000_000_000,
    real_token_reserves=793_100_000,
    real_sol_reserves=0,
    token_total_supply=1_000_000_000,
    complete=False,
)

MIGRATED_CURVE = BondingCurveState(
    virtual_token_reserves=1_073_000_000,
    virtual_sol_reserves=30_000_000_000,
    real_token_reserves=793_100_000,
    real_sol_reserves=85_000_000_000,
    token_total_supply=1_000_000_000,
    complete=True,
)


def _make_token_data(curve: BondingCurveState = None) -> dict:
    data = {"token_address": FAKE_TOKEN}
    if curve is not None:
        data["bonding_curve_state"] = curve
    return data


def _make_position(
    current_price: float = 0.000001,
    entry_price: float = 0.000001,
    amount_tokens: float = 1_000_000,
    amount_sol: float = 0.1,
) -> Position:
    return Position(
        token_address=FAKE_TOKEN,
        entry_time=datetime.now(),
        entry_price=entry_price,
        amount_tokens=amount_tokens,
        amount_sol_invested=amount_sol,
        peak_price=entry_price,
        current_price=current_price,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return BotConfig(
        mode=TradingMode.SIMULATION,
        buy_amount_sol=0.1,
        max_slippage_bps=500,
        priority_fee_lamports=100_000,
    )


@pytest.fixture
def live_config():
    return BotConfig(
        mode=TradingMode.AGGRESSIVE,
        buy_amount_sol=0.1,
        max_slippage_bps=500,
        priority_fee_lamports=100_000,
        use_jito=False,
    )


@pytest.fixture
def mocks():
    """Return a dict of AsyncMock/MagicMock stubs for all engine dependencies."""
    wallet = MagicMock()
    wallet.pubkey = MagicMock()
    wallet.keypair = MagicMock()

    solana_client = AsyncMock()
    jupiter = AsyncMock()
    positions = MagicMock(spec=PositionManager)
    positions.positions = {}
    logger = MagicMock()

    return {
        "wallet": wallet,
        "solana_client": solana_client,
        "jupiter": jupiter,
        "positions": positions,
        "logger": logger,
    }


@pytest.fixture
def sim_engine(config, mocks):
    return TradingEngine(
        config=config,
        wallet=mocks["wallet"],
        solana_client=mocks["solana_client"],
        jupiter=mocks["jupiter"],
        positions=mocks["positions"],
        logger=mocks["logger"],
    )


@pytest.fixture
def live_engine(live_config, mocks):
    return TradingEngine(
        config=live_config,
        wallet=mocks["wallet"],
        solana_client=mocks["solana_client"],
        jupiter=mocks["jupiter"],
        positions=mocks["positions"],
        logger=mocks["logger"],
    )


# ===================================================================
#  Simulation Mode — Buy
# ===================================================================


class TestSimulationBuy:
    @pytest.mark.asyncio
    async def test_sim_buy_opens_position_with_bonding_curve_pricing(self, sim_engine, mocks):
        """Simulation buy uses bonding-curve state for realistic pricing."""
        token_data = _make_token_data(curve=FRESH_CURVE)
        result = await sim_engine.execute_buy(token_data)

        assert result is True

        tokens_out, _ = FRESH_CURVE.calculate_buy_price(0.1)
        expected_price = (0.1 * LAMPORTS_PER_SOL) / tokens_out

        call_args = mocks["positions"].open_position.call_args
        assert call_args.kwargs["token_address"] == FAKE_TOKEN
        assert call_args.kwargs["amount_tokens"] == tokens_out
        assert call_args.kwargs["entry_price"] == pytest.approx(expected_price, rel=1e-6)
        assert call_args.kwargs["amount_sol"] == 0.1

    @pytest.mark.asyncio
    async def test_sim_buy_fallback_without_curve(self, sim_engine, mocks):
        """Without bonding curve data, simulation uses fallback 0.000001 price."""
        token_data = _make_token_data(curve=None)
        result = await sim_engine.execute_buy(token_data)

        assert result is True
        call_args = mocks["positions"].open_position.call_args
        assert call_args.kwargs["entry_price"] == 0.000001

    @pytest.mark.asyncio
    async def test_sim_buy_custom_amount_sol(self, sim_engine, mocks):
        """Explicit amount_sol overrides config default."""
        token_data = _make_token_data(curve=None)
        result = await sim_engine.execute_buy(token_data, amount_sol=0.5)

        assert result is True
        call_args = mocks["positions"].open_position.call_args
        assert call_args.kwargs["amount_sol"] == 0.5


# ===================================================================
#  Simulation Mode — Sell
# ===================================================================


class TestSimulationSell:
    @pytest.mark.asyncio
    async def test_sim_sell_closes_position_and_logs_pnl(self, sim_engine, mocks):
        """Simulation sell logs PnL and delegates to close_position."""
        position = _make_position(current_price=0.000002, entry_price=0.000001)
        mocks["positions"].positions = {FAKE_TOKEN: position}

        result = await sim_engine.execute_sell(FAKE_TOKEN, "take_profit")

        assert result is True
        mocks["positions"].close_position.assert_called_once_with(FAKE_TOKEN, "take_profit")
        # Logger should have been called with PnL information
        assert mocks["logger"].info.call_count >= 2  # at least "executing sell" + "sim exit"

    @pytest.mark.asyncio
    async def test_sim_sell_uses_entry_price_when_current_is_zero(self, sim_engine, mocks):
        """When current_price is None/0, exit price falls back to entry_price."""
        position = _make_position(current_price=0.0)
        mocks["positions"].positions = {FAKE_TOKEN: position}

        result = await sim_engine.execute_sell(FAKE_TOKEN, "timeout")

        assert result is True
        mocks["positions"].close_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_sell_no_position_returns_false(self, sim_engine, mocks):
        """Selling a token with no open position returns False."""
        mocks["positions"].positions = {}
        result = await sim_engine.execute_sell(FAKE_TOKEN, "stop_loss")

        assert result is False
        mocks["positions"].close_position.assert_not_called()


# ===================================================================
#  Live Mode — Buy
# ===================================================================


class TestLiveBuy:
    @pytest.mark.asyncio
    async def test_insufficient_balance_rejects(self, live_engine, mocks):
        """Wallet balance pre-check rejects when SOL is insufficient."""
        mocks["solana_client"].get_balance.return_value = 0.05  # less than 0.1 + fee

        result = await live_engine.execute_buy(_make_token_data())

        assert result is False
        mocks["positions"].open_position.assert_not_called()
        mocks["logger"].warning.assert_called()
        warn_msg = mocks["logger"].warning.call_args[0][0]
        assert "Insufficient" in warn_msg

    @pytest.mark.asyncio
    async def test_price_impact_exceeds_slippage_returns_false(self, live_engine, mocks):
        """Buy is rejected when price impact exceeds max slippage."""
        mocks["solana_client"].get_balance.return_value = 10.0

        # Build a curve where a buy causes huge impact
        thin_curve = BondingCurveState(
            virtual_token_reserves=1_000,
            virtual_sol_reserves=1_000_000,  # extremely thin liquidity
            real_token_reserves=1_000,
            real_sol_reserves=0,
            token_total_supply=1_000_000,
            complete=False,
        )
        # Encode fake account data that decodes to thin_curve
        mocks["solana_client"].get_account_info.return_value = b"x" * 80

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            with patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=thin_curve):
                result = await live_engine.execute_buy(_make_token_data())

        assert result is False

    @pytest.mark.asyncio
    async def test_migrated_token_returns_false(self, live_engine, mocks):
        """Buy is rejected when bonding curve shows token already migrated."""
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["solana_client"].get_account_info.return_value = b"x" * 80

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            with patch.object(
                live_engine.pumpfun, "decode_bonding_curve", return_value=MIGRATED_CURVE
            ):
                result = await live_engine.execute_buy(_make_token_data())

        assert result is False
        warn_msg = mocks["logger"].warning.call_args[0][0]
        assert "migrated" in warn_msg.lower()

    @pytest.mark.asyncio
    async def test_no_account_data_returns_false(self, live_engine, mocks):
        """Buy fails gracefully when bonding curve account is not found."""
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["solana_client"].get_account_info.return_value = None

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            result = await live_engine.execute_buy(_make_token_data())

        assert result is False

    @pytest.mark.asyncio
    async def test_decode_failure_returns_false(self, live_engine, mocks):
        """Buy fails when bonding curve data cannot be decoded."""
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["solana_client"].get_account_info.return_value = b"x" * 80

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            with patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=None):
                result = await live_engine.execute_buy(_make_token_data())

        assert result is False

    @pytest.mark.asyncio
    async def test_successful_live_buy_opens_position(self, live_engine, mocks):
        """Full happy-path live buy: balance ok, curve ok, tx confirmed."""
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["solana_client"].get_account_info.return_value = b"x" * 80
        mocks["solana_client"].get_latest_blockhash.return_value = MagicMock()
        mocks["solana_client"].simulate_transaction.return_value = True
        mocks["solana_client"].send_transaction.return_value = "fake_sig_abc123"

        # Confirmation polling returns confirmed on first attempt
        confirmed_status = MagicMock()
        confirmed_status.err = None
        confirmed_status.confirmation_status = "confirmed"
        mocks["solana_client"].get_signature_statuses.return_value = [confirmed_status]

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            with patch.object(
                live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE
            ):
                with patch.object(
                    live_engine.pumpfun,
                    "get_associated_bonding_curve_address",
                    return_value=MagicMock(),
                ):
                    with patch.object(
                        live_engine.pumpfun, "build_buy_instruction", return_value=MagicMock()
                    ):
                        with patch("fenrir.trading.engine.Message") as MockMsg:
                            with patch("fenrir.trading.engine.Transaction") as MockTx:
                                with patch(
                                    "fenrir.trading.engine.get_associated_token_address",
                                    return_value=MagicMock(),
                                ):
                                    with patch(
                                        "fenrir.trading.engine.set_compute_unit_price",
                                        return_value=MagicMock(),
                                    ):
                                        with patch(
                                            "fenrir.trading.engine.set_compute_unit_limit",
                                            return_value=MagicMock(),
                                        ):
                                            mock_tx_instance = MagicMock()
                                            MockTx.new_unsigned.return_value = mock_tx_instance
                                            MockMsg.new_with_blockhash.return_value = MagicMock()

                                            result = await live_engine.execute_buy(
                                                _make_token_data()
                                            )

        assert result is True
        mocks["positions"].open_position.assert_called_once()
        call_kw = mocks["positions"].open_position.call_args.kwargs
        assert call_kw["token_address"] == FAKE_TOKEN
        assert call_kw["amount_sol"] == 0.1


# ===================================================================
#  Live Mode — Transaction Confirmation Polling
# ===================================================================


class TestConfirmTransaction:
    @pytest.mark.asyncio
    async def test_confirmed_on_first_poll(self, live_engine, mocks):
        """Transaction confirmed on first polling attempt."""
        status = MagicMock()
        status.err = None
        status.confirmation_status = "confirmed"
        mocks["solana_client"].get_signature_statuses.return_value = [status]

        with patch("fenrir.trading.engine.asyncio.sleep", new_callable=AsyncMock):
            result = await live_engine._confirm_transaction("sig123")

        assert result is True

    @pytest.mark.asyncio
    async def test_finalized_counts_as_confirmed(self, live_engine, mocks):
        """'finalized' confirmation_status is accepted."""
        status = MagicMock()
        status.err = None
        status.confirmation_status = "finalized"
        mocks["solana_client"].get_signature_statuses.return_value = [status]

        with patch("fenrir.trading.engine.asyncio.sleep", new_callable=AsyncMock):
            result = await live_engine._confirm_transaction("sig123")

        assert result is True

    @pytest.mark.asyncio
    async def test_on_chain_error_returns_false(self, live_engine, mocks):
        """Transaction that landed but had an on-chain error returns False."""
        status = MagicMock()
        status.err = {"InstructionError": [0, "InsufficientFunds"]}
        status.confirmation_status = "confirmed"
        mocks["solana_client"].get_signature_statuses.return_value = [status]

        with patch("fenrir.trading.engine.asyncio.sleep", new_callable=AsyncMock):
            result = await live_engine._confirm_transaction("sig123")

        assert result is False

    @pytest.mark.asyncio
    async def test_not_confirmed_after_max_attempts(self, live_engine, mocks):
        """Returns False when all polling attempts yield no status."""
        mocks["solana_client"].get_signature_statuses.return_value = [None]

        with patch("fenrir.trading.engine.asyncio.sleep", new_callable=AsyncMock):
            result = await live_engine._confirm_transaction("sig123", max_attempts=3)

        assert result is False
        assert mocks["solana_client"].get_signature_statuses.call_count == 3

    @pytest.mark.asyncio
    async def test_confirms_after_several_retries(self, live_engine, mocks):
        """Transaction not found initially, then confirmed on third poll."""
        confirmed = MagicMock()
        confirmed.err = None
        confirmed.confirmation_status = "confirmed"

        mocks["solana_client"].get_signature_statuses.side_effect = [
            [None],
            [None],
            [confirmed],
        ]

        with patch("fenrir.trading.engine.asyncio.sleep", new_callable=AsyncMock):
            result = await live_engine._confirm_transaction("sig123", max_attempts=5)

        assert result is True
        assert mocks["solana_client"].get_signature_statuses.call_count == 3


# ===================================================================
#  Live Mode — Sell
# ===================================================================


class TestLiveSell:
    @pytest.mark.asyncio
    async def test_sell_uses_actual_on_chain_balance(self, live_engine, mocks):
        """Sell amount is the on-chain token balance, not the position's amount_tokens."""
        position = _make_position(amount_tokens=999_999)
        mocks["positions"].positions = {FAKE_TOKEN: position}

        on_chain_balance = 1_000_001  # differs from position.amount_tokens
        mocks["solana_client"].get_account_info.return_value = b"x" * 80
        mocks["solana_client"].get_token_accounts_by_owner.return_value = {
            "address": MagicMock(),
            "amount": on_chain_balance,
        }
        mocks["solana_client"].get_latest_blockhash.return_value = MagicMock()
        mocks["solana_client"].simulate_transaction.return_value = True
        mocks["solana_client"].send_transaction.return_value = "sell_sig_xyz"

        confirmed_status = MagicMock()
        confirmed_status.err = None
        confirmed_status.confirmation_status = "confirmed"
        mocks["solana_client"].get_signature_statuses.return_value = [confirmed_status]

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            with patch.object(
                live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE
            ):
                with patch.object(
                    live_engine.pumpfun,
                    "get_associated_bonding_curve_address",
                    return_value=MagicMock(),
                ):
                    with patch.object(
                        live_engine.pumpfun, "build_sell_instruction"
                    ) as mock_sell_ix:
                        mock_sell_ix.return_value = MagicMock()
                        with patch("fenrir.trading.engine.Message") as MockMsg:
                            with patch("fenrir.trading.engine.Transaction") as MockTx:
                                with patch(
                                    "fenrir.trading.engine.set_compute_unit_price",
                                    return_value=MagicMock(),
                                ):
                                    with patch(
                                        "fenrir.trading.engine.set_compute_unit_limit",
                                        return_value=MagicMock(),
                                    ):
                                        mock_tx_instance = MagicMock()
                                        MockTx.new_unsigned.return_value = mock_tx_instance
                                        MockMsg.new_with_blockhash.return_value = MagicMock()

                                        result = await live_engine.execute_sell(
                                            FAKE_TOKEN, "take_profit"
                                        )

        assert result is True
        # Verify the sell instruction received the on-chain balance, not position amount
        sell_call = mock_sell_ix.call_args
        assert sell_call.kwargs["amount_tokens"] == on_chain_balance

    @pytest.mark.asyncio
    async def test_sell_migrated_token_falls_back_to_jupiter(self, live_engine, mocks):
        """When curve is complete, sell delegates to Jupiter DEX fallback."""
        position = _make_position()
        mocks["positions"].positions = {FAKE_TOKEN: position}

        mocks["solana_client"].get_account_info.return_value = b"x" * 80

        # Jupiter fallback succeeds
        mocks["solana_client"].get_token_accounts_by_owner.return_value = {
            "address": MagicMock(),
            "amount": 500_000,
        }
        mocks["jupiter"].get_quote.return_value = {"outAmount": "50000000"}
        mocks["jupiter"].get_swap_transaction.return_value = "base64tx"

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            with patch.object(
                live_engine.pumpfun, "decode_bonding_curve", return_value=MIGRATED_CURVE
            ):
                result = await live_engine.execute_sell(FAKE_TOKEN, "take_profit")

        assert result is True
        mocks["jupiter"].get_quote.assert_called_once()
        mocks["positions"].close_position.assert_called_once_with(FAKE_TOKEN, "take_profit")

    @pytest.mark.asyncio
    async def test_sell_no_token_account_returns_false(self, live_engine, mocks):
        """Sell fails when wallet has no token account for the token."""
        position = _make_position()
        mocks["positions"].positions = {FAKE_TOKEN: position}

        mocks["solana_client"].get_account_info.return_value = b"x" * 80
        mocks["solana_client"].get_token_accounts_by_owner.return_value = None

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            with patch.object(
                live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE
            ):
                result = await live_engine.execute_sell(FAKE_TOKEN, "stop_loss")

        assert result is False

    @pytest.mark.asyncio
    async def test_sell_zero_balance_returns_false(self, live_engine, mocks):
        """Sell fails when on-chain token balance is zero."""
        position = _make_position()
        mocks["positions"].positions = {FAKE_TOKEN: position}

        mocks["solana_client"].get_account_info.return_value = b"x" * 80
        mocks["solana_client"].get_token_accounts_by_owner.return_value = {
            "address": MagicMock(),
            "amount": 0,
        }

        with patch.object(
            live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
        ):
            with patch.object(
                live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE
            ):
                result = await live_engine.execute_sell(FAKE_TOKEN, "stop_loss")

        assert result is False


# ===================================================================
#  Position Management
# ===================================================================


class TestManagePositions:
    @pytest.mark.asyncio
    async def test_delegates_to_check_exit_conditions(self, sim_engine, mocks):
        """manage_positions calls check_exit_conditions and sells flagged positions."""
        mocks["positions"].check_exit_conditions.return_value = [
            (FAKE_TOKEN, "Take Profit: 120.00%"),
        ]
        position = _make_position(current_price=0.0000022)
        mocks["positions"].positions = {FAKE_TOKEN: position}

        await sim_engine.manage_positions()

        mocks["positions"].check_exit_conditions.assert_called_once()
        # In simulation mode execute_sell should close the position
        mocks["positions"].close_position.assert_called_once_with(
            FAKE_TOKEN, "Take Profit: 120.00%"
        )

    @pytest.mark.asyncio
    async def test_no_exits_does_nothing(self, sim_engine, mocks):
        """When no exit conditions are met, no sells are triggered."""
        mocks["positions"].check_exit_conditions.return_value = []

        await sim_engine.manage_positions()

        mocks["positions"].close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_exits_processed(self, sim_engine, mocks):
        """Multiple exit signals each trigger a sell."""
        token_a = "TokenAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA1"
        token_b = "TokenBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB2"

        pos_a = _make_position()
        pos_a.token_address = token_a
        pos_b = _make_position()
        pos_b.token_address = token_b

        mocks["positions"].positions = {token_a: pos_a, token_b: pos_b}
        mocks["positions"].check_exit_conditions.return_value = [
            (token_a, "Stop Loss: -30.00%"),
            (token_b, "Max hold time reached"),
        ]

        await sim_engine.manage_positions()

        assert mocks["positions"].close_position.call_count == 2
