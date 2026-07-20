#!/usr/bin/env python3
"""
FENRIR - Trading Engine Tests

Async test suite for TradingEngine: simulation mode, live execution,
wallet balance pre-checks, slippage guards, and position management.

Run with: pytest tests/test_trading_engine.py -v
"""

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from solders.pubkey import Pubkey

from fenrir.config import BotConfig, TradingMode
from fenrir.core.positions import Position, PositionManager
from fenrir.protocol.pumpfun import BondingCurveState
from fenrir.trading.engine import TradingEngine, cap_priority_fee

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_TOKEN = "So11111111111111111111111111111111111111199"

FAKE_CREATOR = "11111111111111111111111111111111"

FRESH_CURVE = BondingCurveState(
    virtual_token_reserves=1_073_000_000,
    virtual_sol_reserves=30_000_000_000,
    real_token_reserves=793_100_000,
    real_sol_reserves=0,
    token_total_supply=1_000_000_000,
    complete=False,
    creator=FAKE_CREATOR,
)

MIGRATED_CURVE = BondingCurveState(
    virtual_token_reserves=1_073_000_000,
    virtual_sol_reserves=30_000_000_000,
    real_token_reserves=793_100_000,
    real_sol_reserves=85_000_000_000,
    token_total_supply=1_000_000_000,
    complete=True,
    creator=FAKE_CREATOR,
)


def _make_token_data(curve: BondingCurveState | None = None) -> dict:
    data: dict[str, Any] = {"token_address": FAKE_TOKEN}
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
#  Priority-fee capping
# ===================================================================


class TestCapPriorityFee:
    """A flat lamport fee must not eat a small position.

    degen presets 2_000_000 lamports (0.002 SOL): 0.4% of its intended 0.5 SOL trade
    but 20% PER SIDE of a 0.01 SOL one — a 40% round-trip drag that turned a real
    +3.10% trade into a wallet loss.
    """

    def test_caps_flat_fee_on_small_trade(self):
        # 3% of 0.01 SOL = 300_000 lamports.
        assert cap_priority_fee(2_000_000, 0.01, 3.0, 50_000) == 300_000

    def test_leaves_fee_alone_when_trade_is_large(self):
        # 3% of 0.5 SOL = 15_000_000 > requested — degen's intended size is unaffected.
        assert cap_priority_fee(2_000_000, 0.5, 3.0, 50_000) == 2_000_000

    def test_never_caps_below_inclusion_floor(self):
        # 3% of 0.0001 SOL = 300 lamports — too low to land; floor wins.
        assert cap_priority_fee(2_000_000, 0.0001, 3.0, 50_000) == 50_000

    def test_disabled_when_pct_zero(self):
        assert cap_priority_fee(2_000_000, 0.01, 0.0, 50_000) == 2_000_000

    def test_unknown_size_passes_through(self):
        assert cap_priority_fee(2_000_000, 0.0, 3.0, 50_000) == 2_000_000

    def test_round_trip_drag_is_bounded(self):
        """The cap bounds total fee drag to ~2*max_pct of the position."""
        size, fee = 0.01, cap_priority_fee(2_000_000, 0.01, 3.0, 50_000)
        round_trip_sol = 2 * (fee + 5_000) / 1_000_000_000
        assert round_trip_sol / size < 0.07  # was 0.40 before capping


# ===================================================================
#  Simulation Mode — Buy
# ===================================================================


class TestSimulationBuy:
    @pytest.mark.asyncio
    async def test_sim_buy_opens_position_with_bonding_curve_pricing(self, sim_engine, mocks):
        """Simulation buy prices entry from the bonding curve (get_price), keeping
        amount_tokens * entry_price == amount_sol so PnL tracks the price ratio."""
        token_data = _make_token_data(curve=FRESH_CURVE)
        result = await sim_engine.execute_buy(token_data)

        assert result is True

        expected_price = FRESH_CURVE.get_price()
        expected_tokens = 0.1 / expected_price

        call_args = mocks["positions"].open_position.call_args
        assert call_args.kwargs["token_address"] == FAKE_TOKEN
        assert call_args.kwargs["entry_price"] == pytest.approx(expected_price, rel=1e-9)
        assert call_args.kwargs["amount_tokens"] == pytest.approx(expected_tokens, rel=1e-9)
        assert call_args.kwargs["amount_sol"] == 0.1

    @pytest.mark.asyncio
    async def test_sim_buy_refused_without_curve(self, sim_engine, mocks):
        """Without any bonding curve, the sim buy is refused rather than opening a
        fabricated position (the old 0.000001 fallback produced nonsensical PnL)."""
        mocks["solana_client"].get_account_info.return_value = None
        token_data = _make_token_data(curve=None)
        result = await sim_engine.execute_buy(token_data)

        assert result is False
        mocks["positions"].open_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_sim_buy_custom_amount_sol(self, sim_engine, mocks):
        """Explicit amount_sol overrides config default."""
        token_data = _make_token_data(curve=FRESH_CURVE)
        result = await sim_engine.execute_buy(token_data, amount_sol=0.5)

        assert result is True
        call_args = mocks["positions"].open_position.call_args
        assert call_args.kwargs["amount_sol"] == 0.5

    @pytest.mark.asyncio
    async def test_sim_buy_books_the_entry_fee(self, sim_engine, mocks):
        """Sim must record the fee a live trade of this size would pay.

        Keystone of the honest-economics work: without this, sim reports gross price
        movement and can't validate the net-PnL / fee-cap logic without real SOL.
        config priority fee is 100_000 lamports, uncapped at 0.1 SOL.
        """
        await sim_engine.execute_buy(_make_token_data(curve=FRESH_CURVE))
        entry_fees = mocks["positions"].open_position.call_args.kwargs["entry_fees_sol"]
        assert entry_fees == pytest.approx((100_000 + 5_000) / 1e9)  # priority + base

    @pytest.mark.asyncio
    async def test_sim_non_curve_buy_prices_entry_from_feed(self, sim_engine, mocks):
        """Sim entry must use the feed the management loop marks against, like live.

        Pricing from the swap quote put entry on a different scale than the mark, so
        PnL was phantom the instant the position was marked.
        """
        mocks["jupiter"].get_quote.return_value = {"outAmount": "2000000"}  # 2.0 @6dp
        feed_quote = MagicMock()
        feed_quote.price = 0.0042  # SOL/token from the feed, differs from 0.01/2.0
        sim_engine.price_feed = MagicMock()
        sim_engine.price_feed.get_price = AsyncMock(return_value=feed_quote)
        td = {"token_address": FAKE_TOKEN, "tier": "large", "symbol": "BIG", "decimals": 6}

        assert await sim_engine.execute_buy(td, 0.01) is True

        kwargs = mocks["positions"].open_position.call_args.kwargs
        assert kwargs["entry_price"] == pytest.approx(0.0042)  # feed, not quote-derived
        # Tokens derived from the feed price keep tokens * entry == amount_sol.
        assert kwargs["amount_tokens"] == pytest.approx(0.01 / 0.0042)

    @pytest.mark.asyncio
    async def test_sim_non_curve_buy_falls_back_to_quote_without_feed(self, sim_engine, mocks):
        """No feed → quote-derived entry (position still openable, entry/mark aligned)."""
        mocks["jupiter"].get_quote.return_value = {"outAmount": "2000000"}  # 2.0 @6dp
        sim_engine.price_feed = None
        td = {"token_address": FAKE_TOKEN, "tier": "large", "symbol": "BIG", "decimals": 6}

        assert await sim_engine.execute_buy(td, 0.01) is True

        kwargs = mocks["positions"].open_position.call_args.kwargs
        assert kwargs["entry_price"] == pytest.approx(0.01 / 2.0)
        assert kwargs["amount_tokens"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_sim_non_curve_buy_survives_bad_decimals(self, sim_engine, mocks):
        """Regression: cbBTC opened at entry ~0 and 'gained' 7,499% when marked.

        A wrong `decimals` (6 for an 8-decimal token) inflates out_ui 100x, which
        made the quote-derived entry 100x too LOW — so the first mark from the feed
        showed a phantom four-figure gain, and took profit on it. Feed pricing plus
        feed-derived tokens make the position immune to the bad decimals read.
        """
        # Real: ~400 SOL/token, so 0.01 SOL buys 0.000025 tokens (2500 base @8dp).
        mocks["jupiter"].get_quote.return_value = {"outAmount": "2500"}
        feed_quote = MagicMock()
        feed_quote.price = 400.0
        sim_engine.price_feed = MagicMock()
        sim_engine.price_feed.get_price = AsyncMock(return_value=feed_quote)
        # decimals misread as 6 (actually 8) → out_ui = 0.0025, 100x too many tokens.
        td = {"token_address": FAKE_TOKEN, "tier": "large", "symbol": "BTCish", "decimals": 6}

        assert await sim_engine.execute_buy(td, 0.01) is True

        kwargs = mocks["positions"].open_position.call_args.kwargs
        assert kwargs["entry_price"] == pytest.approx(400.0)  # NOT the 0.01/0.0025 = 4.0
        assert kwargs["amount_tokens"] == pytest.approx(0.01 / 400.0)
        # Marked at the same feed price, PnL is ~0 — not the +7499% that fired a
        # take-profit on a position that never moved.
        pos = Position(
            token_address=FAKE_TOKEN,
            entry_time=datetime.now(),
            entry_price=kwargs["entry_price"],
            amount_tokens=kwargs["amount_tokens"],
            amount_sol_invested=0.01,
            peak_price=kwargs["entry_price"],
        )
        pos.update_price(400.0)
        assert pos.get_pnl_percent() == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_sim_round_trip_net_pnl_reflects_fees(self, config, mocks):
        """End-to-end in sim with a REAL PositionManager: a small gross gain is a net
        loss once both fees are booked — the exact failure mode a live 0.01 trade hit,
        now observable in simulation at zero cost."""
        from fenrir.core.positions import PositionManager

        pm = PositionManager(config, mocks["logger"])
        engine = TradingEngine(
            config=config,
            wallet=mocks["wallet"],
            solana_client=mocks["solana_client"],
            jupiter=mocks["jupiter"],
            positions=pm,
            logger=mocks["logger"],
        )
        # Tiny position so the flat fee dominates, like the real 0.01 SOL trade.
        assert await engine.execute_buy(_make_token_data(curve=FRESH_CURVE), amount_sol=0.01)
        pos = pm.positions[FAKE_TOKEN]
        pos.update_price(pos.entry_price * 1.03)  # +3% gross

        assert pos.get_pnl_percent() == pytest.approx(3.0, abs=0.01)
        assert await engine.execute_sell(FAKE_TOKEN, "take_profit")
        # Round trip books 2x (100_000 + 5_000) = 0.00021 SOL of fees on a 0.01 SOL
        # position: a +3% gross move (+0.0003 SOL) nets barely positive; drop gross
        # to +1% and it's a loss — the point being sim now SEES the fees.
        assert pos.total_fees_sol == pytest.approx(2 * (105_000 / 1e9))
        assert pos.get_net_pnl_sol() < pos.get_pnl_sol()


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
        # Sim now books the exit fee it WOULD pay, so sim PnL is net like live.
        call = mocks["positions"].close_position.call_args
        assert call.args == (FAKE_TOKEN, "take_profit")
        assert call.kwargs["exit_fees_sol"] > 0
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

        # A shadowable prior buy exists → per-token fee accounts resolve.
        fee_extras = (Pubkey.from_string(FAKE_TOKEN), Pubkey.from_string(FAKE_TOKEN))
        with (
            patch.object(
                live_engine, "_resolve_fee_extras", new=AsyncMock(return_value=fee_extras)
            ),
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE),
            patch.object(
                live_engine.pumpfun, "build_create_ata_instruction", return_value=MagicMock()
            ),
            patch.object(live_engine.pumpfun, "build_buy_instruction", return_value=MagicMock()),
            patch("fenrir.trading.engine.Message") as MockMsg,
            patch("fenrir.trading.engine.Transaction") as MockTx,
            patch("fenrir.trading.engine.set_compute_unit_price", return_value=MagicMock()),
            patch("fenrir.trading.engine.set_compute_unit_limit", return_value=MagicMock()),
        ):
            MockTx.new_unsigned.return_value = MagicMock()
            MockMsg.new_with_blockhash.return_value = MagicMock()
            result = await live_engine.execute_buy(_make_token_data())

        assert result is True
        mocks["positions"].open_position.assert_called_once()
        call_kw = mocks["positions"].open_position.call_args.kwargs
        assert call_kw["token_address"] == FAKE_TOKEN
        assert call_kw["amount_sol"] == 0.1

    @pytest.mark.asyncio
    async def test_first_buyer_derives_cashback_tail_and_buys(self, live_engine, mocks):
        """First buyer (no shadowable prior buy) still buys: the cashback tail
        (bonding_curve_v2 + rotating fee) is derivable, so the buy executes."""
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["solana_client"].get_account_info.return_value = b"x" * 80
        mocks["solana_client"].get_latest_blockhash.return_value = MagicMock()
        mocks["solana_client"].simulate_transaction.return_value = True
        mocks["solana_client"].send_transaction.return_value = "first_buyer_sig"
        confirmed_status = MagicMock()
        confirmed_status.err = None
        confirmed_status.confirmation_status = "confirmed"
        mocks["solana_client"].get_signature_statuses.return_value = [confirmed_status]

        derived_tail = [MagicMock(), MagicMock()]  # bonding_curve_v2 + rotating fee
        with (
            # No legacy shadow available, but the cashback tail resolves (derived).
            patch.object(live_engine, "_resolve_fee_extras", new=AsyncMock(return_value=None)),
            patch.object(
                live_engine, "_resolve_buy_tail", new=AsyncMock(return_value=derived_tail)
            ),
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE),
            patch.object(
                live_engine.pumpfun, "build_create_ata_instruction", return_value=MagicMock()
            ),
            patch.object(
                live_engine.pumpfun, "build_buy_instruction", return_value=MagicMock()
            ) as mock_buy_ix,
            patch("fenrir.trading.engine.Message") as MockMsg,
            patch("fenrir.trading.engine.Transaction") as MockTx,
            patch("fenrir.trading.engine.set_compute_unit_price", return_value=MagicMock()),
            patch("fenrir.trading.engine.set_compute_unit_limit", return_value=MagicMock()),
        ):
            MockTx.new_unsigned.return_value = MagicMock()
            MockMsg.new_with_blockhash.return_value = MagicMock()
            result = await live_engine.execute_buy(_make_token_data())

        assert result is True
        mocks["positions"].open_position.assert_called_once()
        # The derived cashback tail was passed through to the instruction builder.
        assert mock_buy_ix.call_args.kwargs["extra_accounts"] is derived_tail

    @pytest.mark.asyncio
    async def test_buy_fast_fails_when_no_tail_resolvable(self, live_engine, mocks):
        """Neither a cashback tail nor a legacy shadow resolves → fail fast, no send."""
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["solana_client"].get_account_info.return_value = b"x" * 80

        with (
            patch.object(live_engine, "_resolve_buy_tail", new=AsyncMock(return_value=None)),
            patch.object(live_engine, "_resolve_fee_extras", new=AsyncMock(return_value=None)),
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE),
        ):
            result = await live_engine.execute_buy(_make_token_data())

        assert result is False
        mocks["positions"].open_position.assert_not_called()
        mocks["solana_client"].send_transaction.assert_not_called()
        assert any(
            "could not resolve" in str(c.args[0]) for c in mocks["logger"].warning.call_args_list
        )


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
    async def test_live_sell_adopts_onchain_balance_without_position(self, live_engine, mocks):
        """Live sell with NO tracked position adopts the on-chain balance — orphan
        recovery after a restart clears in-memory state (SIM still bails)."""
        mocks["positions"].positions = {}  # nothing tracked

        on_chain_balance = 152_451_106_296
        mocks["solana_client"].get_account_info.return_value = b"x" * 80
        mocks["solana_client"].get_token_accounts_by_owner.return_value = {
            "address": MagicMock(),
            "amount": on_chain_balance,
        }
        mocks["solana_client"].get_latest_blockhash.return_value = MagicMock()
        mocks["solana_client"].simulate_transaction.return_value = True
        mocks["solana_client"].send_transaction.return_value = "orphan_sig"
        confirmed_status = MagicMock()
        confirmed_status.err = None
        confirmed_status.confirmation_status = "confirmed"
        mocks["solana_client"].get_signature_statuses.return_value = [confirmed_status]

        with (
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE),
            patch.object(live_engine.pumpfun, "build_sell_instruction") as mock_sell_ix,
            patch("fenrir.trading.engine.Message") as MockMsg,
            patch("fenrir.trading.engine.Transaction") as MockTx,
            patch("fenrir.trading.engine.set_compute_unit_price", return_value=MagicMock()),
            patch("fenrir.trading.engine.set_compute_unit_limit", return_value=MagicMock()),
        ):
            mock_sell_ix.return_value = MagicMock()
            MockTx.new_unsigned.return_value = MagicMock()
            MockMsg.new_with_blockhash.return_value = MagicMock()
            result = await live_engine.execute_sell(FAKE_TOKEN, "orphan recovery")

        assert result is True  # sold despite no tracked position
        assert mock_sell_ix.call_args.kwargs["amount_tokens"] == on_chain_balance

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

        with (
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=MIGRATED_CURVE),
            patch.object(
                live_engine, "_sign_send_jupiter_swap", new=AsyncMock(return_value="jup_sig")
            ),
            patch.object(live_engine, "_confirm_transaction", new=AsyncMock(return_value=True)),
        ):
            result = await live_engine.execute_sell(FAKE_TOKEN, "take_profit")

        assert result is True
        mocks["jupiter"].get_quote.assert_called_once()
        # The exit fee is recorded so reported PnL can be net of costs.
        call = mocks["positions"].close_position.call_args
        assert call.args == (FAKE_TOKEN, "take_profit")
        assert call.kwargs["exit_fees_sol"] > 0

    @pytest.mark.asyncio
    async def test_sell_migrated_not_closed_when_send_fails(self, live_engine, mocks):
        """If the Jupiter swap never lands, the position must NOT be marked closed."""
        position = _make_position()
        mocks["positions"].positions = {FAKE_TOKEN: position}
        mocks["solana_client"].get_account_info.return_value = b"x" * 80
        mocks["solana_client"].get_token_accounts_by_owner.return_value = {
            "address": MagicMock(),
            "amount": 500_000,
        }
        mocks["jupiter"].get_quote.return_value = {"outAmount": "50000000"}
        mocks["jupiter"].get_swap_transaction.return_value = "base64tx"

        with (
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=MIGRATED_CURVE),
            patch.object(live_engine, "_sign_send_jupiter_swap", new=AsyncMock(return_value=None)),
        ):
            result = await live_engine.execute_sell(FAKE_TOKEN, "take_profit")

        assert result is False
        mocks["positions"].close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_sell_no_token_account_returns_false(self, live_engine, mocks):
        """Sell fails when wallet has no token account for the token."""
        position = _make_position()
        mocks["positions"].positions = {FAKE_TOKEN: position}

        mocks["solana_client"].get_account_info.return_value = b"x" * 80
        mocks["solana_client"].get_token_accounts_by_owner.return_value = None

        with (
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE),
            patch("fenrir.trading.engine.asyncio.sleep", new_callable=AsyncMock),
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

        with (
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(live_engine.pumpfun, "decode_bonding_curve", return_value=FRESH_CURVE),
            patch("fenrir.trading.engine.asyncio.sleep", new_callable=AsyncMock),
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
        # In simulation mode execute_sell should close the position (now fee-aware).
        call = mocks["positions"].close_position.call_args
        assert call.args == (FAKE_TOKEN, "Take Profit: 120.00%")
        assert call.kwargs["exit_fees_sol"] > 0

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


# ===================================================================
#  Per-token v2 buyback fee-account resolution
# ===================================================================


class TestResolveFeeExtras:
    """_resolve_fee_extras shadows the token's own recent buy for its idx16/17."""

    def _fake_buy_tx(self, keys):
        """Build a minimal tx object with an 18-account pump buy instruction."""
        import base58

        from fenrir.protocol.pumpfun import BUY_DISCRIMINATOR, PUMP_PROGRAM_ID

        ix = SimpleNamespace(
            program_id=PUMP_PROGRAM_ID,
            data=base58.b58encode(BUY_DISCRIMINATOR + b"\x00" * 17).decode(),
            accounts=list(range(18)),
        )
        message = SimpleNamespace(account_keys=keys, instructions=[ix])
        return SimpleNamespace(
            transaction=SimpleNamespace(transaction=SimpleNamespace(message=message))
        )

    @pytest.mark.asyncio
    async def test_extracts_idx16_17_and_caches(self, live_engine, mocks):
        keys = [Pubkey.from_bytes(bytes([i + 1] * 32)) for i in range(18)]
        mocks["solana_client"].get_recent_signatures.return_value = [
            SimpleNamespace(err=None, signature="sig1")
        ]
        mocks["solana_client"].get_transaction.return_value = self._fake_buy_tx(keys)

        extras = await live_engine._resolve_fee_extras(keys[2])
        assert extras == (keys[16], keys[17])

        # Second call must hit the per-mint cache (no RPC needed).
        mocks["solana_client"].get_recent_signatures.return_value = []
        assert await live_engine._resolve_fee_extras(keys[2]) == (keys[16], keys[17])

    @pytest.mark.asyncio
    async def test_returns_none_when_no_prior_buy(self, live_engine, mocks):
        mocks["solana_client"].get_recent_signatures.return_value = []
        mint = Pubkey.from_bytes(bytes([99] * 32))
        assert await live_engine._resolve_fee_extras(mint) is None


# ===================================================================
#  Dynamic priority fee
# ===================================================================


class TestDynamicPriorityFee:
    @pytest.mark.asyncio
    async def test_uses_p75_clamped(self, live_engine, mocks):
        live_engine.config.dynamic_priority_fee_enabled = True
        live_engine.config.priority_fee_lamports = 100
        live_engine.config.max_priority_fee_lamports = 10_000
        # p75 of [0,100,200,...,1000] non-zero -> 800, within clamp
        mocks["solana_client"].get_recent_prioritization_fees.return_value = list(
            range(0, 1100, 100)
        )
        fee = await live_engine._resolve_priority_fee("sniper")
        assert fee == 800

    @pytest.mark.asyncio
    async def test_clamps_to_ceiling(self, live_engine, mocks):
        live_engine.config.dynamic_priority_fee_enabled = True
        live_engine.config.priority_fee_lamports = 100
        live_engine.config.max_priority_fee_lamports = 500
        mocks["solana_client"].get_recent_prioritization_fees.return_value = [1_000, 2_000, 3_000]
        fee = await live_engine._resolve_priority_fee("sniper")
        assert fee == 500  # ceiling

    @pytest.mark.asyncio
    async def test_falls_back_to_floor_when_no_data(self, live_engine, mocks):
        live_engine.config.dynamic_priority_fee_enabled = True
        live_engine.config.priority_fee_lamports = 777
        mocks["solana_client"].get_recent_prioritization_fees.return_value = []
        fee = await live_engine._resolve_priority_fee("sniper")
        assert fee == 777

    @pytest.mark.asyncio
    async def test_disabled_uses_flat_fee(self, live_engine, mocks):
        live_engine.config.dynamic_priority_fee_enabled = False
        live_engine.config.priority_fee_lamports = 555
        fee = await live_engine._resolve_priority_fee("sniper")
        assert fee == 555
        mocks["solana_client"].get_recent_prioritization_fees.assert_not_called()


# ===================================================================
#  Non-curve (Jupiter) buy path — migrated / established tokens
# ===================================================================


class TestNonCurveBuy:
    def test_is_non_curve_token(self):
        assert TradingEngine._is_non_curve_token({"migrated": True}) is True
        assert TradingEngine._is_non_curve_token({"tier": "mid"}) is True
        assert TradingEngine._is_non_curve_token({"tier": "large"}) is True
        # fresh launch: on a curve, no tier/migrated flag
        assert TradingEngine._is_non_curve_token({"bonding_curve_state": object()}) is False
        assert TradingEngine._is_non_curve_token({"tier": "low"}) is False

    @pytest.mark.asyncio
    async def test_execute_buy_routes_non_curve_to_jupiter(self, live_engine):
        td = {"token_address": FAKE_TOKEN, "tier": "large", "symbol": "BIG"}
        with patch.object(
            live_engine, "execute_buy_via_jupiter", new=AsyncMock(return_value=True)
        ) as jup:
            result = await live_engine.execute_buy(td, amount_sol=0.01)
        assert result is True
        jup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_buy_via_jupiter_success_opens_position(self, live_engine, mocks):
        mocks["solana_client"].get_balance.return_value = 10.0
        # quote outAmount 2_000_000 base units at 6 decimals -> 2.0 whole tokens
        mocks["jupiter"].get_quote.return_value = {"outAmount": "2000000"}
        mocks["jupiter"].get_swap_transaction.return_value = "b64tx"
        td = {"token_address": FAKE_TOKEN, "tier": "large", "symbol": "BIG", "decimals": 6}
        with (
            patch.object(live_engine, "_sign_send_jupiter_swap", new=AsyncMock(return_value="sig")),
            patch.object(live_engine, "_confirm_transaction", new=AsyncMock(return_value=True)),
        ):
            result = await live_engine.execute_buy_via_jupiter(td, 0.01, "default")
        assert result is True
        kwargs = mocks["positions"].open_position.call_args.kwargs
        assert kwargs["amount_tokens"] == pytest.approx(2.0)
        assert kwargs["entry_price"] == pytest.approx(0.01 / 2.0)  # SOL per whole token

    @pytest.mark.asyncio
    async def test_execute_buy_via_jupiter_not_confirmed_no_position(self, live_engine, mocks):
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["jupiter"].get_quote.return_value = {"outAmount": "1000000"}
        mocks["jupiter"].get_swap_transaction.return_value = "b64tx"
        td = {"token_address": FAKE_TOKEN, "tier": "large", "symbol": "BIG"}
        with (
            patch.object(live_engine, "_sign_send_jupiter_swap", new=AsyncMock(return_value="sig")),
            patch.object(live_engine, "_confirm_transaction", new=AsyncMock(return_value=False)),
        ):
            result = await live_engine.execute_buy_via_jupiter(td, 0.01, "default")
        assert result is False
        mocks["positions"].open_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_buy_via_jupiter_uses_onchain_decimals(self, live_engine, mocks):
        """On-chain decimals override a wrong token_data hint for correct sizing."""
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["solana_client"].get_token_decimals.return_value = 9  # authoritative
        # outAmount 2e9 base units at 9 decimals -> 2.0 whole tokens (not 2000 at 6)
        mocks["jupiter"].get_quote.return_value = {"outAmount": "2000000000"}
        mocks["jupiter"].get_swap_transaction.return_value = "b64tx"
        td = {"token_address": FAKE_TOKEN, "tier": "large", "symbol": "BIG", "decimals": 6}
        with (
            patch.object(live_engine, "_sign_send_jupiter_swap", new=AsyncMock(return_value="sig")),
            patch.object(live_engine, "_confirm_transaction", new=AsyncMock(return_value=True)),
        ):
            result = await live_engine.execute_buy_via_jupiter(td, 0.01, "default")
        assert result is True
        kwargs = mocks["positions"].open_position.call_args.kwargs
        assert kwargs["amount_tokens"] == pytest.approx(2.0)  # 9 decimals used, not 6
        assert kwargs["entry_price"] == pytest.approx(0.01 / 2.0)

    @pytest.mark.asyncio
    async def test_jupiter_buy_prices_entry_from_feed(self, live_engine, mocks):
        """Entry price comes from the price feed (marks' scale), not the quote,
        when the feed is available — the fix for the phantom-PnL scale mismatch."""
        mocks["solana_client"].get_balance.return_value = 10.0
        mocks["jupiter"].get_quote.return_value = {"outAmount": "2000000"}  # 2.0 @6dp
        mocks["jupiter"].get_swap_transaction.return_value = "b64tx"
        feed_quote = MagicMock()
        feed_quote.price = 0.0042  # SOL/token from the feed, differs from 0.01/2.0
        live_engine.price_feed = MagicMock()
        live_engine.price_feed.get_price = AsyncMock(return_value=feed_quote)
        td = {"token_address": FAKE_TOKEN, "tier": "large", "symbol": "BIG", "decimals": 6}
        with (
            patch.object(live_engine, "_sign_send_jupiter_swap", new=AsyncMock(return_value="sig")),
            patch.object(live_engine, "_confirm_transaction", new=AsyncMock(return_value=True)),
        ):
            result = await live_engine.execute_buy_via_jupiter(td, 0.01, "default")
        assert result is True
        kwargs = mocks["positions"].open_position.call_args.kwargs
        assert kwargs["entry_price"] == pytest.approx(0.0042)  # feed, not quote-derived
        assert kwargs["amount_tokens"] == pytest.approx(2.0)  # size still from the quote

    @pytest.mark.asyncio
    async def test_resolve_decimals_prefers_onchain(self, live_engine, mocks):
        mocks["solana_client"].get_token_decimals.return_value = 9
        got = await live_engine._resolve_decimals({"token_address": FAKE_TOKEN, "decimals": 6})
        assert got == 9

    @pytest.mark.asyncio
    async def test_resolve_decimals_falls_back_to_hint(self, live_engine, mocks):
        mocks["solana_client"].get_token_decimals.return_value = None
        got = await live_engine._resolve_decimals({"token_address": FAKE_TOKEN, "decimals": 8})
        assert got == 8

    @pytest.mark.asyncio
    async def test_resolve_decimals_defaults_to_6(self, live_engine, mocks):
        mocks["solana_client"].get_token_decimals.return_value = None
        got = await live_engine._resolve_decimals({"token_address": FAKE_TOKEN})
        assert got == 6

    @pytest.mark.asyncio
    async def test_sell_no_curve_routes_to_jupiter(self, live_engine, mocks):
        position = _make_position()
        mocks["positions"].positions = {FAKE_TOKEN: position}
        mocks["solana_client"].get_account_info.return_value = None  # no pump curve
        with (
            patch.object(
                live_engine.pumpfun, "derive_bonding_curve_address", return_value=(MagicMock(), 0)
            ),
            patch.object(
                live_engine, "_execute_sell_via_jupiter", new=AsyncMock(return_value=True)
            ) as jup,
        ):
            result = await live_engine.execute_sell(FAKE_TOKEN, "take_profit")
        assert result is True
        jup.assert_awaited_once()
