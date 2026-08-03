#!/usr/bin/env python3
"""
FENRIR - Signal Backtester (Phase 6, backtesting rigor)

Replays historical candidates through the SAME code that trades live: the strategy's
own ``evaluate_token``, the unified ``normalize_signal`` (Phase 5), and the strategy's
own ``TradeParams`` exit rules. Nothing here re-implements entry/exit logic, so there is
no backtest-vs-production drift — which is the whole point of "rigor". A strategy tweak
is measured by exactly the logic that will run.

Entry: ``strategy.evaluate_token(token_data, market_data)`` on the sample's entry
snapshot; a non-None signal enters at the sample's entry price.
Exit: simulate the forward price path, applying the strategy's stop-loss, trailing stop,
take-profit, and max-hold — stops checked before targets (pessimistic).

Signal (market-data) strategies only: their ``evaluate_token`` is synchronous and
self-contained. Classic strategies (async ``should_evaluate`` + AI) are out of scope for
the offline backtester.
"""

from __future__ import annotations

from typing import Any

from fenrir.backtest.metrics import compute_metrics
from fenrir.backtest.models import BacktestResult, BacktestSample, BacktestTrade
from fenrir.signals.adapters import normalize_signal


class SignalBacktester:
    """Offline, deterministic backtester over the real strategy + signal path."""

    def run(self, strategy: Any, samples: list[BacktestSample]) -> BacktestResult:
        trades: list[BacktestTrade] = []
        evaluated = 0
        entered = 0

        for sample in samples:
            if not sample.forward_prices:
                continue
            evaluated += 1

            signal = strategy.evaluate_token(sample.token_data, sample.market_data)
            if signal is None:
                continue

            entered += 1
            strength = normalize_signal(signal).strength  # unified conviction axis
            trades.append(self._simulate_exit(strategy, sample, strength))

        return BacktestResult(
            strategy_id=getattr(strategy, "strategy_id", "unknown"),
            trades=trades,
            metrics=compute_metrics(trades),
            samples_evaluated=evaluated,
            samples_entered=entered,
        )

    def _simulate_exit(
        self, strategy: Any, sample: BacktestSample, strength: float
    ) -> BacktestTrade:
        params = strategy.get_trade_params()
        prices = sample.forward_prices
        entry = prices[0]

        tp_price = entry * (1.0 + params.take_profit_pct / 100.0)
        sl_price = entry * (1.0 - params.stop_loss_pct / 100.0)
        frame_seconds = sample.frame_seconds if sample.frame_seconds > 0 else 60.0
        max_frames = int(params.max_position_age_minutes * 60.0 / frame_seconds)

        peak = entry
        exit_price = prices[-1]
        exit_reason = "end_of_data"
        hold_frames = len(prices) - 1

        for i in range(1, len(prices)):
            price = prices[i]
            peak = max(peak, price)
            trail_price = peak * (1.0 - params.trailing_stop_pct / 100.0)

            # Stops before targets — pessimistic on a bar that could be read either way.
            if price <= sl_price:
                exit_price, exit_reason, hold_frames = price, "stop_loss", i
                break
            if price <= trail_price:
                exit_price, exit_reason, hold_frames = price, "trailing_stop", i
                break
            if price >= tp_price:
                exit_price, exit_reason, hold_frames = price, "take_profit", i
                break
            if max_frames > 0 and i >= max_frames:
                exit_price, exit_reason, hold_frames = price, "max_hold", i
                break

        pnl_pct = (exit_price - entry) / entry * 100.0 if entry > 0 else 0.0

        return BacktestTrade(
            token_address=sample.token_address,
            strategy_id=getattr(strategy, "strategy_id", "unknown"),
            entry_price=entry,
            exit_price=exit_price,
            exit_reason=exit_reason,
            hold_frames=hold_frames,
            pnl_pct=pnl_pct,
            strength=strength,
            symbol=sample.symbol,
        )
