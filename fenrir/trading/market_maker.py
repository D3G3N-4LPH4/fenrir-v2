#!/usr/bin/env python3
"""
FENRIR - Market-Making Inventory Primitive (Phase 4, read-only / simulation)

The first slice of the market-making strategy. Market-making is not a one-shot
directional trade — it continuously quotes a bid and an ask around fair value,
earns the spread when both sides fill, and manages *inventory* so it does not
accumulate a large one-sided position and get run over.

pump.fun bonding curves and graduated AMMs have no central limit order book, so
"market-making" here is spread capture around a reservation price with inventory
skew (an Avellaneda-Stoikov-lite model): as inventory grows past target the quotes
are pulled down so the ask fills first (shed inventory) and the bid backs off; as
inventory falls below target the quotes lift so the bid fills first (rebuild).

This module is deliberately **execution-free**. It computes quotes and runs a
deterministic paper-trading loop (`simulate`) over an observed mid-price series to
prove the machinery — inventory stays bounded, oscillating markets capture spread,
one-sided markets stop at the inventory cap. Nothing here signs a transaction or
touches the engine; wiring live quoting/execution is a later PR (gated + opt-in).
"""

from __future__ import annotations

from dataclasses import dataclass, field

_BPS = 10_000.0


@dataclass
class MarketMakerConfig:
    """Tunable parameters for the market-making inventory model.

    All spreads/skews are in basis points (1 bp = 0.01%). Sizes/limits are in SOL.
    """

    # Total quoted spread (bid→ask), split evenly around the reservation price.
    spread_bps: int = 200  # 2% — memecoin volatility warrants a wide spread
    # Never quote tighter than this (protects against fee/slippage bleed).
    min_spread_bps: int = 60
    # Max quote shift from inventory being fully at the cap (Avellaneda skew scale).
    inventory_skew_bps: int = 150
    # Notional per quote/fill.
    order_size_sol: float = 0.05
    # Hard cap on inventory value (SOL) — the primitive never buys beyond this.
    max_inventory_sol: float = 1.0
    # Target inventory as a fraction of the cap (0.5 = balanced book).
    inventory_target_ratio: float = 0.5
    # Fair-value EMA smoothing for the simulator: the maker quotes around a SLOW
    # fair value while fast trade prints hit the resting quotes. Lower = slower fair
    # value = more spread capture on noise but more inventory risk on trends.
    fair_value_ema_alpha: float = 0.15

    def __post_init__(self) -> None:
        if self.spread_bps < self.min_spread_bps:
            self.spread_bps = self.min_spread_bps


@dataclass
class Quote:
    """A two-sided quote around a reservation price."""

    mid: float
    reservation: float
    bid_price: float
    ask_price: float
    skew_bps: float  # signed inventory skew applied to the reservation
    inventory_deviation: float  # (value - target) / cap, ~[-ratio, 1-ratio]

    @property
    def spread_bps(self) -> float:
        return (self.ask_price - self.bid_price) / self.mid * _BPS if self.mid > 0 else 0.0


@dataclass
class InventoryState:
    """Mark-to-market inventory + cash for the maker."""

    base_tokens: float = 0.0  # tokens currently held
    avg_entry_price: float = 0.0  # SOL per token, cost basis of the held tokens
    cash_sol: float = 0.0  # uninvested SOL
    realized_pnl_sol: float = 0.0  # locked-in spread capture

    def value_sol(self, mid: float) -> float:
        """Mark-to-market value of held tokens at ``mid``."""
        return self.base_tokens * mid

    def unrealized_pnl_sol(self, mid: float) -> float:
        return self.base_tokens * (mid - self.avg_entry_price)


@dataclass
class Fill:
    """A single simulated fill."""

    step: int
    side: str  # "buy" or "sell"
    price: float
    size_tokens: float
    size_sol: float


@dataclass
class MarketMakerSimResult:
    """Outcome of a paper-trading run."""

    fills: list[Fill] = field(default_factory=list)
    realized_pnl_sol: float = 0.0
    unrealized_pnl_sol: float = 0.0
    ending_inventory_tokens: float = 0.0
    ending_inventory_value_sol: float = 0.0
    ending_cash_sol: float = 0.0
    max_inventory_value_sol: float = 0.0

    @property
    def buys(self) -> int:
        return sum(1 for f in self.fills if f.side == "buy")

    @property
    def sells(self) -> int:
        return sum(1 for f in self.fills if f.side == "sell")

    @property
    def total_pnl_sol(self) -> float:
        return self.realized_pnl_sol + self.unrealized_pnl_sol


class MarketMaker:
    """Inventory-aware quoting model + deterministic paper-trading simulator.

    Execution-free: ``quote`` computes prices, ``record_fill`` updates inventory,
    and ``simulate`` runs a fill model over a mid-price series. No network, no
    signing. ``simulation`` is fixed True for now and asserted, so a future live
    path must consciously flip it.
    """

    def __init__(self, config: MarketMakerConfig | None = None, simulation: bool = True) -> None:
        self.config = config or MarketMakerConfig()
        # This primitive is read-only. Live quoting/execution is a later, gated PR.
        if not simulation:
            raise NotImplementedError(
                "MarketMaker live execution is not implemented — this is the "
                "read-only/simulation inventory primitive."
            )
        self.simulation = True
        self.inventory = InventoryState()
        # Online paper-trading state (see reset/step). fair is the slow EMA the maker
        # quotes around; None until the first tick seeds it.
        self.fair: float | None = None
        self._fills: list[Fill] = []
        self._max_inv_value_sol = 0.0
        self._last_price = 0.0
        self._step_index = 0

    # ── Quoting ────────────────────────────────────────────────────────

    def inventory_deviation(self, mid: float) -> float:
        """Signed inventory deviation from target, normalized by the cap.

        0 at target, positive when over-inventoried, negative when under. Ranges
        roughly [-target_ratio, 1 - target_ratio].
        """
        cap = self.config.max_inventory_sol
        if cap <= 0:
            return 0.0
        target_value = self.config.inventory_target_ratio * cap
        return (self.inventory.value_sol(mid) - target_value) / cap

    def quote(self, mid: float) -> Quote:
        """Compute a two-sided quote around an inventory-skewed reservation price.

        reservation = mid * (1 - skew);  skew = inventory_skew_bps * deviation.
        Over-inventoried (deviation > 0) pulls the reservation *down* so the ask sits
        closer to mid (sheds inventory) and the bid backs off; under-inventoried lifts
        it so the bid fills first (rebuilds).
        """
        if mid <= 0:
            raise ValueError("mid price must be positive")

        deviation = self.inventory_deviation(mid)
        skew_bps = self.config.inventory_skew_bps * deviation
        reservation = mid * (1.0 - skew_bps / _BPS)

        half_spread_bps = self.config.spread_bps / 2.0
        bid_price = reservation * (1.0 - half_spread_bps / _BPS)
        ask_price = reservation * (1.0 + half_spread_bps / _BPS)

        return Quote(
            mid=mid,
            reservation=reservation,
            bid_price=bid_price,
            ask_price=ask_price,
            skew_bps=skew_bps,
            inventory_deviation=deviation,
        )

    # ── Inventory bookkeeping ──────────────────────────────────────────

    def can_buy(self, mid: float) -> bool:
        """Whether adding one order keeps inventory value under the cap and we have
        the cash for it."""
        cfg = self.config
        if self.inventory.cash_sol < cfg.order_size_sol:
            return False
        return self.inventory.value_sol(mid) + cfg.order_size_sol <= cfg.max_inventory_sol + 1e-9

    def record_fill(self, side: str, price: float, size_sol: float) -> Fill | None:
        """Apply a fill to inventory. ``buy`` spends SOL for tokens (weighted-avg cost
        basis); ``sell`` returns tokens for SOL and realizes PnL vs. cost basis.

        Returns the Fill, or None if it could not be executed (no cash / no tokens).
        """
        inv = self.inventory
        if price <= 0 or size_sol <= 0:
            return None

        if side == "buy":
            if inv.cash_sol < size_sol:
                return None
            tokens = size_sol / price
            new_base = inv.base_tokens + tokens
            # Weighted-average cost basis.
            inv.avg_entry_price = (
                (inv.base_tokens * inv.avg_entry_price + tokens * price) / new_base
                if new_base > 0
                else 0.0
            )
            inv.base_tokens = new_base
            inv.cash_sol -= size_sol
            return Fill(step=-1, side="buy", price=price, size_tokens=tokens, size_sol=size_sol)

        if side == "sell":
            if inv.base_tokens <= 0:
                return None
            tokens = min(inv.base_tokens, size_sol / price)
            proceeds = tokens * price
            inv.realized_pnl_sol += tokens * (price - inv.avg_entry_price)
            inv.base_tokens -= tokens
            inv.cash_sol += proceeds
            if inv.base_tokens <= 1e-18:
                inv.base_tokens = 0.0
                inv.avg_entry_price = 0.0
            return Fill(step=-1, side="sell", price=price, size_tokens=tokens, size_sol=proceeds)

        raise ValueError(f"unknown side: {side!r}")

    # ── Paper trading ──────────────────────────────────────────────────

    def reset(self, starting_cash_sol: float | None = None) -> None:
        """Reset all paper-trading state for a fresh run/session."""
        self.inventory = InventoryState(
            cash_sol=(
                starting_cash_sol
                if starting_cash_sol is not None
                else self.config.max_inventory_sol
            )
        )
        self.fair = None
        self._fills = []
        self._max_inv_value_sol = 0.0
        self._last_price = 0.0
        self._step_index = 0

    def step(self, print_price: float) -> Fill | None:
        """Advance the paper maker by one trade print (online / streaming).

        The maker quotes around a SLOW fair value (an EMA of prints), not around each
        jumpy print — that is what lets it capture spread on mean-reverting noise
        rather than getting adversely selected. A print through the bid fills a buy (at
        the bid); through the ask fills a sell (at the ask); at most one fill per print.
        Buys respect the inventory cap and cash, sells respect tokens held. The fair
        value updates *after* quoting. Returns the Fill for this tick, or None.

        Identical semantics to the batch ``simulate``; this is the unit used both by
        ``simulate`` and by the live paper session (sim-against-real-data), which never
        places an order.
        """
        if print_price <= 0:
            return None
        if self.fair is None:
            self.fair = print_price  # seed the fair value on the first tick
        if self.fair <= 0:
            return None

        q = self.quote(self.fair)
        fill: Fill | None = None
        if print_price <= q.bid_price and self.can_buy(self.fair):
            fill = self.record_fill("buy", q.bid_price, self.config.order_size_sol)
        elif print_price >= q.ask_price and self.inventory.base_tokens > 0:
            fill = self.record_fill("sell", q.ask_price, self.config.order_size_sol)

        if fill is not None:
            fill.step = self._step_index
            self._fills.append(fill)

        self._max_inv_value_sol = max(self._max_inv_value_sol, self.inventory.value_sol(self.fair))
        # Update the fair value AFTER quoting on this print.
        alpha = self.config.fair_value_ema_alpha
        self.fair = (1.0 - alpha) * self.fair + alpha * print_price
        self._last_price = print_price
        self._step_index += 1
        return fill

    def result(self, mark_price: float | None = None) -> MarketMakerSimResult:
        """Snapshot the paper-trading outcome, marking open inventory at ``mark_price``
        (defaults to the last print seen)."""
        mark = mark_price if mark_price is not None else self._last_price
        return MarketMakerSimResult(
            fills=list(self._fills),
            realized_pnl_sol=self.inventory.realized_pnl_sol,
            unrealized_pnl_sol=self.inventory.unrealized_pnl_sol(mark),
            ending_inventory_tokens=self.inventory.base_tokens,
            ending_inventory_value_sol=self.inventory.value_sol(mark),
            ending_cash_sol=self.inventory.cash_sol,
            max_inventory_value_sol=self._max_inv_value_sol,
        )

    def simulate(
        self, prices: list[float], starting_cash_sol: float | None = None
    ) -> MarketMakerSimResult:
        """Batch paper-trading over a print series — a thin wrapper over reset/step.

        Properties this proves (see tests): oscillation around a stable level captures
        spread (realized PnL > 0); a sustained trend accumulates inventory only up to
        the cap and then stops (bounded inventory, honest adverse-selection loss).
        """
        self.reset(starting_cash_sol)
        if not prices:
            return self.result(mark_price=0.0)
        for print_price in prices:
            self.step(print_price)
        return self.result(mark_price=prices[-1])
