#!/usr/bin/env python3
"""
FENRIR - Market-Making Paper Session (Phase 4.3b, sim-against-real-data)

Runs the proven market-making inventory primitive (``MarketMaker``) against a LIVE
price series — sampling the real price feed — without ever placing an order. The
point is to learn, on real memecoin price action, whether the maker would capture
spread (mean-reverting noise) or bleed via inventory risk (trends), so the decision
to build live quoting is driven by data rather than faith.

Read-only by construction: the session only ever calls ``MarketMaker.step`` (paper
fills) and reads prices. It has no reference to the trading engine or a wallet.

Wiring: pass ``price_source_from_feed(price_feed, mint)`` as the source in
production. Tests inject a scripted async source, so the streaming path is verified
to match the batch simulator with no network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fenrir.trading.market_maker import Fill, MarketMaker, MarketMakerConfig

PriceSource = Callable[[], Awaitable[float | None]]


def price_source_from_feed(price_feed: Any, token_mint: str) -> PriceSource:
    """Build a read-only price source from the aggregated price feed.

    ``price_feed.get_price`` returns an AggregatedPrice whose ``.price`` is
    SOL-per-token; None when no fresh quote is available. This closure is the only
    bridge to real data — it reads, never trades.
    """

    async def _get() -> float | None:
        quote = await price_feed.get_price(token_mint)
        return quote.price if quote is not None else None

    return _get


class MarketMakingPaperSession:
    """A live paper-trading session for one token.

    Feed it prices (via :meth:`on_price`, or drive it with :meth:`run` over a price
    source) and it streams them through a ``MarketMaker`` in paper mode, tracking the
    outcome. Nothing here can execute a real trade.
    """

    def __init__(
        self,
        token_address: str,
        maker: MarketMaker | None = None,
        config: MarketMakerConfig | None = None,
        starting_cash_sol: float | None = None,
        logger: Any = None,
    ) -> None:
        self.token_address = token_address
        self.maker = maker or MarketMaker(config or MarketMakerConfig())
        self.maker.reset(starting_cash_sol)
        self._logger = logger
        self.ticks = 0
        self.skipped = 0  # non-positive / missing prices

    def on_price(self, price: float | None) -> Fill | None:
        """Feed one observed price. Missing/non-positive prices are skipped (a feed
        can return None); a valid price advances the paper maker by one step."""
        if price is None or price <= 0:
            self.skipped += 1
            return None
        self.ticks += 1
        return self.maker.step(price)

    def report(self) -> dict:
        """Current paper-trading summary (marked at the last price seen)."""
        res = self.maker.result()
        round_trips = min(res.buys, res.sells)
        return {
            "token_address": self.token_address,
            "ticks": self.ticks,
            "skipped": self.skipped,
            "buys": res.buys,
            "sells": res.sells,
            "round_trips": round_trips,
            "realized_pnl_sol": res.realized_pnl_sol,
            "unrealized_pnl_sol": res.unrealized_pnl_sol,
            "total_pnl_sol": res.total_pnl_sol,
            "ending_inventory_tokens": res.ending_inventory_tokens,
            "ending_inventory_value_sol": res.ending_inventory_value_sol,
            "ending_cash_sol": res.ending_cash_sol,
            "max_inventory_value_sol": res.max_inventory_value_sol,
            "simulation": True,  # always — this session never trades live
        }

    async def run(
        self,
        source: PriceSource,
        ticks: int,
        interval_seconds: float = 2.0,
    ) -> dict:
        """Poll ``source`` for ``ticks`` samples at ``interval_seconds`` apart,
        streaming each through the paper maker, and return the final report.

        Read-only: ``source`` yields prices; the session never places an order. In
        production ``source`` is :func:`price_source_from_feed`.
        """
        for _ in range(ticks):
            try:
                price = await source()
            except Exception as e:  # noqa: BLE001 - a feed hiccup must not kill the session
                self._log("warning", f"paper MM {self.token_address[:8]}...: price error: {e}")
                price = None
            self.on_price(price)  # None → counted as skipped here
            if interval_seconds > 0:
                await asyncio.sleep(interval_seconds)

        report = self.report()
        self._log(
            "info",
            f"paper MM {self.token_address[:8]}... | ticks={report['ticks']} "
            f"rt={report['round_trips']} realized={report['realized_pnl_sol']:.6f} "
            f"total={report['total_pnl_sol']:.6f} SOL",
        )
        return report

    def _log(self, level: str, msg: str) -> None:
        if self._logger is None:
            return
        method = getattr(self._logger, level, None)
        try:
            if callable(method):
                method(msg)
                return
        except TypeError:
            pass
        fallback = getattr(self._logger, "warning", None) or getattr(self._logger, "info", None)
        if callable(fallback):
            fallback(msg)
