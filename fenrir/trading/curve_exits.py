"""Curve-account exits. The poll is the backstop, not the stop.

Subscribe to each open bonding-curve account. A stop-loss on a curve
position is mechanical: the brain may tighten a trailing stop, it may not
hold through the hard floor. Call nudge_exits() so the loop does not wait
out the poll.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from solders.pubkey import Pubkey


class CurveExitWatcher:
    def __init__(self, client, positions, on_stop: Callable[[str, str], Awaitable[None]], nudge):
        self.client = client
        self.positions = positions
        self.on_stop = on_stop
        self.nudge = nudge
        self._tasks: dict[str, asyncio.Task] = {}

    def watch(self, token_address: str, curve_address: str, stop_loss_pct: float) -> None:
        if token_address in self._tasks:
            return
        self._tasks[token_address] = asyncio.create_task(
            self._run(token_address, curve_address, stop_loss_pct)
        )

    def unwatch(self, token_address: str) -> None:
        task = self._tasks.pop(token_address, None)
        if task:
            task.cancel()

    def stop(self) -> None:
        for token_address in list(self._tasks):
            self.unwatch(token_address)

    async def _run(self, token_address: str, curve_address: str, stop_loss_pct: float) -> None:
        pubkey = Pubkey.from_string(curve_address)
        async for account in self.client.account_subscribe(pubkey):
            position = self.positions.positions.get(token_address)
            if position is None:
                return
            price = account.get_price()
            if price <= 0:
                continue
            position.update_price(price)
            self.nudge()
            if position.should_stop_loss(stop_loss_pct):
                pnl = position.get_pnl_percent()
                await self.on_stop(token_address, f"Curve stop: {pnl:.2f}%")
                return
