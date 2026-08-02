#!/usr/bin/env python3
"""
FENRIR - Portfolio-level risk

Sits ABOVE the per-strategy budget tracker (fenrir.core.budget). Where that gates
each strategy in isolation, this looks across the whole book:

  - Aggregate exposure: total live SOL across every open position.
  - Correlation exposure: too many open positions sharing one creator wallet, or
    clustered in a single launch window — a coordinated-launch / same-deployer risk
    that per-strategy limits can't see.
  - Portfolio drawdown circuit breaker: halt ALL new buys once realized PnL falls a
    configured amount below its peak, in addition to the existing per-strategy one.

Pure and self-contained (no I/O), so it works identically in simulation and live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PortfolioRiskConfig:
    """Thresholds for the portfolio-level gates. A 0 disables that gate."""

    max_total_exposure_sol: float = 0.0  # cap on summed open-position SOL (0 = off)
    max_positions_per_creator: int = 2  # coordinated-launch guard (0 = off)
    launch_window_seconds: float = 60.0  # window for "clustered" opens
    max_positions_per_window: int = 3  # opens allowed within one window (0 = off)
    max_drawdown_sol: float = 0.0  # halt buys after this drop from peak (0 = off)
    drawdown_min_trades: int = 5  # don't trip the breaker on a tiny sample


@dataclass
class OpenExposure:
    token_address: str
    strategy_id: str
    amount_sol: float
    creator: str | None
    opened_at: datetime


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    flags: list[str] = field(default_factory=list)


class PortfolioRiskManager:
    """Cross-strategy risk gate + exposure/drawdown tracker."""

    def __init__(self, config: PortfolioRiskConfig | None = None) -> None:
        self.config = config or PortfolioRiskConfig()
        self._open: dict[str, OpenExposure] = {}
        self._realized_pnl_sol: float = 0.0
        self._peak_pnl_sol: float = 0.0
        self._closed_trades: int = 0
        self._breaker_tripped: bool = False

    # ── State ──────────────────────────────────────────────────────────

    @property
    def total_exposure_sol(self) -> float:
        return sum(e.amount_sol for e in self._open.values())

    @property
    def realized_pnl_sol(self) -> float:
        return self._realized_pnl_sol

    @property
    def drawdown_sol(self) -> float:
        """How far realized PnL sits below its peak (>= 0)."""
        return max(0.0, self._peak_pnl_sol - self._realized_pnl_sol)

    @property
    def breaker_tripped(self) -> bool:
        return self._breaker_tripped

    def _positions_for_creator(self, creator: str | None) -> int:
        if not creator:
            return 0
        return sum(1 for e in self._open.values() if e.creator == creator)

    def _positions_in_window(self, now: datetime) -> int:
        window = self.config.launch_window_seconds
        return sum(1 for e in self._open.values() if (now - e.opened_at).total_seconds() <= window)

    # ── Gate ───────────────────────────────────────────────────────────

    def check(
        self,
        strategy_id: str,
        amount_sol: float,
        creator: str | None = None,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Decide whether a buy is allowed at the PORTFOLIO level.

        Called after the per-strategy budget authorizes the trade. Returns a
        RiskDecision; `flags` carries advisory notes even when allowed.
        """
        now = now or datetime.now()
        cfg = self.config
        flags: list[str] = []

        # 1. Drawdown circuit breaker — halts everything until reset.
        if self._breaker_tripped:
            return RiskDecision(False, "portfolio drawdown breaker tripped", ["drawdown_breaker"])

        # 2. Aggregate exposure cap.
        if cfg.max_total_exposure_sol > 0:
            projected = self.total_exposure_sol + amount_sol
            if projected > cfg.max_total_exposure_sol:
                return RiskDecision(
                    False,
                    f"portfolio exposure {projected:.3f} > {cfg.max_total_exposure_sol:.3f} SOL",
                    ["exposure_cap"],
                )

        # 3. Correlation: shared creator lineage.
        if cfg.max_positions_per_creator > 0 and creator:
            if self._positions_for_creator(creator) >= cfg.max_positions_per_creator:
                return RiskDecision(
                    False,
                    f"{cfg.max_positions_per_creator} open position(s) already from creator "
                    f"{creator[:8]}...",
                    ["creator_concentration"],
                )

        # 4. Correlation: launch-window clustering.
        if cfg.max_positions_per_window > 0:
            if self._positions_in_window(now) >= cfg.max_positions_per_window:
                return RiskDecision(
                    False,
                    f"{cfg.max_positions_per_window} position(s) opened within "
                    f"{cfg.launch_window_seconds:.0f}s — launch-window cluster",
                    ["window_cluster"],
                )

        return RiskDecision(True, flags=flags)

    # ── Record hooks ───────────────────────────────────────────────────

    def record_open(
        self,
        token_address: str,
        strategy_id: str,
        amount_sol: float,
        creator: str | None = None,
        now: datetime | None = None,
    ) -> None:
        self._open[token_address] = OpenExposure(
            token_address=token_address,
            strategy_id=strategy_id,
            amount_sol=amount_sol,
            creator=creator,
            opened_at=now or datetime.now(),
        )

    def record_close(self, token_address: str, pnl_sol: float) -> None:
        """Close an exposure and fold its realized PnL into the drawdown tracker."""
        self._open.pop(token_address, None)
        self._realized_pnl_sol += pnl_sol
        self._peak_pnl_sol = max(self._peak_pnl_sol, self._realized_pnl_sol)
        self._closed_trades += 1

        if (
            self.config.max_drawdown_sol > 0
            and self._closed_trades >= self.config.drawdown_min_trades
            and self.drawdown_sol >= self.config.max_drawdown_sol
        ):
            self._breaker_tripped = True

    def reset_breaker(self) -> None:
        """Re-arm after a tripped breaker; peak resets to current realized PnL."""
        self._breaker_tripped = False
        self._peak_pnl_sol = self._realized_pnl_sol

    def status(self) -> dict:
        return {
            "open_positions": len(self._open),
            "total_exposure_sol": round(self.total_exposure_sol, 6),
            "realized_pnl_sol": round(self._realized_pnl_sol, 6),
            "drawdown_sol": round(self.drawdown_sol, 6),
            "breaker_tripped": self._breaker_tripped,
            "closed_trades": self._closed_trades,
        }
