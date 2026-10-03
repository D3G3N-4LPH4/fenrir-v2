"""Tests for fenrir/discovery/regime.py — the deterministic SOL regime classifier."""

from fenrir.discovery import regime
from fenrir.discovery.regime import (
    CHOP,
    TREND_DOWN,
    TREND_UP,
    UNKNOWN,
    classify_at,
    classify_regime,
    current_regime,
)


def _trend(n=72, drift=0.02, noise=0.003):
    """Deterministic drift with alternating noise (nonzero vol)."""
    return [100.0 * ((1 + drift) ** i) * (1 + noise * ((i % 2) * 2 - 1)) for i in range(n)]


def _chop(n=72):
    """Mean-reverting alternation: zero net drift, real vol."""
    return [100.0 * (1.005 if i % 2 == 0 else 0.995) for i in range(n)]


def test_trend_up():
    assert classify_regime(_trend(drift=0.02)) == TREND_UP


def test_trend_down():
    assert classify_regime(_trend(drift=-0.02)) == TREND_DOWN


def test_chop():
    assert classify_regime(_chop()) == CHOP


def test_flat_is_chop():
    # Zero vol: a market that doesn't move is chop by definition.
    assert classify_regime([100.0] * 72) == CHOP


def test_too_few_closes_is_unknown():
    assert classify_regime([100.0 * (1.01**i) for i in range(20)]) == UNKNOWN
    assert classify_regime([]) == UNKNOWN


def test_garbage_is_unknown():
    assert classify_regime([0.0, 0.0, 0.0] * 20) == UNKNOWN
    assert classify_regime([-1.0, -2.0] * 30) == UNKNOWN
    assert classify_regime(None) == UNKNOWN  # type: ignore[arg-type]
    assert classify_regime(["abc"] * 72) == UNKNOWN  # type: ignore[list-item]


def test_min_closes_boundary():
    assert classify_regime(_trend(n=49, drift=0.02)) == TREND_UP
    assert classify_regime(_trend(n=48, drift=0.02)) == UNKNOWN


def test_classify_at_slices_history():
    closes = _trend(n=72, drift=0.02)
    pairs = [(1_000_000.0 + i * 3600.0, c) for i, c in enumerate(closes)]
    # Full history -> trend_up.
    assert classify_at(pairs, 1_000_000.0 + 71 * 3600.0) == TREND_UP
    # Only 11 closes visible at this ts -> unknown.
    assert classify_at(pairs, 1_000_000.0 + 10 * 3600.0) == UNKNOWN
    # Before any data -> unknown.
    assert classify_at(pairs, 1_000_000.0 - 1.0) == UNKNOWN
    assert classify_at([], 1_000_000.0) == UNKNOWN


def test_classify_at_mixed_regimes():
    up = _trend(n=60, drift=0.02)
    down = [up[-1] * (0.98**i) for i in range(1, 61)]
    closes = up + down
    pairs = [(2_000_000.0 + i * 3600.0, c) for i, c in enumerate(closes)]
    assert classify_at(pairs, 2_000_000.0 + 59 * 3600.0) == TREND_UP
    assert classify_at(pairs, 2_000_000.0 + 119 * 3600.0) == TREND_DOWN


def test_current_regime_fail_open(monkeypatch):
    async def _boom(limit: int = 200):
        raise RuntimeError("network down")

    monkeypatch.setattr(regime, "fetch_sol_hourly", _boom)
    assert current_regime() == UNKNOWN


def test_current_regime_uses_fetched_closes(monkeypatch):
    async def _ok(limit: int = 200):
        return [(3_000_000.0 + i * 3600.0, c) for i, c in enumerate(_trend(drift=0.02))]

    monkeypatch.setattr(regime, "fetch_sol_hourly", _ok)
    assert current_regime() == TREND_UP


def test_regime_values_are_stable_strings():
    assert set(regime.REGIMES) == {"trend_up", "chop", "trend_down", "unknown"}
    for v in (TREND_UP, CHOP, TREND_DOWN, UNKNOWN):
        assert isinstance(v, str)


def test_z_threshold_documented():
    # Sanity: a one-daily-vol day move is the boundary. Build a series whose
    # last-24h z is just above 1 and just below 1.
    base = [100.0] * 48

    # Append 24h: drift d per hour, zero noise -> use tiny noise to keep vol > 0.
    def tail(drift):
        return [
            base[-1] * ((1 + drift) ** i) * (1 + 1e-6 * ((i % 2) * 2 - 1)) for i in range(1, 25)
        ]

    # vol per hour ~2e-6 (alternating) -> daily vol ~9.8e-6.
    assert classify_regime(base + tail(1e-6)) == TREND_UP  # z ~ 2.4
    assert classify_regime(base + tail(0.0)) == CHOP  # z ~ 0
