"""Token all-time-high from GeckoTerminal candles. Fail-open.

The scout card shows ATH distance ("-48% / 22h") — the single best context
line for whether a runner is extended or basing. One hourly-candle fetch
per call; callers must only invoke it for tokens actually being emitted
(alert candidates / misfits), never per scanned token.
"""

from __future__ import annotations

import logging
import time

from fenrir.discovery.models import Chain

logger = logging.getLogger(__name__)

GECKO_BASE = "https://api.geckoterminal.com/api/v2"
_GECKO_NETWORKS: dict[Chain, str] = {
    Chain.SOLANA: "solana",
    Chain.ROBINHOOD: "robinhood",
}

# Hourly candles, 7d window — enough to catch the local ATH for young runners.
OHLCV_LIMIT = 168
FETCH_TIMEOUT_SECONDS = 12.0


def _parse_highs(payload: object) -> list[tuple[float, float]]:
    """GeckoTerminal ohlcv payload -> sorted [(ts, high)]."""
    try:
        from fenrir.discovery.second_life import parse_ohlcv

        return sorted(
            ((ts, h) for ts, _o, h, _l, _c, _v in parse_ohlcv(payload) if h > 0),
            key=lambda r: r[0],
        )
    except Exception:  # noqa: BLE001 - fail-open
        return []


async def fetch_ath(
    chain: Chain,
    pair_address: str | None,
    price_now: float | None,
) -> dict | None:
    """Return {"ath_price", "drop_pct", "hours_ago"} or None. Never raises.

    ATH = max hourly high over the trailing 7d window. drop_pct is measured
    from the ATH to ``price_now`` (negative when below ATH).
    """
    network = _GECKO_NETWORKS.get(chain)
    if network is None or not pair_address or not price_now or price_now <= 0:
        return None
    try:
        import aiohttp

        url = (
            f"{GECKO_BASE}/networks/{network}/pools/{pair_address}"
            f"/ohlcv/hour?aggregate=1&limit={OHLCV_LIMIT}"
        )
        async with aiohttp.ClientSession(
            trust_env=True, headers={"User-Agent": "FENRIR/2.0 ath"}
        ) as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)
            ) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
    except Exception as e:  # noqa: BLE001 - fail-open
        logger.warning("ATH ohlcv failed for %s: %s", pair_address, e)
        return None
    highs = _parse_highs(payload)
    if not highs:
        return None
    ath_ts, ath_price = max(highs, key=lambda r: r[1])
    if ath_price <= 0:
        return None
    drop_pct = (price_now - ath_price) / ath_price * 100.0
    return {
        "ath_price": ath_price,
        "drop_pct": round(drop_pct, 1),
        "hours_ago": round((time.time() - ath_ts) / 3600, 1),
    }
