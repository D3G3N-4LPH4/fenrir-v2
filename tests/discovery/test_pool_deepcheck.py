"""Tests for fenrir.discovery.pool_deepcheck (pure logic; no network)."""

from types import SimpleNamespace

from fenrir.discovery.pool_deepcheck import (
    LivePool,
    PoolDeepCheck,
    _decode_initialize_for_token,
    _liquidity_from_swap_data,
    _qualify,
    format_deepcheck,
    verdict_needs_deepcheck,
)


def _report(signals=(), flags=(), headline=None):
    return SimpleNamespace(
        signals=list(signals),
        safety=SimpleNamespace(risk_flags=list(flags)),
        headline=headline,
    )


class TestVerdictNeedsDeepcheck:
    def test_main_pool_signal_triggers(self):
        r = _report(signals=[("main pool empty", "high")])
        assert verdict_needs_deepcheck(r) is True

    def test_pull_liquidity_headline_triggers(self):
        r = _report(headline="dev can pull liquidity")
        assert verdict_needs_deepcheck(r) is True

    def test_risk_flag_triggers(self):
        r = _report(flags=["LP unlocked — dev can pull liquidity"])
        assert verdict_needs_deepcheck(r) is True

    def test_benign_report_no_trigger(self):
        r = _report(
            signals=[("mint disabled", "low")],
            flags=["high buy tax"],
            headline="Caution — concentrated holders",
        )
        assert verdict_needs_deepcheck(r) is False

    def test_empty_report_no_trigger(self):
        assert verdict_needs_deepcheck(_report()) is False
        assert verdict_needs_deepcheck(None) is False
        assert verdict_needs_deepcheck(object()) is False


def _pool(pid, live_liq, max_liq=None, block=100):
    return LivePool(
        pool_id=pid,
        block_number=block,
        base_address="0xbase",
        swaps=5,
        last_liquidity=live_liq,
        max_liquidity=max_liq if max_liq is not None else live_liq,
        yanked=(max_liq or 0) > 0 and live_liq == 0,
    )


class TestQualify:
    def test_no_pools_unknown(self):
        assert _qualify(PoolDeepCheck(token_address="0xabc")) == "UNKNOWN"

    def test_all_empty_confirmed(self):
        dc = PoolDeepCheck(token_address="0xabc")
        dc.pools = [_pool("0x1", 0), _pool("0x2", 0)]
        dc.canonical_pool_id = "0x1"
        assert _qualify(dc) == "CONFIRMED"

    def test_canonical_empty_but_live_misattributed(self):
        # The MOONLET shape: verdict looked at the wrong pool.
        dc = PoolDeepCheck(token_address="0xabc")
        dc.pools = [_pool("0x1", 0, block=100), _pool("0x2", 999, block=200)]
        dc.canonical_pool_id = "0x1"
        assert _qualify(dc) == "MISATTRIBUTED"

    def test_canonical_live_confirmed(self):
        dc = PoolDeepCheck(token_address="0xabc")
        dc.pools = [_pool("0x1", 500, block=100), _pool("0x2", 999, block=200)]
        dc.canonical_pool_id = "0x1"
        assert _qualify(dc) == "CONFIRMED"

    def test_yanked_pool_pullable(self):
        dc = PoolDeepCheck(token_address="0xabc")
        dc.pools = [_pool("0x1", 0, max_liq=777, block=100)]
        dc.canonical_pool_id = "0x1"
        dc.yanked_pool_ids = ["0x1"]
        assert _qualify(dc) == "PULLABLE"

    def test_eoa_lp_pullable(self):
        dc = PoolDeepCheck(token_address="0xabc")
        dc.pools = [_pool("0x1", 500, block=100)]
        dc.canonical_pool_id = "0x1"
        dc.pullable_pool_ids = ["0x1"]
        assert _qualify(dc) == "PULLABLE"


class TestSwapDecoding:
    def test_topic0_matches_chain(self):
        # Observed live on Robinhood PoolManager 2026-10-01 (3282 hits in a
        # 2k-block window). Guards against a repeat of the missing-fee-param
        # bug that made every pool read as zero swaps.
        from fenrir.discovery.pool_deepcheck import _swap_topic0

        assert (
            _swap_topic0() == "0x40e9cecb9f5f1f1c5b9c97dec2917b7ee92e57ba5563708daca94dd84ad7112f"
        )

    def test_liquidity_word(self):
        # v4 Swap data: amount0, amount1, sqrtPriceX96, liquidity, tick, fee
        words = [0, 0, 0, 123456789, 0, 0]
        data = "0x" + b"".join(w.to_bytes(32, "big") for w in words).hex()
        assert _liquidity_from_swap_data(data) == 123456789

    def test_short_data_none(self):
        assert _liquidity_from_swap_data("0x1234") is None
        assert _liquidity_from_swap_data(None) is None


class TestInitializeDecode:
    def test_token_on_either_side_kept(self):
        from fenrir.discovery.providers.rh_onchain import INITIALIZE_TOPIC0

        token = "0x" + "ab" * 20
        other = "0x" + "cd" * 20
        log = {
            "topics": [
                INITIALIZE_TOPIC0,
                "0x" + "11" * 32,
                "0x" + "00" * 12 + token[2:],
                "0x" + "00" * 12 + other[2:],
            ],
            "blockNumber": "0x64",
        }
        p = _decode_initialize_for_token(log, token.lower())
        assert p is not None
        assert p["pool_id"] == "0x" + "11" * 32
        assert p["block_number"] == 100
        assert p["base_address"].lower() == other.lower()

    def test_unrelated_pool_dropped(self):
        from fenrir.discovery.providers.rh_onchain import INITIALIZE_TOPIC0

        log = {
            "topics": [
                INITIALIZE_TOPIC0,
                "0x" + "11" * 32,
                "0x" + "00" * 12 + "ab" * 20,
                "0x" + "00" * 12 + "cd" * 20,
            ],
            "blockNumber": "0x64",
        }
        assert _decode_initialize_for_token(log, "0x" + "ff" * 20) is None


class TestFormat:
    def test_sections(self):
        dc = PoolDeepCheck(
            token_address="0xabc",
            qualification="UNKNOWN",
            detail="no v4 pools found for token in scan window",
        )
        assert "inconclusive" in format_deepcheck(dc)
        dc.qualification = "CONFIRMED"
        dc.pools = [_pool("0x1", 0)]
        assert "verdict holds" in format_deepcheck(dc)
        dc.qualification = "MISATTRIBUTED"
        dc.pools = [_pool("0x1", 0, block=100), _pool("0x2", 9, block=200)]
        dc.canonical_pool_id = "0x1"
        assert "wrong pool" in format_deepcheck(dc)
        dc.qualification = "PULLABLE"
        dc.pullable_pool_ids = ["0x2"]
        assert "CONFIRMED" in format_deepcheck(dc)
