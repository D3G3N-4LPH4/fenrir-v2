"""Tests for fenrir.discovery.chart_patterns (pure logic; no network)."""

from fenrir.discovery.chart_patterns import (
    PatternMatch,
    _detect_triangle_wedge,
    _detect_triple_bottom,
    _detect_triple_top,
    detect_patterns,
    zigzag,
)


def _candles(closes):
    return [(float(i), c, c, c, c, 0.0) for i, c in enumerate(closes)]


class TestZigzag:
    def test_finds_peaks_and_valleys(self):
        closes = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        piv = zigzag(closes, deviation=0.3)
        kinds = [k for _, _, k in piv]
        assert "high" in kinds and "low" in kinds

    def test_flat_series_no_crash(self):
        assert zigzag([1.0] * 50) is not None

    def test_rejects_nonpositive(self):
        assert zigzag([0, 1, 2]) == []


class TestDoubleTopBottom:
    def test_double_top_detected(self):
        # two peaks ~equal with a valley between, 8% zigzag deviation
        closes = (
            [100 + i for i in range(10)]
            + [110 - i * 2 for i in range(10)]  # down to 92
            + [92 + i * 1.8 for i in range(10)]  # back to ~108
            + [108 - i for i in range(6)]
        )
        pats = detect_patterns(_candles(closes))
        ids = [p.pattern_id for p in pats]
        assert "double_top" in ids
        dt = next(p for p in pats if p.pattern_id == "double_top")
        assert dt.direction == "bearish"

    def test_double_bottom_detected(self):
        closes = (
            [110 - i for i in range(10)]
            + [100 + i * 2 for i in range(10)]
            + [118 - i * 1.8 for i in range(10)]
            + [102 + i for i in range(6)]
        )
        pats = detect_patterns(_candles(closes))
        ids = [p.pattern_id for p in pats]
        assert "double_bottom" in ids

    def test_too_short_yields_nothing(self):
        assert detect_patterns(_candles([100.0] * 10)) == []


class TestTrianglesWedges:
    def _pivots(self, seq):
        # seq: list of (price, kind)
        return [(i, p, k) for i, (p, k) in enumerate(seq)]

    def test_ascending_triangle(self):
        piv = self._pivots(
            [
                (100, "high"),
                (80, "low"),
                (101, "high"),
                (85, "low"),
                (100, "high"),
                (90, "low"),
            ]
        )
        m = _detect_triangle_wedge(piv)
        assert isinstance(m, PatternMatch)
        assert m.pattern_id == "ascending_triangle"
        assert m.direction == "bullish"

    def test_descending_triangle(self):
        piv = self._pivots(
            [
                (100, "high"),
                (80, "low"),
                (95, "high"),
                (81, "low"),
                (90, "high"),
                (80, "low"),
            ]
        )
        m = _detect_triangle_wedge(piv)
        assert m is not None and m.pattern_id == "descending_triangle"
        assert m.direction == "bearish"

    def test_symmetrical_triangle(self):
        piv = self._pivots(
            [
                (110, "high"),
                (80, "low"),
                (105, "high"),
                (85, "low"),
                (100, "high"),
                (90, "low"),
            ]
        )
        m = _detect_triangle_wedge(piv)
        assert m is not None and m.pattern_id == "symmetrical_triangle"

    def test_rising_wedge(self):
        piv = self._pivots(
            [
                (100, "high"),
                (80, "low"),
                (108, "high"),
                (92, "low"),
                (114, "high"),
                (102, "low"),
            ]
        )
        m = _detect_triangle_wedge(piv)
        assert m is not None and m.pattern_id == "rising_wedge"
        assert m.direction == "bearish"

    def test_no_pattern_when_trendless(self):
        piv = self._pivots(
            [
                (100, "high"),
                (90, "low"),
                (100, "high"),
                (90, "low"),
                (100, "high"),
                (90, "low"),
            ]
        )
        assert _detect_triangle_wedge(piv) is None


class TestTriple:
    def _pivots(self, seq):
        return [(i, p, k) for i, (p, k) in enumerate(seq)]

    def test_triple_top(self):
        piv = self._pivots(
            [
                (100, "high"),
                (85, "low"),
                (101, "high"),
                (86, "low"),
                (100, "high"),
            ]
        )
        m = _detect_triple_top(piv)
        assert m is not None and m.direction == "bearish"

    def test_triple_bottom(self):
        piv = self._pivots(
            [
                (85, "low"),
                (100, "high"),
                (84, "low"),
                (101, "high"),
                (85, "low"),
            ]
        )
        m = _detect_triple_bottom(piv)
        assert m is not None and m.direction == "bullish"

    def test_triple_top_rejects_uneven_peaks(self):
        piv = self._pivots(
            [
                (100, "high"),
                (85, "low"),
                (130, "high"),
                (86, "low"),
                (100, "high"),
            ]
        )
        assert _detect_triple_top(piv) is None
