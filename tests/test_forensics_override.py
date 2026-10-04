"""Tests for the forensics-priority holder override in tools/evaluate.py.

darwin case 2026-10-04: the provider said top holder 23.0% (every
concentration-capped filter failed); the direct on-chain read was 2.5%.
The on-chain forensics read (owner-aggregated, vaults excluded) must win,
and it must be applied after the concurrent provider legs so a slow
provider write can't clobber it.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fenrir.discovery.solana_forensics import SolanaForensicsReport  # noqa: E402
from tools.evaluate import _apply_forensics_override  # noqa: E402


def _snap(top=None, top10=None):
    return SimpleNamespace(top_holder_pct=top, top10_holder_pct=top10)


def _report(top=2.54, top10=23.03):
    return SolanaForensicsReport(
        top_holder_pct=top,
        top10_holder_pct=top10,
        holders_measured=19,
        excluded_vault_pct=12.73,
        detail="top holder 2.5%, top-10 23.0%",
    )


def test_override_replaces_provider_figures():
    snap = _snap(top=23.0, top10=40.0)
    notes: list[str] = []
    _apply_forensics_override(snap, _report(), notes)
    assert snap.top_holder_pct == pytest.approx(2.54)
    assert snap.top10_holder_pct == pytest.approx(23.03)


def test_override_note_documents_large_discrepancy():
    snap = _snap(top=23.0, top10=40.0)
    notes: list[str] = []
    _apply_forensics_override(snap, _report(), notes)
    assert any("23.0%" in n and "2.5%" in n for n in notes), notes


def test_no_note_for_close_agreement():
    snap = _snap(top=2.6, top10=23.0)
    notes: list[str] = []
    _apply_forensics_override(snap, _report(), notes)
    assert snap.top_holder_pct == pytest.approx(2.54)
    assert not any("provider" in n for n in notes), notes


def test_fills_when_provider_missing():
    snap = _snap(top=None, top10=None)
    notes: list[str] = []
    _apply_forensics_override(snap, _report(), notes)
    assert snap.top_holder_pct == pytest.approx(2.54)
    assert snap.top10_holder_pct == pytest.approx(23.03)


def test_none_report_is_noop():
    snap = _snap(top=23.0, top10=40.0)
    notes: list[str] = []
    _apply_forensics_override(snap, None, notes)
    assert snap.top_holder_pct == 23.0
    assert snap.top10_holder_pct == 40.0
    assert notes == []
