#!/usr/bin/env python3
"""
FENRIR - Logger Test Suite

Pins the UTF-8 logging fix: emoji / non-latin log lines must not raise on a
cp1252 console and must persist to the (UTF-8) log file.

Run with: pytest tests/test_logger.py -v
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from fenrir.config import BotConfig, TradingMode
from fenrir.logger import FenrirLogger, _ensure_utf8_stream


@pytest.fixture(autouse=True)
def _reset_fenrir_logger() -> Iterator[None]:
    """FenrirLogger shares the module-level 'FENRIR' + 'fenrir' loggers; reset both."""

    def _clear() -> None:
        for name in ("FENRIR", "fenrir"):
            lg = logging.getLogger(name)
            for h in list(lg.handlers):
                h.close()
                lg.removeHandler(h)
            lg.propagate = True

    _clear()
    yield
    _clear()


class _FakeStream:
    def __init__(self) -> None:
        self.encoding: str | None = None
        self.errors: str | None = None

    def reconfigure(self, *, encoding: str | None = None, errors: str | None = None) -> None:
        self.encoding = encoding
        self.errors = errors


class TestEnsureUtf8Stream:
    def test_reconfigures_to_utf8(self) -> None:
        s = _FakeStream()
        _ensure_utf8_stream(s)
        assert s.encoding == "utf-8"
        assert s.errors == "backslashreplace"

    def test_stream_without_reconfigure_is_noop(self) -> None:
        _ensure_utf8_stream(object())  # must not raise

    def test_reconfigure_error_swallowed(self) -> None:
        class Bad:
            def reconfigure(self, **_: object) -> None:
                raise ValueError("cannot")

        _ensure_utf8_stream(Bad())  # must not raise


class TestFenrirLoggerUtf8:
    def _logger(self, tmp_path: Path) -> FenrirLogger:
        cfg = BotConfig(
            mode=TradingMode.SIMULATION,
            ai_analysis_enabled=False,
            log_file=str(tmp_path / "fenrir.log"),
        )
        return FenrirLogger(cfg)

    def test_file_handler_is_utf8(self, tmp_path: Path) -> None:
        self._logger(tmp_path)
        handlers = logging.getLogger("FENRIR").handlers
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert file_handlers
        assert all(h.encoding == "utf-8" for h in file_handlers)

    def test_emoji_line_does_not_raise_and_persists(self, tmp_path: Path) -> None:
        log = self._logger(tmp_path)
        # The exact line that tripped cp1252 in the live dry-runs.
        log.info("🧠 AI Brain: OFFLINE")
        for h in logging.getLogger("FENRIR").handlers:
            h.flush()
            h.close()
        content = (tmp_path / "fenrir.log").read_text(encoding="utf-8")
        assert "🧠 AI Brain: OFFLINE" in content

    def test_package_module_loggers_reach_bot_log(self, tmp_path: Path) -> None:
        """fenrir.* module loggers (decision_engine, provider_resilience) must
        route into the bot log so served-model / fallback / AI errors are visible."""
        self._logger(tmp_path)
        pkg = logging.getLogger("fenrir")
        assert any(isinstance(h, logging.FileHandler) for h in pkg.handlers)
        assert pkg.propagate is False  # no double-emit via root

        logging.getLogger("fenrir.ai.provider_resilience").warning("Model fallback active: X")
        for h in pkg.handlers:
            h.flush()
        content = (tmp_path / "fenrir.log").read_text(encoding="utf-8")
        assert "Model fallback active: X" in content
