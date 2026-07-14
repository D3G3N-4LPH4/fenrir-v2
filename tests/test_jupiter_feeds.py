#!/usr/bin/env python3
"""
FENRIR - Jupiter token-feed URL construction tests

The ``recent`` category is a newly-created-pairs feed that takes NO interval
(``/tokens/v2/recent``); the interval-scoped categories keep ``/{interval}``.
Verified by capturing the URL passed to a fake aiohttp session — no network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from fenrir.core.jupiter import JupiterSwapEngine


class _FakeResp:
    status = 200

    async def json(self) -> list[Any]:
        return []

    async def __aenter__(self) -> _FakeResp:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False


class _FakeSession:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str) -> _FakeResp:
        self.urls.append(url)
        return _FakeResp()


def _engine_with_session() -> tuple[JupiterSwapEngine, _FakeSession]:
    eng = JupiterSwapEngine(config=MagicMock(), logger=MagicMock())
    session = _FakeSession()
    eng.session = session  # type: ignore[assignment]
    return eng, session


class TestTrendingFeedUrls:
    @pytest.mark.asyncio
    async def test_recent_omits_interval(self) -> None:
        eng, session = _engine_with_session()
        await eng.get_trending_tokens("recent", "24h")
        assert session.urls[-1].endswith("/tokens/v2/recent")

    @pytest.mark.asyncio
    async def test_interval_category_keeps_interval(self) -> None:
        eng, session = _engine_with_session()
        await eng.get_trending_tokens("toptrending", "1h")
        assert session.urls[-1].endswith("/tokens/v2/toptrending/1h")
