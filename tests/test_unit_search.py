"""
tests/test_unit_search.py
--------------------------
Unit tests for ``core.search``.

Real HTTP calls are patched out; tests verify fallback logic, keyword
routing, and result formatting without requiring a NewsAPI key.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.search import _select_mock, search_news


class TestSelectMock:
    """Tests for the internal ``_select_mock`` keyword router."""

    @pytest.mark.parametrize(
        "query,expected_snippet",
        [
            ("latest AI breakthroughs", "OpenAI"),
            ("bitcoin price today", "Bitcoin"),
            ("federal reserve interest rate", "Federal Reserve"),
            ("SpaceX Starship launch", "SpaceX"),
            ("EU tech regulation monopoly", "EU"),
            ("Amazon deforestation climate", "CO2"),
            ("S&P 500 earnings hedge funds", "S&P 500"),
            ("completely unrelated query xyz", "Tech giants"),
        ],
    )
    def test_keyword_routing(self, query: str, expected_snippet: str) -> None:
        """``_select_mock`` should select the correct bucket for each keyword group."""
        result = _select_mock(query)
        assert expected_snippet in result


class TestSearchNews:
    """Tests for ``search_news``."""

    def test_returns_mock_when_no_api_key(self) -> None:
        """``search_news`` should fall back to mock when NEWS_API_KEY is empty."""
        with patch("core.search.settings") as mock_settings:
            mock_settings.news_api_key = ""
            result = search_news("AI news")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_mock_on_quota_exceeded_426(self) -> None:
        """HTTP 426 from NewsAPI should trigger the mock fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 426

        with (
            patch("core.search.settings") as mock_settings,
            patch("core.search.requests.get", return_value=mock_response),
        ):
            mock_settings.news_api_key = "fake-key"
            result = search_news("crypto")

        assert isinstance(result, str)

    def test_returns_mock_on_quota_exceeded_429(self) -> None:
        """HTTP 429 from NewsAPI should trigger the mock fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with (
            patch("core.search.settings") as mock_settings,
            patch("core.search.requests.get", return_value=mock_response),
        ):
            mock_settings.news_api_key = "fake-key"
            result = search_news("markets")

        assert isinstance(result, str)

    def test_returns_mock_on_network_error(self) -> None:
        """Network errors should silently fall back to mock headlines."""
        with (
            patch("core.search.settings") as mock_settings,
            patch("core.search.requests.get", side_effect=ConnectionError("timeout")),
        ):
            mock_settings.news_api_key = "fake-key"
            result = search_news("space")

        assert isinstance(result, str)

    def test_formats_real_articles(self) -> None:
        """Valid NewsAPI responses should be formatted as pipe-separated strings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Big AI News",
                    "description": "AI does something amazing.",
                    "source": {"name": "TechCrunch"},
                }
            ]
        }

        with (
            patch("core.search.settings") as mock_settings,
            patch("core.search.requests.get", return_value=mock_response),
        ):
            mock_settings.news_api_key = "fake-key"
            result = search_news("AI", max_results=1)

        assert "TechCrunch" in result
        assert "Big AI News" in result

    def test_returns_mock_when_no_articles(self) -> None:
        """An empty ``articles`` list from NewsAPI should trigger the mock fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"articles": []}

        with (
            patch("core.search.settings") as mock_settings,
            patch("core.search.requests.get", return_value=mock_response),
        ):
            mock_settings.news_api_key = "fake-key"
            result = search_news("anything")

        assert isinstance(result, str)
        assert len(result) > 0
