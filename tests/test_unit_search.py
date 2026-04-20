"""
tests/test_unit_search.py — unit tests for core.search
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from core.search import _select_mock, search_news


class TestSelectMock:
    @pytest.mark.parametrize("query,snippet", [
        ("latest AI breakthroughs", "OpenAI"),
        ("bitcoin price today", "Bitcoin"),
        ("federal reserve interest rate", "Federal Reserve"),
        ("SpaceX Starship launch", "SpaceX"),
        ("EU tech regulation monopoly", "EU"),
        ("Amazon deforestation climate", "CO2"),
        ("S&P 500 earnings hedge funds", "S&P 500"),
        ("completely unrelated query xyz", "Tech giants"),
    ])
    def test_keyword_routing(self, query, snippet):
        assert snippet in _select_mock(query)


class TestSearchNews:
    def test_returns_mock_when_no_api_key(self):
        with patch("core.search.settings") as s:
            s.news_api_key = ""
            result = search_news("AI news")
        assert isinstance(result, str) and len(result) > 0

    def test_returns_mock_on_quota_exceeded(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with (
            patch("core.search.settings") as s,
            patch("core.search.requests.get", return_value=mock_resp),
        ):
            s.news_api_key = "fake"
            result = search_news("crypto")
        assert isinstance(result, str)

    def test_returns_mock_on_network_error(self):
        with (
            patch("core.search.settings") as s,
            patch("core.search.requests.get", side_effect=ConnectionError("timeout")),
        ):
            s.news_api_key = "fake"
            result = search_news("space")
        assert isinstance(result, str)

    def test_formats_real_articles(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"articles": [{
            "title": "Big AI News",
            "description": "AI does something amazing.",
            "source": {"name": "TechCrunch"},
        }]}
        with (
            patch("core.search.settings") as s,
            patch("core.search.requests.get", return_value=mock_resp),
        ):
            s.news_api_key = "fake"
            result = search_news("AI", max_results=1)
        assert "TechCrunch" in result
        assert "Big AI News" in result

    def test_returns_mock_when_no_articles(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"articles": []}
        with (
            patch("core.search.settings") as s,
            patch("core.search.requests.get", return_value=mock_resp),
        ):
            s.news_api_key = "fake"
            result = search_news("anything")
        assert len(result) > 0
