"""
core/search.py
--------------
News search via NewsAPI with a keyed mock fallback.

Real search requires ``NEWS_API_KEY`` in the environment (free tier: 100 req/day).
When the key is absent, quota is exhausted, or any network error occurs, the
module transparently returns pre-written mock headlines so content generation
never hard-blocks on external availability.
"""

from datetime import datetime, timedelta

import requests

from core.config import settings
from core.logging_config import get_logger

log = get_logger(__name__)

_NEWS_API_URL = "https://newsapi.org/v2/everything"

_MOCK_HEADLINES: dict[str, str] = {
    "ai": (
        "OpenAI releases GPT-5 with real-time reasoning. "
        "Google DeepMind counters with Gemini Ultra 2. "
        "AI startup funding hits $50B globally in Q1."
    ),
    "crypto": (
        "Bitcoin surges past $105,000 as spot ETF inflows hit record $2B in a week. "
        "Ethereum L2 adoption triples. Analysts predict BTC to $200K next cycle."
    ),
    "fed": (
        "Federal Reserve holds rates at 5.25%; signals two cuts possible by year-end. "
        "S&P 500 climbs 1.4% on dovish Fed minutes. CPI cools to 2.8% YoY."
    ),
    "space": (
        "SpaceX Starship completes sixth test flight; full orbital insertion achieved. "
        "NASA confirms Artemis III Moon landing for 2026. Musk announces Mars crew by 2029."
    ),
    "tech": (
        "EU fines Meta €1.2B for GDPR violations. "
        "US Senate debates AI Regulation Act. Apple faces antitrust probe in 12 countries."
    ),
    "climate": (
        "Global CO2 hits record 425ppm. "
        "Renewable energy now cheaper than coal in 80% of markets. "
        "Amazon deforestation up 15% YoY."
    ),
    "markets": (
        "S&P 500 hits all-time high on strong earnings. "
        "Nvidia surpasses $3T market cap. "
        "Hedge funds rotate into defensive sectors amid macro uncertainty."
    ),
    "default": (
        "Tech giants report record Q1 earnings. "
        "Global AI investment surpasses $200B. "
        "Geopolitical tensions rattle emerging markets."
    ),
}

_KEYWORD_MAP: dict[str, list[str]] = {
    "ai":      ["ai", "openai", "gpt", "llm", "artificial"],
    "crypto":  ["crypto", "bitcoin", "btc", "ethereum", "blockchain"],
    "fed":     ["fed", "interest rate", "inflation", "recession", "cpi"],
    "space":   ["space", "spacex", "mars", "nasa", "rocket", "starship"],
    "tech":    ["regulation", "monopoly", "big tech", "surveillance", "privacy"],
    "climate": ["climate", "environment", "carbon", "nature", "deforestation"],
    "markets": ["market", "stock", "s&p", "nasdaq", "earnings", "hedge"],
}


def _select_mock(query: str) -> str:
    """
    Select the most relevant mock headline bucket for ``query``.

    Args:
        query: The search query string.

    Returns:
        A mock headline string from ``_MOCK_HEADLINES``.
    """
    lowered = query.lower()
    for bucket, keywords in _KEYWORD_MAP.items():
        if any(kw in lowered for kw in keywords):
            return _MOCK_HEADLINES[bucket]
    return _MOCK_HEADLINES["default"]


def search_news(query: str, max_results: int = 3) -> str:
    """
    Search for recent news headlines using NewsAPI with a mock fallback.

    Falls back to mock headlines when:
    - ``NEWS_API_KEY`` is not configured.
    - The API quota is exceeded (HTTP 426 / 429).
    - Any network or parsing error occurs.

    Args:
        query:       Search query string (4–8 words recommended).
        max_results: Maximum number of articles to include in the result.

    Returns:
        A pipe-separated string of ``[Source] Title. Description`` entries,
        or a single mock headline string on fallback.
    """
    if not settings.news_api_key:
        log.debug("news_mock_fallback", reason="no_api_key", query=query)
        return _select_mock(query)

    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "q":        query,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": max_results,
        "from":     from_date,
        "apiKey":   settings.news_api_key,
    }

    try:
        response = requests.get(_NEWS_API_URL, params=params, timeout=10)

        if response.status_code in (426, 429):
            log.warning("news_quota_exceeded", query=query)
            return _select_mock(query)

        response.raise_for_status()
        articles: list[dict] = response.json().get("articles", [])

        if not articles:
            log.debug("news_no_results", query=query)
            return _select_mock(query)

        lines: list[str] = []
        for article in articles[:max_results]:
            title = article.get("title", "").strip()
            description = article.get("description", "").strip()
            source = article.get("source", {}).get("name", "Unknown")
            if title:
                lines.append(f"[{source}] {title}. {description}")

        log.info("news_fetched", query=query, article_count=len(articles))
        return " | ".join(lines)

    except Exception as exc:
        log.warning("news_api_error", query=query, error=str(exc))
        return _select_mock(query)
