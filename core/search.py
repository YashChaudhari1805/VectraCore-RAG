"""
core/search.py
--------------
Real news search via NewsAPI (free tier: 100 req/day).
Falls back to mock headlines if API key is missing or quota is exceeded.

Get a free key at: https://newsapi.org/register
Add to .env: NEWS_API_KEY=your_key_here
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# ── Fallback mock headlines (used when no API key or quota exceeded) ──────────

MOCK_HEADLINES = {
    "ai":         "OpenAI releases GPT-5 with real-time reasoning. Google DeepMind counters with Gemini Ultra 2. AI startup funding hits $50B globally in Q1.",
    "crypto":     "Bitcoin surges past $105,000 as spot ETF inflows hit record $2B in a week. Ethereum L2 adoption triples. Analysts predict BTC to $200K next cycle.",
    "fed":        "Federal Reserve holds rates at 5.25%; signals two cuts possible by year-end. S&P 500 climbs 1.4% on dovish Fed minutes. CPI cools to 2.8% YoY.",
    "space":      "SpaceX Starship completes sixth test flight; full orbital insertion achieved. NASA confirms Artemis III Moon landing for 2026. Musk announces Mars crew by 2029.",
    "tech":       "EU fines Meta €1.2B for GDPR violations. US Senate debates AI Regulation Act. Apple faces antitrust probe in 12 countries.",
    "climate":    "Global CO2 hits record 425ppm. Renewable energy now cheaper than coal in 80% of markets. Amazon deforestation up 15% YoY.",
    "markets":    "S&P 500 hits all-time high on strong earnings. Nvidia surpasses $3T market cap. Hedge funds rotate into defensive sectors amid macro uncertainty.",
    "default":    "Tech giants report record Q1 earnings. Global AI investment surpasses $200B. Geopolitical tensions rattle emerging markets.",
}


def _get_mock_headline(query: str) -> str:
    """Returns a relevant mock headline based on keywords in the query."""
    q = query.lower()
    if any(k in q for k in ["ai", "openai", "gpt", "llm", "artificial"]):
        return MOCK_HEADLINES["ai"]
    if any(k in q for k in ["crypto", "bitcoin", "btc", "ethereum", "blockchain"]):
        return MOCK_HEADLINES["crypto"]
    if any(k in q for k in ["fed", "interest rate", "inflation", "recession", "cpi"]):
        return MOCK_HEADLINES["fed"]
    if any(k in q for k in ["space", "spacex", "mars", "nasa", "rocket", "starship"]):
        return MOCK_HEADLINES["space"]
    if any(k in q for k in ["regulation", "monopoly", "big tech", "surveillance", "privacy"]):
        return MOCK_HEADLINES["tech"]
    if any(k in q for k in ["climate", "environment", "carbon", "nature", "deforestation"]):
        return MOCK_HEADLINES["climate"]
    if any(k in q for k in ["market", "stock", "s&p", "nasdaq", "earnings", "hedge"]):
        return MOCK_HEADLINES["markets"]
    return MOCK_HEADLINES["default"]


def search_news(query: str, max_results: int = 3) -> str:
    """
    Searches for real news headlines using NewsAPI.
    Falls back gracefully to mock headlines if:
      - NEWS_API_KEY is not set
      - API quota is exceeded
      - Network error occurs

    Args:
        query       : search query string
        max_results : max number of articles to return (default 3)

    Returns:
        A formatted string of headlines and descriptions.
    """
    if not NEWS_API_KEY:
        print(f"  [search] No NEWS_API_KEY found — using mock headlines for: '{query}'")
        return _get_mock_headline(query)

    try:
        # NewsAPI free tier only allows articles from last 30 days
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        params = {
            "q":        query,
            "sortBy":   "publishedAt",
            "language": "en",
            "pageSize": max_results,
            "from":     from_date,
            "apiKey":   NEWS_API_KEY,
        }

        response = requests.get(NEWS_API_URL, params=params, timeout=10)

        # Handle quota exceeded
        if response.status_code == 426 or response.status_code == 429:
            print(f"  [search] NewsAPI quota exceeded — falling back to mock for: '{query}'")
            return _get_mock_headline(query)

        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            print(f"  [search] No results from NewsAPI — falling back to mock for: '{query}'")
            return _get_mock_headline(query)

        # Format results
        lines = []
        for a in articles[:max_results]:
            title       = a.get("title", "").strip()
            description = a.get("description", "").strip()
            source      = a.get("source", {}).get("name", "Unknown")
            if title:
                lines.append(f"[{source}] {title}. {description}")

        result = " | ".join(lines)
        print(f"  [search] NewsAPI returned {len(articles)} real article(s) for: '{query}'")
        return result

    except Exception as e:
        print(f"  [search] NewsAPI error ({e}) — falling back to mock for: '{query}'")
        return _get_mock_headline(query)
