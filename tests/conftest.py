"""
tests/conftest.py
-----------------
Shared pytest fixtures for the Grid07 test suite.

Provides:
- ``test_client`` — a ``TestClient`` with auth and rate-limit dependencies
  overridden so tests don't need real API keys or worry about limits.
- ``mock_embed`` — patches ``_embed`` in router and bot_memory to return
  deterministic fixed vectors, eliminating real HuggingFace API calls.
- ``mock_llm`` — patches ChatGroq so no real Groq API calls are made.
"""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.security import check_rate_limit, verify_api_key


def _dummy_vector(seed: int = 42) -> np.ndarray:
    """Return a reproducible unit-normalised 384-dim float32 vector."""
    rng = np.random.default_rng(seed)
    vec = rng.random(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture(scope="session")
def test_client() -> Generator[TestClient, None, None]:
    """
    Session-scoped ``TestClient`` with auth and rate-limit deps overridden.

    Auth and rate-limit dependency overrides are applied once for the whole
    test session so individual tests don't have to supply API key headers.
    """
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[check_rate_limit] = lambda: None
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_embed():
    """
    Patch the HuggingFace embedding call in both ``core.router`` and
    ``core.bot_memory`` so tests never hit a real external API.

    Yields the mock object so tests can configure ``return_value`` further.
    """
    vec = _dummy_vector()
    with (
        patch("core.router._embed", return_value=vec) as router_mock,
        patch("core.bot_memory._embed", return_value=vec),
    ):
        yield router_mock


@pytest.fixture
def mock_llm():
    """
    Patch ``ChatGroq`` so no real LLM calls are made during unit tests.

    The mock returns a fixed post JSON string by default; individual tests
    can override ``mock_llm.invoke.return_value`` as needed.
    """
    fake_response = MagicMock()
    fake_response.content = (
        '{"bot_id": "Bot_A_TechMaximalist", "topic": "AI", '
        '"post_content": "AI will save us all. No question."}'
    )
    with patch("core.content_engine._llm") as llm_mock:
        llm_mock.invoke.return_value = fake_response
        yield llm_mock
