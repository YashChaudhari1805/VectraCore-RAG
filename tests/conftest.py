"""
tests/conftest.py — shared pytest fixtures
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


def _dummy_vector(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture(scope="session")
def test_client() -> Generator[TestClient, None, None]:
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def mock_embed():
    vec = _dummy_vector()
    with (
        patch("core.router._embed", return_value=vec) as router_mock,
        patch("core.bot_memory._embed", return_value=vec),
    ):
        yield router_mock


@pytest.fixture
def mock_llm():
    fake = MagicMock()
    fake.content = '{"bot_id": "Bot_A_TechMaximalist", "topic": "AI", "post_content": "AI will save us all."}'
    with patch("core.content_engine._llm") as llm_mock:
        llm_mock.invoke.return_value = fake
        yield llm_mock
