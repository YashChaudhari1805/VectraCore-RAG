"""
tests/test_unit_router.py
-------------------------
Unit tests for ``core.router``.

All HuggingFace API calls are patched so tests run without credentials.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from core.router import build_index, get_all_scores, route_post


def _vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


class TestBuildIndex:
    """Tests for ``build_index``."""

    def test_idempotent(self, mock_embed) -> None:
        """Calling ``build_index`` twice should not embed personas twice."""
        import core.router as router_mod

        router_mod._index = None
        router_mod._bot_ids = None

        with patch("core.router._embed", return_value=_vec()) as embed_mock:
            build_index()
            first_call_count = embed_mock.call_count
            build_index()
            assert embed_mock.call_count == first_call_count

    def test_index_contains_all_personas(self, mock_embed) -> None:
        """Index should contain one vector per persona."""
        import core.router as router_mod
        from core.personas import PERSONAS

        router_mod._index = None
        router_mod._bot_ids = None

        with patch("core.router._embed", return_value=_vec()):
            build_index()

        assert router_mod._index is not None
        assert router_mod._index.ntotal == len(PERSONAS)


class TestRoutePost:
    """Tests for ``route_post``."""

    def test_returns_list(self, mock_embed) -> None:
        """``route_post`` should always return a list."""
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist"]),
        ):
            idx.search.return_value = (np.array([[0.9]]), np.array([[0]]))
            result = route_post("Some post content", threshold=0.1)
        assert isinstance(result, list)

    def test_filters_by_threshold(self, mock_embed) -> None:
        """Results below the threshold must be excluded."""
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist"]),
        ):
            idx.search.return_value = (np.array([[0.05]]), np.array([[0]]))
            result = route_post("Some post content", threshold=0.5)
        assert result == []

    def test_sorted_descending(self, mock_embed) -> None:
        """Matched bots must be returned in descending similarity order."""
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist", "Bot_B_Doomer"]),
        ):
            idx.search.return_value = (
                np.array([[0.6, 0.8]]),
                np.array([[0, 1]]),
            )
            result = route_post("Some post content", threshold=0.1)

        similarities = [r["similarity"] for r in result]
        assert similarities == sorted(similarities, reverse=True)

    def test_result_schema(self, mock_embed) -> None:
        """Each result dict must contain 'bot_id' and 'similarity' keys."""
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist"]),
        ):
            idx.search.return_value = (np.array([[0.9]]), np.array([[0]]))
            result = route_post("AI post", threshold=0.1)

        assert len(result) == 1
        assert "bot_id" in result[0]
        assert "similarity" in result[0]


class TestGetAllScores:
    """Tests for ``get_all_scores``."""

    def test_returns_all_bots(self, mock_embed) -> None:
        """``get_all_scores`` should return a score for every bot."""
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist", "Bot_B_Doomer"]),
        ):
            idx.search.return_value = (
                np.array([[0.7, 0.3]]),
                np.array([[0, 1]]),
            )
            result = get_all_scores("Some text")

        assert len(result) == 2
