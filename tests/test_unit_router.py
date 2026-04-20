"""
tests/test_unit_router.py — unit tests for core.router
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
    def test_idempotent(self):
        import core.router as r
        r._index = None
        r._bot_ids = None
        with patch("core.router._embed", return_value=_vec()) as m:
            build_index()
            count = m.call_count
            build_index()           # second call — should be no-op
            assert m.call_count == count

    def test_index_contains_all_personas(self):
        import core.router as r
        from core.personas import PERSONAS
        r._index = None
        r._bot_ids = None
        with patch("core.router._embed", return_value=_vec()):
            build_index()
        assert r._index.ntotal == len(PERSONAS)


class TestRoutePost:
    def test_returns_list(self):
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist"]),
        ):
            idx.search.return_value = (np.array([[0.9]]), np.array([[0]]))
            assert isinstance(route_post("some text", threshold=0.1), list)

    def test_filters_by_threshold(self):
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist"]),
        ):
            idx.search.return_value = (np.array([[0.05]]), np.array([[0]]))
            assert route_post("text", threshold=0.5) == []

    def test_sorted_descending(self):
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist", "Bot_B_Doomer"]),
        ):
            idx.search.return_value = (np.array([[0.6, 0.8]]), np.array([[0, 1]]))
            result = route_post("text", threshold=0.1)
        sims = [r["similarity"] for r in result]
        assert sims == sorted(sims, reverse=True)

    def test_result_schema(self):
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist"]),
        ):
            idx.search.return_value = (np.array([[0.9]]), np.array([[0]]))
            result = route_post("AI post", threshold=0.1)
        assert "bot_id" in result[0]
        assert "similarity" in result[0]


class TestGetAllScores:
    def test_returns_all_bots(self):
        with (
            patch("core.router._index") as idx,
            patch("core.router._bot_ids", ["Bot_A_TechMaximalist", "Bot_B_Doomer"]),
        ):
            idx.search.return_value = (np.array([[0.7, 0.3]]), np.array([[0, 1]]))
            assert len(get_all_scores("text")) == 2
