"""
tests/test_unit_bot_memory.py — unit tests for core.bot_memory
"""

from __future__ import annotations
import pickle
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from core.bot_memory import BotMemory, get_memory


def _vec(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def tmp_mem(tmp_path):
    with patch("core.bot_memory.settings") as s:
        s.hf_token = ""
        s.memory_dir = str(tmp_path)
        yield tmp_path


class TestBotMemoryInit:
    def test_fresh_starts_empty(self, tmp_mem):
        with patch("core.bot_memory._embed", return_value=_vec()):
            m = BotMemory("test_bot")
        assert m.posts == [] and m.index.ntotal == 0

    def test_loads_existing_pickle(self, tmp_mem):
        vec = _vec()
        data = {"posts": [{"text": "hi", "topic": "AI", "timestamp": "2024-01-01T00:00:00"}], "vectors": [vec]}
        with open(tmp_mem / "test_bot.pkl", "wb") as f:
            pickle.dump(data, f)
        with patch("core.bot_memory._embed", return_value=vec):
            m = BotMemory("test_bot")
        assert len(m.posts) == 1 and m.index.ntotal == 1


class TestAddPost:
    def test_increments_count(self, tmp_mem):
        with patch("core.bot_memory._embed", return_value=_vec()):
            m = BotMemory("bot")
            m.add_post("Test", "AI")
        assert len(m.posts) == 1 and m.index.ntotal == 1

    def test_stores_correct_fields(self, tmp_mem):
        with patch("core.bot_memory._embed", return_value=_vec()):
            m = BotMemory("bot")
            m.add_post("Content", "markets")
        assert m.posts[0]["text"] == "Content"
        assert m.posts[0]["topic"] == "markets"
        assert "timestamp" in m.posts[0]

    def test_persists_to_disk(self, tmp_mem):
        with patch("core.bot_memory._embed", return_value=_vec()):
            m = BotMemory("bot_p")
            m.add_post("Disk test", "tech")
        assert (tmp_mem / "bot_p.pkl").exists()


class TestRecall:
    def test_empty_returns_empty(self, tmp_mem):
        with patch("core.bot_memory._embed", return_value=_vec()):
            m = BotMemory("bot_r")
            assert m.recall("anything") == []

    def test_returns_results_after_add(self, tmp_mem):
        vec = _vec()
        with patch("core.bot_memory._embed", return_value=vec):
            m = BotMemory("bot_r2")
            m.add_post("First", "AI")
            m.add_post("Second", "crypto")
            results = m.recall("AI topic", top_k=2)
        assert all("text" in r and "similarity" in r for r in results)

    def test_top_k_capped(self, tmp_mem):
        with patch("core.bot_memory._embed", return_value=_vec()):
            m = BotMemory("bot_cap")
            m.add_post("Only", "fed")
            assert len(m.recall("x", top_k=10)) <= 1


class TestSummary:
    def test_empty_memory(self, tmp_mem):
        with patch("core.bot_memory._embed", return_value=_vec()):
            m = BotMemory("bot_s")
        assert "No memories" in m.summary()

    def test_non_empty(self, tmp_mem):
        with patch("core.bot_memory._embed", return_value=_vec()):
            m = BotMemory("bot_s2")
            m.add_post("Post about AI", "AI")
        assert "1" in m.summary() and "AI" in m.summary()


class TestRegistry:
    def test_same_instance(self, tmp_mem):
        import core.bot_memory as mod
        mod._registry.clear()
        with patch("core.bot_memory._embed", return_value=_vec()):
            assert get_memory("Bot_A_TechMaximalist") is get_memory("Bot_A_TechMaximalist")

    def test_different_bots_different_instances(self, tmp_mem):
        import core.bot_memory as mod
        mod._registry.clear()
        with patch("core.bot_memory._embed", return_value=_vec()):
            assert get_memory("Bot_A_TechMaximalist") is not get_memory("Bot_B_Doomer")
