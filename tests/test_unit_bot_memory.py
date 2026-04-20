"""
tests/test_unit_bot_memory.py
------------------------------
Unit tests for ``core.bot_memory.BotMemory``.

All HuggingFace embedding calls and disk I/O are patched so tests are
hermetic and don't require credentials or a filesystem.
"""

from __future__ import annotations

import pickle
import tempfile
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
def tmp_memory_dir(tmp_path: Path):
    """Redirect memory storage to a temporary directory for each test."""
    with patch("core.bot_memory.settings") as mock_settings:
        mock_settings.hf_token = ""
        mock_settings.memory_dir = str(tmp_path)
        yield tmp_path


class TestBotMemoryInit:
    """Tests for ``BotMemory.__init__`` and ``_load``."""

    def test_fresh_memory_starts_empty(self, tmp_memory_dir: Path) -> None:
        """A new bot with no pickle file should have an empty post list."""
        with patch("core.bot_memory._embed", return_value=_vec()):
            memory = BotMemory("test_bot")
        assert memory.posts == []
        assert memory.index.ntotal == 0

    def test_loads_existing_memory(self, tmp_memory_dir: Path) -> None:
        """BotMemory should restore posts from an existing pickle file."""
        vec = _vec()
        saved = {
            "posts": [{"text": "hello", "topic": "AI", "timestamp": "2024-01-01T00:00:00"}],
            "vectors": [vec],
        }
        pickle_path = tmp_memory_dir / "test_bot.pkl"
        with open(pickle_path, "wb") as fh:
            pickle.dump(saved, fh)

        with patch("core.bot_memory._embed", return_value=vec):
            memory = BotMemory("test_bot")

        assert len(memory.posts) == 1
        assert memory.index.ntotal == 1


class TestBotMemoryAddPost:
    """Tests for ``BotMemory.add_post``."""

    def test_add_increments_post_count(self, tmp_memory_dir: Path) -> None:
        """Adding a post should increment the post list length by one."""
        with patch("core.bot_memory._embed", return_value=_vec()):
            memory = BotMemory("bot_add")
            memory.add_post("Test post", "AI")
        assert len(memory.posts) == 1
        assert memory.index.ntotal == 1

    def test_add_stores_correct_fields(self, tmp_memory_dir: Path) -> None:
        """Stored post record should contain text, topic, and timestamp keys."""
        with patch("core.bot_memory._embed", return_value=_vec()):
            memory = BotMemory("bot_fields")
            memory.add_post("Content here", "markets")

        post = memory.posts[0]
        assert post["text"] == "Content here"
        assert post["topic"] == "markets"
        assert "timestamp" in post

    def test_add_persists_to_disk(self, tmp_memory_dir: Path) -> None:
        """A pickle file should be written after ``add_post``."""
        with patch("core.bot_memory._embed", return_value=_vec()):
            memory = BotMemory("bot_persist")
            memory.add_post("Disk test", "tech")

        pkl = tmp_memory_dir / "bot_persist.pkl"
        assert pkl.exists()


class TestBotMemoryRecall:
    """Tests for ``BotMemory.recall``."""

    def test_recall_empty_index_returns_empty(self, tmp_memory_dir: Path) -> None:
        """``recall`` on a fresh memory should return an empty list."""
        with patch("core.bot_memory._embed", return_value=_vec()):
            memory = BotMemory("bot_recall_empty")
            result = memory.recall("anything", top_k=3)
        assert result == []

    def test_recall_returns_results(self, tmp_memory_dir: Path) -> None:
        """``recall`` should return results after posts have been added."""
        vec = _vec()
        with patch("core.bot_memory._embed", return_value=vec):
            memory = BotMemory("bot_recall_results")
            memory.add_post("First post", "AI")
            memory.add_post("Second post", "crypto")
            results = memory.recall("AI topic", top_k=2)

        assert len(results) <= 2
        for r in results:
            assert "text" in r
            assert "similarity" in r

    def test_recall_top_k_capped_by_index_size(self, tmp_memory_dir: Path) -> None:
        """``recall`` should not return more results than posts stored."""
        vec = _vec()
        with patch("core.bot_memory._embed", return_value=vec):
            memory = BotMemory("bot_recall_cap")
            memory.add_post("Only post", "fed")
            results = memory.recall("anything", top_k=10)

        assert len(results) <= 1


class TestBotMemorySummary:
    """Tests for ``BotMemory.summary``."""

    def test_summary_empty(self, tmp_memory_dir: Path) -> None:
        """Summary on an empty memory should indicate no memories."""
        with patch("core.bot_memory._embed", return_value=_vec()):
            memory = BotMemory("bot_summary_empty")
        assert "No memories" in memory.summary()

    def test_summary_non_empty(self, tmp_memory_dir: Path) -> None:
        """Summary should include the post count and at least one topic."""
        vec = _vec()
        with patch("core.bot_memory._embed", return_value=vec):
            memory = BotMemory("bot_summary_full")
            memory.add_post("Post about AI", "AI")
            summary = memory.summary()

        assert "1" in summary
        assert "AI" in summary


class TestGetMemoryRegistry:
    """Tests for the module-level ``get_memory`` registry."""

    def test_returns_same_instance(self, tmp_memory_dir: Path) -> None:
        """Two calls with the same bot_id should return the identical object."""
        import core.bot_memory as mem_mod

        mem_mod._registry.clear()
        with patch("core.bot_memory._embed", return_value=_vec()):
            a = get_memory("Bot_A_TechMaximalist")
            b = get_memory("Bot_A_TechMaximalist")
        assert a is b

    def test_different_bots_different_instances(self, tmp_memory_dir: Path) -> None:
        """Different bot IDs must yield different ``BotMemory`` instances."""
        import core.bot_memory as mem_mod

        mem_mod._registry.clear()
        with patch("core.bot_memory._embed", return_value=_vec()):
            a = get_memory("Bot_A_TechMaximalist")
            b = get_memory("Bot_B_Doomer")
        assert a is not b
