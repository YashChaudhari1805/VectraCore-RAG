"""
core/bot_memory.py
------------------
Persistent per-bot memory backed by FAISS and disk serialisation.

Each bot maintains a personal ``IndexFlatIP`` FAISS index alongside a list
of post records.  Before generating content the bot retrieves semantically
relevant past opinions (RAG), ensuring cross-session consistency.

Memory files are stored at ``<memory_dir>/<bot_id>.pkl`` and survive
container restarts when mounted as a Docker named volume.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests

from core.config import settings
from core.logging_config import get_logger

log = get_logger(__name__)

_EMBEDDING_DIM = 384
_HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    f"sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)


def _hf_headers() -> dict[str, str]:
    """Build HuggingFace API request headers, injecting the bearer token when available."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"
    return headers


def _embed(text: str) -> np.ndarray:
    """
    Fetch an L2-normalised embedding vector from the HuggingFace Inference API.

    Args:
        text: The text to embed.

    Returns:
        A 1-D float32 numpy array of length ``_EMBEDDING_DIM``, unit-normalised.

    Raises:
        RuntimeError: On a 401 Unauthorized response (bad / missing HF token).
        requests.HTTPError: On any other non-2xx response.
    """
    response = requests.post(
        _HF_API_URL,
        headers=_hf_headers(),
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30,
    )

    if response.status_code == 401:
        token_hint = (settings.hf_token[:8] + "…") if settings.hf_token else "not set"
        raise RuntimeError(
            f"HuggingFace API returned 401 Unauthorized. "
            f"HF_TOKEN={token_hint}. "
            "Ensure your .env contains a valid HF_TOKEN."
        )

    response.raise_for_status()

    arr = np.array(response.json(), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)

    norm = np.linalg.norm(arr)
    return arr / (norm + 1e-10)


class PostRecord(dict):
    """
    Typed alias for a stored post dict.

    Keys:
        text      – generated post content.
        topic     – short label (1–4 words).
        timestamp – ISO-8601 datetime string.
    """


class BotMemory:
    """
    Per-bot persistent memory combining a FAISS vector index with a post list.

    Attributes:
        bot_id: Identifier matching a key in ``PERSONAS``.
        dim:    Embedding dimensionality (default 384 for MiniLM-L6-v2).
        index:  In-memory FAISS ``IndexFlatIP`` for cosine similarity search.
        posts:  Ordered list of ``PostRecord`` dicts, newest last.
    """

    def __init__(self, bot_id: str, dim: int = _EMBEDDING_DIM) -> None:
        self.bot_id = bot_id
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.posts: list[PostRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _storage_path(self) -> Path:
        """Return the pickle path for this bot's memory file."""
        memory_dir = Path(settings.memory_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)
        return memory_dir / f"{self.bot_id}.pkl"

    def _load(self) -> None:
        """
        Restore posts and FAISS index from disk if a memory file exists.

        Silently initialises empty memory when no file is found.
        """
        path = self._storage_path()
        if not path.exists():
            log.info("fresh_memory", bot_id=self.bot_id)
            return

        with open(path, "rb") as fh:
            saved: dict = pickle.load(fh)

        self.posts = saved.get("posts", [])
        vectors: list[np.ndarray] = saved.get("vectors", [])

        if vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            self.index.add(matrix)

        log.info("memory_loaded", bot_id=self.bot_id, total_posts=len(self.posts))

    def save(self) -> None:
        """
        Persist the current post list and embedding vectors to disk.

        Vectors are extracted from the FAISS index via ``reconstruct`` and
        stored alongside posts so the index can be rebuilt on next load.
        """
        vectors: list[np.ndarray] = [
            self.index.reconstruct(i) for i in range(self.index.ntotal)
        ]
        with open(self._storage_path(), "wb") as fh:
            pickle.dump({"posts": self.posts, "vectors": vectors}, fh)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_post(self, text: str, topic: str) -> None:
        """
        Embed ``text``, add the vector to the FAISS index, and persist.

        Args:
            text:  The generated post content to store.
            topic: A short (1–4 word) topic label.
        """
        vec = _embed(text).reshape(1, -1)
        self.index.add(vec)
        self.posts.append(
            PostRecord(
                text=text,
                topic=topic,
                timestamp=datetime.now().isoformat(),
            )
        )
        self.save()
        log.info("post_stored", bot_id=self.bot_id, total_posts=len(self.posts))

    def recall(self, context: str, top_k: int = 3) -> list[PostRecord]:
        """
        Retrieve the ``top_k`` most semantically relevant past posts.

        Args:
            context: A query string (topic phrase or incoming post text).
            top_k:   Maximum number of results to return.

        Returns:
            List of ``PostRecord`` dicts enriched with a ``similarity`` float,
            ordered by descending relevance.  Empty list when the index is empty.
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        vec = _embed(context).reshape(1, -1)
        similarities, indices = self.index.search(vec, k)

        results: list[PostRecord] = []
        for sim, idx in zip(similarities[0], indices[0]):
            record = PostRecord(self.posts[idx])
            record["similarity"] = round(float(sim), 4)
            results.append(record)

        return results

    def summary(self) -> str:
        """Return a human-readable summary of stored memories."""
        if not self.posts:
            return "No memories yet."
        recent_topics = [p["topic"] for p in self.posts[-5:]]
        return f"{len(self.posts)} total memories. Recent topics: {', '.join(recent_topics)}"


# ------------------------------------------------------------------
# Module-level registry — one BotMemory instance per bot_id
# ------------------------------------------------------------------

_registry: dict[str, BotMemory] = {}


def get_memory(bot_id: str) -> BotMemory:
    """
    Return (or lazily create) the ``BotMemory`` instance for ``bot_id``.

    Args:
        bot_id: A key from ``PERSONAS``.

    Returns:
        The cached ``BotMemory`` instance for this bot.
    """
    if bot_id not in _registry:
        _registry[bot_id] = BotMemory(bot_id)
    return _registry[bot_id]
