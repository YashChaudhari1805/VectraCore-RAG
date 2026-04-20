"""
core/bot_memory.py
------------------
Persistent per-bot memory using FAISS + disk serialisation.

Every post a bot generates is embedded and added to its personal FAISS index.
Before posting, the bot retrieves its most relevant past opinions so it stays
consistent and never contradicts itself — this is true RAG.

Memory is saved to data/memory/<bot_id>.pkl between runs.
"""

import os
import pickle
import numpy as np
import faiss
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MEMORY_DIR = Path("data/memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)
def _get_headers() -> dict:
    token = os.environ.get("HF_TOKEN", "")
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_embedding(text: str) -> np.ndarray:
    response = requests.post(
        HF_API_URL,
        headers=_get_headers(),
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30,
    )
    response.raise_for_status()
    arr = np.array(response.json(), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr / (np.linalg.norm(arr) + 1e-10)


class BotMemory:
    """
    Manages a single bot's persistent memory.

    Stores posts as (embedding, text, timestamp) tuples.
    FAISS index enables semantic retrieval of past opinions.
    """

    def __init__(self, bot_id: str, dim: int = 384):
        self.bot_id   = bot_id
        self.dim      = dim
        self.index    = faiss.IndexFlatIP(dim)
        self.posts: list[dict] = []   # [{"text": str, "topic": str, "timestamp": str}]
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _path(self) -> Path:
        return MEMORY_DIR / f"{self.bot_id}.pkl"

    def _load(self) -> None:
        """Load existing memory from disk if available."""
        path = self._path()
        if path.exists():
            with open(path, "rb") as f:
                saved = pickle.load(f)
            self.posts = saved["posts"]
            vectors    = saved["vectors"]
            if len(vectors) > 0:
                matrix = np.vstack(vectors).astype(np.float32)
                self.index.add(matrix)
            print(f"  [memory] Loaded {len(self.posts)} memories for {self.bot_id}")
        else:
            print(f"  [memory] Fresh memory for {self.bot_id}")

    def save(self) -> None:
        """Persist memory to disk."""
        # Reconstruct vectors from index (FAISS doesn't expose stored vectors directly)
        # We store them separately in the pickle
        vectors = []
        if self.index.ntotal > 0:
            # Reconstruct via index.reconstruct
            for i in range(self.index.ntotal):
                vectors.append(self.index.reconstruct(i))

        with open(self._path(), "wb") as f:
            pickle.dump({"posts": self.posts, "vectors": vectors}, f)

    # ── Core Operations ───────────────────────────────────────────────────────

    def add_post(self, text: str, topic: str) -> None:
        """
        Embeds a new post and adds it to the bot's memory index.
        Also persists to disk immediately.
        """
        vec = _get_embedding(text).reshape(1, -1)
        self.index.add(vec)
        self.posts.append({
            "text":      text,
            "topic":     topic,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()
        print(f"  [memory] Stored new post for {self.bot_id} (total: {len(self.posts)})")

    def recall(self, context: str, top_k: int = 3) -> list[dict]:
        """
        Retrieves the top_k most semantically relevant past posts
        given a context string (e.g. the current topic or incoming post).

        Returns:
            List of {"text": str, "topic": str, "timestamp": str, "similarity": float}
        """
        if self.index.ntotal == 0:
            return []

        k   = min(top_k, self.index.ntotal)
        vec = _get_embedding(context).reshape(1, -1)
        similarities, indices = self.index.search(vec, k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            post = self.posts[idx].copy()
            post["similarity"] = round(float(sim), 4)
            results.append(post)

        return results

    def summary(self) -> str:
        """Returns a human-readable summary of what this bot remembers."""
        if not self.posts:
            return "No memories yet."
        topics = [p["topic"] for p in self.posts[-5:]]
        return f"{len(self.posts)} total memories. Recent topics: {', '.join(topics)}"


# ── Module-level registry — one BotMemory per bot ─────────────────────────────

_memories: dict[str, BotMemory] = {}


def get_memory(bot_id: str) -> BotMemory:
    """Returns (or creates) the BotMemory instance for a given bot."""
    if bot_id not in _memories:
        _memories[bot_id] = BotMemory(bot_id)
    return _memories[bot_id]