"""
core/router.py
--------------
Vector-based persona router. Builds a FAISS index from bot persona embeddings
and routes incoming posts to semantically matching bots via cosine similarity.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 via HuggingFace Inference API
"""

import os
import requests
import numpy as np
import faiss
from dotenv import load_dotenv
from core.personas import PERSONAS

# Load .env here so HF_TOKEN is available when _get_embedding is first called
load_dotenv()

HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Singleton index — built once, reused across calls
_index:   faiss.IndexFlatIP = None
_bot_ids: list[str]         = None


def _get_headers() -> dict:
    """
    Build headers fresh on every call so HF_TOKEN is always read
    after load_dotenv() has had a chance to populate os.environ.
    """
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_embedding(text: str) -> np.ndarray:
    """Fetches L2-normalised embedding from HuggingFace Inference API."""
    response = requests.post(
        HF_API_URL,
        headers=_get_headers(),
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30,
    )

    if response.status_code == 401:
        token = os.environ.get("HF_TOKEN", "")
        raise RuntimeError(
            f"HF API 401 Unauthorized.\n"
            f"  HF_TOKEN loaded: {'YES — ' + token[:8] + '...' if token else 'NO — not found in environment'}\n"
            f"  Make sure your .env file is in the project root (same folder as main.py)\n"
            f"  and contains: HF_TOKEN=hf_xxxxxxxxxxxx"
        )
    response.raise_for_status()

    arr = np.array(response.json(), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    arr = arr / (np.linalg.norm(arr) + 1e-10)
    return arr


def build_index() -> None:
    """
    Builds the FAISS persona index from all personas in PERSONAS.
    Called once at startup. Subsequent calls are no-ops.
    """
    global _index, _bot_ids

    if _index is not None:
        return  # already built

    print("[router] Building persona index...")
    _bot_ids   = list(PERSONAS.keys())
    embeddings = []

    for bot_id in _bot_ids:
        vec = _get_embedding(PERSONAS[bot_id]["description"])
        embeddings.append(vec)
        print(f"  [router] Embedded {bot_id}")

    matrix    = np.vstack(embeddings)
    dimension = matrix.shape[1]
    _index    = faiss.IndexFlatIP(dimension)
    _index.add(matrix)
    print(f"[router] Index ready — {_index.ntotal} personas, dim={dimension}\n")


def route_post(post_content: str, threshold: float = 0.18) -> list[dict]:
    """
    Routes a post to all bots whose persona cosine similarity exceeds threshold.
    """
    if _index is None:
        build_index()

    vec = _get_embedding(post_content).reshape(1, -1)
    similarities, indices = _index.search(vec, len(_bot_ids))

    matched = [
        {"bot_id": _bot_ids[idx], "similarity": round(float(sim), 4)}
        for sim, idx in zip(similarities[0], indices[0])
        if sim >= threshold
    ]
    matched.sort(key=lambda x: x["similarity"], reverse=True)
    return matched


def get_all_scores(post_content: str) -> list[dict]:
    """Returns similarity scores for ALL bots regardless of threshold. Used for eval."""
    if _index is None:
        build_index()

    vec = _get_embedding(post_content).reshape(1, -1)
    similarities, indices = _index.search(vec, len(_bot_ids))

    return [
        {"bot_id": _bot_ids[idx], "similarity": round(float(sim), 4)}
        for sim, idx in zip(similarities[0], indices[0])
    ]