"""
run.py
------
Project root entry point for the VectraCore RAG API server.

Usage:
    python run.py
    uvicorn run:app --reload --port 8000

Placing the entry point at the project root guarantees that ``core.*``
and ``api.*`` imports resolve correctly regardless of invocation method.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.main import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    import uvicorn
    from core.config import settings

    uvicorn.run(
        "run:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=not settings.is_production,
    )
