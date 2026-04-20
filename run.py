"""
run.py  —  project root entry point
------------------------------------
Start the Grid07 API server from the project root:

    python run.py
    # or
    uvicorn run:app --reload --port 8000

Keeping this in the root ensures all `core.*` imports resolve correctly.
"""

import sys
from pathlib import Path

# Guarantee project root is on sys.path regardless of how this is invoked
sys.path.insert(0, str(Path(__file__).parent))

from api.main import app          # noqa: F401  (uvicorn needs this name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)
