import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure relative imports work regardless of where the script is executed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import get_settings
from app.api.router import api_router
from app.services.rag_engine import rag_engine

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Sequence: Build the Vector DB before accepting requests
    print("[startup] Initializing VectraCore RAG Engine...")
    rag_engine.build_persona_index()
    print(f"[startup] Application ready. Access the UI at http://localhost:8000/dashboard")
    yield
    # Graceful shutdown logic goes here if needed

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Apply CORS Middleware globally
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the Domain-Driven API Routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# ---------------------------------------------------------
# Health check & Root Redirect
# ---------------------------------------------------------
@app.get("/")
def root():
    # Automatically redirect users to the dashboard.
    # Note: Docker Healthchecks via urllib follow redirects, so this still returns a 200 OK.
    return RedirectResponse(url="/dashboard")

# ---------------------------------------------------------
# Static File Routing (The Monolith Pattern)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
ASSETS_DIR = os.path.join(DASHBOARD_DIR, "assets")

# Serve the built JS/CSS bundle at the exact root-absolute paths
# the built index.html expects (e.g. /assets/index-XXXX.js)
if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.get("/dashboard")
@app.get("/dashboard/{full_path:path}")
async def serve_dashboard(full_path: str = ""):
    index_path = os.path.join(DASHBOARD_DIR, "index.html")

    if not os.path.exists(index_path):
        return JSONResponse(
            status_code=404,
            content={"detail": "Dashboard not found. Ensure 'npm run build' was executed in the frontend directory."}
        )

    # Always serve the React SPA shell; client-side routing (if any)
    # is handled inside the app itself.
    return FileResponse(index_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)