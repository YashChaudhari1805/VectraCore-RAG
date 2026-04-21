"""
api/main.py — VectraCore RAG FastAPI backend
Start via: python run.py
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
from collections import defaultdict, deque
from typing import Optional

load_dotenv()

from core.bot_memory import get_memory
from core.combat_engine import generate_defense_reply
from core.config import settings
from core.content_engine import generate_post
from core.personas import PERSONAS
from core.router import build_index, route_post

_DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

# ── Rate limiter ──────────────────────────────────────────────────────────────

_rate_windows: dict[str, deque] = defaultdict(deque)

def _client_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    window = _rate_windows[ip]
    now = time.monotonic()
    cutoff = now - 60.0
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {settings.rate_limit_per_minute} requests/minute.",
        )
    window.append(now)

# ── API Key auth ──────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(x_api_key: Optional[str] = Depends(api_key_header)) -> None:
    if not settings.auth_enabled:
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    if x_api_key not in settings.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key.")

# Special endpoint: verify key without full auth (used by UI login screen)
def verify_api_key_loose(x_api_key: Optional[str] = Depends(api_key_header)) -> bool:
    if not settings.auth_enabled:
        return True
    return x_api_key in settings.api_keys

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Building persona index...")
    build_index()
    for bot_id in PERSONAS:
        get_memory(bot_id)
    print(f"[startup] Ready — http://localhost:{settings.port}/dashboard")
    yield

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VectraCore RAG AI Engine",
    description="Cognitive routing, autonomous content generation, and RAG combat engine.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

if _DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_DASHBOARD_DIR)), name="static")

# Shared dependencies for protected routes
_protected = [Depends(verify_api_key), Depends(check_rate_limit)]

# ── Request models ────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    post_content: str = Field(..., min_length=1)
    threshold: float = Field(default=0.18, ge=0.0, le=1.0)

class GenerateRequest(BaseModel):
    bot_id: str

class CommentRecord(BaseModel):
    author: str
    text: str

class ReplyRequest(BaseModel):
    bot_id: str
    parent_post: str
    comment_history: list[CommentRecord] = []
    human_reply: str = Field(..., min_length=1)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "VectraCore RAG AI Engine", "version": "2.0.0"}


@app.post("/api/auth/verify")
def verify_key(request: Request, x_api_key: Optional[str] = Depends(api_key_header)):
    """
    UI uses this to validate an API key before unlocking the dashboard.
    Returns 200 + auth_required=True/False so UI knows whether auth is on.
    """
    check_rate_limit(request)
    auth_required = settings.auth_enabled
    if not auth_required:
        return {"valid": True, "auth_required": False}
    if not x_api_key:
        return {"valid": False, "auth_required": True}
    valid = x_api_key in settings.api_keys
    if not valid:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return {"valid": True, "auth_required": True}


@app.get("/api/bots", dependencies=_protected)
def list_bots():
    return {
        "bots": [
            {"id": pid, "display_name": p["display_name"], "description": p["description"]}
            for pid, p in PERSONAS.items()
        ]
    }


@app.post("/api/route", dependencies=_protected)
def route(req: RouteRequest):
    matches = route_post(req.post_content, threshold=req.threshold)
    return {
        "post_content":  req.post_content,
        "threshold":     req.threshold,
        "matched_bots":  matches,
        "total_matched": len(matches),
    }


@app.post("/api/generate", dependencies=_protected)
def generate(req: GenerateRequest):
    if req.bot_id not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Bot '{req.bot_id}' not found.")
    return generate_post(req.bot_id)


@app.post("/api/reply", dependencies=_protected)
def reply(req: ReplyRequest):
    if req.bot_id not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Bot '{req.bot_id}' not found.")
    return generate_defense_reply(
        bot_id=req.bot_id,
        parent_post=req.parent_post,
        comment_history=[c.model_dump() for c in req.comment_history],
        human_reply=req.human_reply,
    )


@app.get("/api/feed", dependencies=_protected)
def get_feed():
    all_posts = []
    for bot_id in PERSONAS:
        memory = get_memory(bot_id)
        for post in memory.posts:
            all_posts.append({
                "bot_id": bot_id,
                "display_name": PERSONAS[bot_id]["display_name"],
                **post,
            })
    all_posts.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"total": len(all_posts), "posts": all_posts}


@app.get("/api/memory/{bot_id}", dependencies=_protected)
def get_bot_memory(bot_id: str):
    if bot_id not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found.")
    memory = get_memory(bot_id)
    return {
        "bot_id": bot_id,
        "total_posts": len(memory.posts),
        "summary": memory.summary(),
        "recent_posts": memory.posts[-5:],
    }


@app.get("/dashboard")
def dashboard():
    index_path = _DASHBOARD_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return FileResponse(str(index_path))
