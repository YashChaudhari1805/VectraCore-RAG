"""
api/main.py
-----------
FastAPI backend for VectraCore RAG.
Start via ``python run.py`` from the project root.
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from core.bot_memory import get_memory
from core.combat_engine import generate_defense_reply
from core.config import settings
from core.content_engine import generate_post
from core.personas import PERSONAS
from core.router import build_index, route_post

_DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    print("[startup] Building persona index...")
    build_index()
    for bot_id in PERSONAS:
        get_memory(bot_id)
    print(f"[startup] Ready — http://localhost:8000/dashboard")
    yield

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


# ── Startup ───────────────────────────────────────────────────────────────────



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


@app.get("/api/bots")
def list_bots():
    return {
        "bots": [
            {"id": pid, "display_name": p["display_name"], "description": p["description"]}
            for pid, p in PERSONAS.items()
        ]
    }


@app.post("/api/route")
def route(req: RouteRequest):
    matches = route_post(req.post_content, threshold=req.threshold)
    return {
        "post_content":  req.post_content,
        "threshold":     req.threshold,
        "matched_bots":  matches,
        "total_matched": len(matches),
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.bot_id not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Bot '{req.bot_id}' not found.")
    return generate_post(req.bot_id)


@app.post("/api/reply")
def reply(req: ReplyRequest):
    if req.bot_id not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Bot '{req.bot_id}' not found.")
    return generate_defense_reply(
        bot_id          = req.bot_id,
        parent_post     = req.parent_post,
        comment_history = [c.model_dump() for c in req.comment_history],
        human_reply     = req.human_reply,
    )


@app.get("/api/feed")
def get_feed():
    all_posts = []
    for bot_id in PERSONAS:
        memory = get_memory(bot_id)
        for post in memory.posts:
            all_posts.append({
                "bot_id":       bot_id,
                "display_name": PERSONAS[bot_id]["display_name"],
                **post,
            })
    all_posts.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"total": len(all_posts), "posts": all_posts}


@app.get("/api/memory/{bot_id}")
def get_bot_memory(bot_id: str):
    if bot_id not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found.")
    memory = get_memory(bot_id)
    return {
        "bot_id":       bot_id,
        "total_posts":  len(memory.posts),
        "summary":      memory.summary(),
        "recent_posts": memory.posts[-5:],
    }


@app.get("/dashboard")
def dashboard():
    index_path = _DASHBOARD_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return FileResponse(str(index_path))
