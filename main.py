"""
api/main.py
-----------
FastAPI backend for Grid07.
Always start via `python run.py` from the project root — never run this file directly.

Endpoints:
  GET  /                    - Health check
  GET  /api/bots            - List all bot personas
  POST /api/route           - Route a post to matching bots
  POST /api/generate        - Trigger a bot to autonomously generate a post
  POST /api/reply           - Bot replies to a thread (with injection defence)
  GET  /api/feed            - All generated posts from bot memory
  GET  /api/memory/{bot_id} - Memory summary for a specific bot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from core.router import build_index, route_post
from core.content_engine import generate_post
from core.combat_engine import generate_defense_reply
from core.bot_memory import get_memory
from core.personas import PERSONAS

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Grid07 AI Engine",
    description="Cognitive routing, autonomous content generation, and RAG combat engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dashboard — resolve path relative to project root
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("[startup] Building persona index...")
    build_index()
    print("[startup] Ready — visit http://localhost:8000/dashboard")


# ── Request / Response models ─────────────────────────────────────────────────

class RouteRequest(BaseModel):
    post_content: str
    threshold:    float = 0.18

class GenerateRequest(BaseModel):
    bot_id: str

class ReplyRequest(BaseModel):
    bot_id:          str
    parent_post:     str
    comment_history: list[dict] = []
    human_reply:     str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "Grid07 AI Engine", "version": "2.0.0"}


@app.get("/api/bots")
def list_bots():
    return {
        "bots": [
            {
                "id":           pid,
                "display_name": p["display_name"],
                "description":  p["description"],
            }
            for pid, p in PERSONAS.items()
        ]
    }


@app.post("/api/route")
def route(req: RouteRequest):
    if not req.post_content.strip():
        raise HTTPException(status_code=400, detail="post_content cannot be empty")

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
        raise HTTPException(
            status_code=404,
            detail=f"Bot '{req.bot_id}' not found. Available: {list(PERSONAS.keys())}"
        )
    return generate_post(req.bot_id)


@app.post("/api/reply")
def reply(req: ReplyRequest):
    if req.bot_id not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Bot '{req.bot_id}' not found")
    if not req.human_reply.strip():
        raise HTTPException(status_code=400, detail="human_reply cannot be empty")

    return generate_defense_reply(
        bot_id          = req.bot_id,
        parent_post     = req.parent_post,
        comment_history = req.comment_history,
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
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")

    memory = get_memory(bot_id)
    return {
        "bot_id":       bot_id,
        "total_posts":  len(memory.posts),
        "summary":      memory.summary(),
        "recent_posts": memory.posts[-5:],
    }


@app.get("/dashboard")
def dashboard():
    index_path = DASHBOARD_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(str(index_path))
