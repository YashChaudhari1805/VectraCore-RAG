"""
api/main.py
-----------
FastAPI application for Grid07.

Start via ``python run.py`` from the project root — never invoke this file
directly so that ``core.*`` imports resolve against the project root.

Endpoints
---------
GET  /                      Health check
GET  /api/bots              List all bot personas
POST /api/route             Route a post to matching bots
POST /api/generate          Trigger a bot to autonomously generate a post
POST /api/reply             Bot replies to a thread (RAG + injection defence)
GET  /api/feed              All generated posts from bot memory
GET  /api/memory/{bot_id}   Memory summary for a specific bot
GET  /dashboard             Serve the interactive dashboard
"""

from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    BotListResponse,
    BotSummary,
    FeedResponse,
    GenerateRequest,
    HealthResponse,
    MatchedBot,
    MemoryResponse,
    PostRecord,
    ReplyRequest,
    ReplyResponse,
    RouteRequest,
    RouteResponse,
)
from api.security import check_rate_limit, verify_api_key
from core.bot_memory import get_memory
from core.combat_engine import generate_defense_reply
from core.config import settings
from core.content_engine import generate_post
from core.logging_config import configure_logging, get_logger
from core.personas import PERSONAS
from core.router import build_index, route_post

configure_logging()
log = get_logger(__name__)

_APP_VERSION = "2.0.0"
_DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

app = FastAPI(
    title="Grid07 AI Engine",
    description="Cognitive routing, autonomous content generation, and RAG combat engine.",
    version=_APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

if _DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_DASHBOARD_DIR)), name="static")

_shared_deps = [Depends(verify_api_key), Depends(check_rate_limit)]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    """Build the persona index and initialise bot memories on first boot."""
    log.info("startup_begin")
    build_index()
    for bot_id in PERSONAS:
        get_memory(bot_id)
    log.info("startup_complete", dashboard_url=f"http://{settings.host}:{settings.port}/dashboard")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Return a simple liveness indicator."""
    return HealthResponse(status="ok", service="Grid07 AI Engine", version=_APP_VERSION)


@app.get(
    "/api/bots",
    response_model=BotListResponse,
    tags=["Bots"],
    dependencies=_shared_deps,
)
def list_bots() -> BotListResponse:
    """Return metadata for all configured bot personas."""
    return BotListResponse(
        bots=[
            BotSummary(id=pid, display_name=p["display_name"], description=p["description"])
            for pid, p in PERSONAS.items()
        ]
    )


@app.post(
    "/api/route",
    response_model=RouteResponse,
    tags=["Router"],
    dependencies=_shared_deps,
)
def route(req: RouteRequest) -> RouteResponse:
    """
    Route ``post_content`` to all bots whose cosine similarity exceeds ``threshold``.

    Returns an ordered list of matched bots with their similarity scores.
    """
    matches = route_post(req.post_content, threshold=req.threshold)
    log.info("route_post", matched_count=len(matches), threshold=req.threshold)
    return RouteResponse(
        post_content=req.post_content,
        threshold=req.threshold,
        matched_bots=[MatchedBot(**m) for m in matches],
        total_matched=len(matches),
    )


@app.post(
    "/api/generate",
    tags=["Content"],
    dependencies=_shared_deps,
)
def generate(req: GenerateRequest) -> dict:
    """
    Trigger a bot to autonomously generate a post via the LangGraph pipeline.

    The bot selects a topic, searches for news, recalls past opinions, and
    drafts a ≤280-character post that is stored in its memory.
    """
    if req.bot_id not in PERSONAS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot '{req.bot_id}' not found. Available: {list(PERSONAS)}",
        )
    result = generate_post(req.bot_id)
    log.info("post_generated", bot_id=req.bot_id, topic=result.get("topic"))
    return result


@app.post(
    "/api/reply",
    response_model=ReplyResponse,
    tags=["Combat"],
    dependencies=_shared_deps,
)
def reply(req: ReplyRequest) -> ReplyResponse:
    """
    Generate an in-character reply using the RAG combat engine.

    Includes prompt-injection detection and enforced persona fidelity.
    """
    if req.bot_id not in PERSONAS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot '{req.bot_id}' not found.",
        )

    result = generate_defense_reply(
        bot_id=req.bot_id,
        parent_post=req.parent_post,
        comment_history=[c.model_dump() for c in req.comment_history],
        human_reply=req.human_reply,
    )
    return ReplyResponse(**result)


@app.get(
    "/api/feed",
    response_model=FeedResponse,
    tags=["Feed"],
    dependencies=_shared_deps,
)
def get_feed() -> FeedResponse:
    """Return all generated posts from all bot memories, newest first."""
    all_posts: list[dict] = []
    for bot_id in PERSONAS:
        memory = get_memory(bot_id)
        for post in memory.posts:
            all_posts.append(
                {
                    "bot_id": bot_id,
                    "display_name": PERSONAS[bot_id]["display_name"],
                    **post,
                }
            )

    all_posts.sort(key=lambda x: x["timestamp"], reverse=True)
    return FeedResponse(
        total=len(all_posts),
        posts=[PostRecord(**p) for p in all_posts],
    )


@app.get(
    "/api/memory/{bot_id}",
    response_model=MemoryResponse,
    tags=["Memory"],
    dependencies=_shared_deps,
)
def get_bot_memory(bot_id: str) -> MemoryResponse:
    """Return memory stats and the five most recent posts for a specific bot."""
    if bot_id not in PERSONAS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot '{bot_id}' not found.",
        )
    memory = get_memory(bot_id)
    return MemoryResponse(
        bot_id=bot_id,
        total_posts=len(memory.posts),
        summary=memory.summary(),
        recent_posts=memory.posts[-5:],
    )


@app.get("/dashboard", tags=["Dashboard"], include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the interactive Grid07 dashboard."""
    index_path = _DASHBOARD_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard not found.",
        )
    return FileResponse(str(index_path))
