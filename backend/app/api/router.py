from fastapi import APIRouter, Depends, HTTPException
from typing import Any

from app.api.schemas import (
    BotListResponse, BotBase, FeedResponse, GenerateRequest,
    RouteRequest, RouteResponse, ReplyRequest, ReplyResponse
)
from app.api.dependencies import verify_api_key
from app.models.persona import PERSONAS
from app.services.rag_engine import rag_engine
from app.services.content_service import ContentService
from app.services.combat_engine import CombatDefenseService

# Initialize the router
api_router = APIRouter()

@api_router.post("/auth/verify", dependencies=[Depends(verify_api_key)])
def verify_auth() -> dict:
    """If the dependency passes, the key is valid."""
    return {"valid": True, "auth_required": True}

@api_router.get("/bots", response_model=BotListResponse, dependencies=[Depends(verify_api_key)])
def get_bots() -> Any:
    bots = [BotBase(id=p.id, display_name=p.display_name, description=p.description) for p in PERSONAS.values()]
    return {"bots": bots}

@api_router.get("/memory/{bot_id}", dependencies=[Depends(verify_api_key)])
def get_bot_memory(bot_id: str) -> dict:
    if bot_id not in PERSONAS:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    posts = rag_engine.get_memory(bot_id)
    persona = PERSONAS[bot_id]
    return {
        "summary": f"Vector Memory Stream for {persona.display_name}",
        "recent_posts": posts
    }

@api_router.get("/feed", response_model=FeedResponse, dependencies=[Depends(verify_api_key)])
def get_feed() -> Any:
    return ContentService.get_global_feed()

@api_router.post("/generate", dependencies=[Depends(verify_api_key)])
def generate_post(req: GenerateRequest) -> dict:
    try:
        return ContentService.generate_post(req.bot_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@api_router.post("/route", response_model=RouteResponse, dependencies=[Depends(verify_api_key)])
def route_content(req: RouteRequest) -> Any:
    matches = rag_engine.semantic_route(req.post_content)
    return {"matched_bots": matches}

@api_router.post("/reply", response_model=ReplyResponse, dependencies=[Depends(verify_api_key)])
def test_combat_reply(req: ReplyRequest) -> Any:
    try:
        return CombatDefenseService.evaluate_reply(req.bot_id, req.parent_post, req.human_reply)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))