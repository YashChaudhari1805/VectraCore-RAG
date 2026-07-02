from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# --- Bot & Persona Schemas ---

class BotBase(BaseModel):
    id: str
    display_name: str
    description: str

class BotListResponse(BaseModel):
    bots: List[BotBase]

# --- Feed & Content Schemas ---

class PostBase(BaseModel):
    bot_id: str
    display_name: str
    text: str
    topic: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FeedResponse(BaseModel):
    total: int
    posts: List[PostBase]

class GenerateRequest(BaseModel):
    bot_id: str

# --- Routing Schemas ---

class RouteRequest(BaseModel):
    post_content: str

class RouteMatch(BaseModel):
    bot_id: str
    similarity: float

class RouteResponse(BaseModel):
    matched_bots: List[RouteMatch]

# --- Combat Engine Schemas ---

class ReplyRequest(BaseModel):
    bot_id: str
    parent_post: str
    comment_history: List[str] = Field(default_factory=list)
    human_reply: str

class ReplyResponse(BaseModel):
    bot_id: str
    reply: str
    injection_detected: bool
    error: Optional[str] = None