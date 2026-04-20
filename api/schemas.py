"""
api/schemas.py
--------------
Pydantic request and response models for the VectraCore RAG API.

Centralising schemas here keeps ``api/main.py`` focused on routing logic and
makes it easy to version models independently of the endpoints.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    """Payload for ``POST /api/route``."""

    post_content: str = Field(
        ...,
        min_length=1,
        description="Social-media post text to route to matching bots.",
        examples=["AI will cure cancer and solve climate change."],
    )
    threshold: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for a bot to be matched.",
    )


class GenerateRequest(BaseModel):
    """Payload for ``POST /api/generate``."""

    bot_id: str = Field(
        ...,
        description="Identifier of the bot that should generate a post.",
        examples=["Bot_A_TechMaximalist"],
    )


class CommentRecord(BaseModel):
    """A single comment entry within a thread."""

    author: str = Field(..., description="Display name or identifier of the comment author.")
    text: str = Field(..., description="Text content of the comment.")


class ReplyRequest(BaseModel):
    """Payload for ``POST /api/reply``."""

    bot_id: str = Field(
        ...,
        description="Identifier of the bot that should reply.",
        examples=["Bot_B_Doomer"],
    )
    parent_post: str = Field(
        ...,
        description="The original post that started the thread.",
    )
    comment_history: list[CommentRecord] = Field(
        default_factory=list,
        description="Prior comments in the thread, in chronological order.",
    )
    human_reply: str = Field(
        ...,
        min_length=1,
        description="The latest human message the bot must respond to.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class BotSummary(BaseModel):
    """A summarised view of a single bot persona."""

    id: str
    display_name: str
    description: str


class BotListResponse(BaseModel):
    """Response for ``GET /api/bots``."""

    bots: list[BotSummary]


class MatchedBot(BaseModel):
    """A single routing match result."""

    bot_id: str
    similarity: float


class RouteResponse(BaseModel):
    """Response for ``POST /api/route``."""

    post_content: str
    threshold: float
    matched_bots: list[MatchedBot]
    total_matched: int


class PostRecord(BaseModel):
    """A single stored post from bot memory."""

    bot_id: str
    display_name: str
    text: str
    topic: str
    timestamp: str


class FeedResponse(BaseModel):
    """Response for ``GET /api/feed``."""

    total: int
    posts: list[PostRecord]


class MemoryResponse(BaseModel):
    """Response for ``GET /api/memory/{bot_id}``."""

    bot_id: str
    total_posts: int
    summary: str
    recent_posts: list[dict]


class ReplyResponse(BaseModel):
    """Response for ``POST /api/reply``."""

    bot_id: str
    reply: str
    injection_detected: bool


class HealthResponse(BaseModel):
    """Response for ``GET /``."""

    status: str
    service: str
    version: str
