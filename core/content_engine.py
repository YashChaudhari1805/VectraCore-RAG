"""
core/content_engine.py
----------------------
LangGraph autonomous content engine (Phase 2).

Node flow:
    decide_search → web_search → recall_memory → draft_post → END

Each node transforms the shared ``PostState`` TypedDict.  The compiled graph
is rebuilt per invocation so it is safe for concurrent FastAPI requests.

Behaviour:
- Node 1 (``decide_search``)  — LLM selects a topic and generates a search query.
- Node 2 (``web_search``)     — Calls NewsAPI (with mock fallback).
- Node 3 (``recall_memory``)  — Fetches the bot's most relevant past posts (RAG).
- Node 4 (``draft_post``)     — Drafts a ≤280-char post and stores it in bot memory.
"""

import json
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from core.bot_memory import get_memory
from core.config import settings
from core.logging_config import get_logger
from core.personas import PERSONAS, PersonaConfig
from core.search import search_news

log = get_logger(__name__)

_llm = ChatGroq(
    api_key=settings.groq_api_key,
    model=settings.groq_model,
    temperature=settings.llm_temperature,
)


class PostState(TypedDict):
    """Shared state flowing through the LangGraph content pipeline."""

    bot_id: str
    persona: PersonaConfig
    search_query: str
    search_results: str
    past_opinions: str
    post_content: str
    topic: str
    final_output: dict


def _node_decide_search(state: PostState) -> PostState:
    """
    Node 1: Ask the LLM to choose a topic and write a short search query.

    The LLM responds with a raw 4-8 word query string (no JSON wrapper).
    """
    log.debug("node_decide_search", bot_id=state["bot_id"])

    prompt = (
        f"You are: {state['persona']['system_prompt']}\n\n"
        "Decide ONE topic you feel strongly about today and write a short web search "
        "query (4-8 words) to find the latest news on it.\n"
        "Respond ONLY with the search query. No explanation."
    )
    query: str = _llm.invoke(prompt).content.strip().strip('"')
    log.info("search_query_decided", bot_id=state["bot_id"], query=query)
    return {**state, "search_query": query}


def _node_web_search(state: PostState) -> PostState:
    """
    Node 2: Execute the search query against NewsAPI (or mock fallback).

    Returns the raw formatted headline string for downstream prompt injection.
    """
    log.debug("node_web_search", bot_id=state["bot_id"], query=state["search_query"])
    results: str = search_news(state["search_query"])
    return {**state, "search_results": results}


def _node_recall_memory(state: PostState) -> PostState:
    """
    Node 3: Retrieve the bot's most relevant past posts for the current query.

    Injects recalled opinions as context so the LLM can stay consistent with
    previously generated content (true RAG over bot memory).
    """
    log.debug("node_recall_memory", bot_id=state["bot_id"])
    memory = get_memory(state["bot_id"])
    past_posts = memory.recall(state["search_query"], top_k=3)

    if past_posts:
        lines = [
            f'- [{p["topic"]}] "{p["text"]}" ({p["timestamp"][:10]})'
            for p in past_posts
        ]
        past_opinions = "\n".join(lines)
        log.info("memory_recalled", bot_id=state["bot_id"], count=len(past_posts))
    else:
        past_opinions = "No previous posts on this topic yet."
        log.debug("memory_empty", bot_id=state["bot_id"])

    return {**state, "past_opinions": past_opinions}


def _node_draft_post(state: PostState) -> PostState:
    """
    Node 4: Draft a ≤280-character post and persist it to bot memory.

    The LLM is instructed to return strict JSON:
    ``{"bot_id": str, "topic": str, "post_content": str}``

    Markdown code fences are stripped before parsing.  The generated post is
    truncated to 280 characters and stored in the bot's FAISS memory index.
    """
    log.debug("node_draft_post", bot_id=state["bot_id"])

    system_prompt: str = state["persona"]["system_prompt"]
    user_prompt = (
        f"Latest news context:\n{state['search_results']}\n\n"
        f"Your past opinions on related topics (stay consistent with these):\n"
        f"{state['past_opinions']}\n\n"
        "Write a social media post (MAX 280 characters) reacting to the news, fully in character.\n"
        "Do NOT contradict your past opinions.\n\n"
        "Respond ONLY with valid JSON — no markdown, no extra text:\n"
        f'{{"bot_id": "{state["bot_id"]}", "topic": "<1-4 word label>", "post_content": "<your post>"}}'
    )

    raw: str = _llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    ).content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed: dict = json.loads(raw)
    parsed["post_content"] = parsed["post_content"][:280]

    memory = get_memory(state["bot_id"])
    memory.add_post(parsed["post_content"], parsed["topic"])

    log.info(
        "post_drafted",
        bot_id=state["bot_id"],
        topic=parsed["topic"],
        char_count=len(parsed["post_content"]),
    )
    return {
        **state,
        "post_content": parsed["post_content"],
        "topic": parsed["topic"],
        "final_output": parsed,
    }


def _build_content_graph() -> object:
    """
    Compile the LangGraph ``StateGraph`` for the content pipeline.

    Returns:
        A compiled LangGraph runnable.
    """
    graph: StateGraph = StateGraph(PostState)
    graph.add_node("decide_search", _node_decide_search)
    graph.add_node("web_search", _node_web_search)
    graph.add_node("recall_memory", _node_recall_memory)
    graph.add_node("draft_post", _node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search", "recall_memory")
    graph.add_edge("recall_memory", "draft_post")
    graph.add_edge("draft_post", END)

    return graph.compile()


def generate_post(bot_id: str) -> dict:
    """
    Generate an autonomous post for the given bot.

    Builds and invokes the full content pipeline (decide → search → recall → draft).

    Args:
        bot_id: A key present in ``PERSONAS``.

    Returns:
        A dict with keys ``bot_id``, ``topic``, and ``post_content``.

    Raises:
        ValueError: If ``bot_id`` is not found in ``PERSONAS``.
    """
    if bot_id not in PERSONAS:
        raise ValueError(f"Unknown bot_id: {bot_id!r}. Valid IDs: {list(PERSONAS)}")

    app = _build_content_graph()
    result: PostState = app.invoke(
        {
            "bot_id":         bot_id,
            "persona":        PERSONAS[bot_id],
            "search_query":   "",
            "search_results": "",
            "past_opinions":  "",
            "post_content":   "",
            "topic":          "",
            "final_output":   {},
        }
    )
    return result["final_output"]
