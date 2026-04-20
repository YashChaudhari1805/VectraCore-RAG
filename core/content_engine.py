"""
core/content_engine.py
----------------------
LangGraph autonomous content engine (Phase 2 — improved).

Improvements over v1:
  - Uses real NewsAPI search (with mock fallback)
  - Bot recalls its own past posts before drafting (persistent memory RAG)
  - Past opinions injected into prompt so bot stays consistent
  - New post stored back into memory after generation

Node flow: decide_search → web_search → recall_memory → draft_post → END
"""

import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from core.personas import PERSONAS
from core.search import search_news
from core.bot_memory import get_memory

load_dotenv()

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.8,
)


# ── State ─────────────────────────────────────────────────────────────────────

class PostState(TypedDict):
    bot_id:         str
    persona:        dict        # full persona dict from PERSONAS
    search_query:   str
    search_results: str
    past_opinions:  str         # retrieved from bot memory
    post_content:   str
    topic:          str
    final_output:   dict


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_decide_search(state: PostState) -> PostState:
    """Node 1: LLM picks a topic and formats a search query based on persona."""
    print(f"\n  [Node 1] Deciding search query for {state['bot_id']}...")

    prompt = (
        f"You are: {state['persona']['system_prompt']}\n\n"
        "Decide ONE topic you feel strongly about today and write a short web search "
        "query (4-8 words) to find the latest news on it.\n"
        "Respond ONLY with the search query. No explanation."
    )
    query = llm.invoke(prompt).content.strip().strip('"')
    print(f"  [Node 1] Query: {query}")
    return {**state, "search_query": query}


def node_web_search(state: PostState) -> PostState:
    """Node 2: Calls real NewsAPI search (with mock fallback)."""
    print(f"\n  [Node 2] Searching: '{state['search_query']}'...")
    results = search_news(state["search_query"])
    print(f"  [Node 2] Results: {results[:120]}...")
    return {**state, "search_results": results}


def node_recall_memory(state: PostState) -> PostState:
    """
    Node 3 (NEW): Retrieves this bot's most relevant past posts given the
    current search query. Injects them into the prompt so the bot stays
    consistent with its own previous opinions.
    """
    print(f"\n  [Node 3] Recalling memory for {state['bot_id']}...")
    memory  = get_memory(state["bot_id"])
    past    = memory.recall(state["search_query"], top_k=3)

    if past:
        lines = [f'- [{p["topic"]}] "{p["text"]}" ({p["timestamp"][:10]})' for p in past]
        past_str = "\n".join(lines)
        print(f"  [Node 3] Found {len(past)} relevant memories")
    else:
        past_str = "No previous posts on this topic yet."
        print(f"  [Node 3] No relevant memories found")

    return {**state, "past_opinions": past_str}


def node_draft_post(state: PostState) -> PostState:
    """
    Node 4: Drafts a 280-char post using persona + news context + past opinions.
    Stores the post back into bot memory after generation.
    Output is guaranteed JSON.
    """
    print(f"\n  [Node 4] Drafting post for {state['bot_id']}...")

    system = state["persona"]["system_prompt"]

    user = f"""Latest news context:
{state['search_results']}

Your past opinions on related topics (stay consistent with these):
{state['past_opinions']}

Write a social media post (MAX 280 characters) reacting to the news, fully in character.
Do NOT contradict your past opinions.

Respond ONLY with valid JSON — no markdown, no extra text:
{{"bot_id": "{state['bot_id']}", "topic": "<1-4 word label>", "post_content": "<your post>"}}"""

    raw    = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed                  = json.loads(raw)
    parsed["post_content"]  = parsed["post_content"][:280]

    # Store this post in bot memory
    memory = get_memory(state["bot_id"])
    memory.add_post(parsed["post_content"], parsed["topic"])

    print(f"  [Node 4] Post: {parsed['post_content'][:80]}...")
    return {**state, "post_content": parsed["post_content"], "topic": parsed["topic"], "final_output": parsed}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_content_graph():
    graph = StateGraph(PostState)
    graph.add_node("decide_search",  node_decide_search)
    graph.add_node("web_search",     node_web_search)
    graph.add_node("recall_memory",  node_recall_memory)
    graph.add_node("draft_post",     node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "recall_memory")
    graph.add_edge("recall_memory", "draft_post")
    graph.add_edge("draft_post",    END)

    return graph.compile()


def generate_post(bot_id: str) -> dict:
    """
    Public API: generates a post for the given bot_id.
    Returns the final_output dict: {"bot_id", "topic", "post_content"}
    """
    if bot_id not in PERSONAS:
        raise ValueError(f"Unknown bot_id: {bot_id}")

    app = build_content_graph()
    result = app.invoke({
        "bot_id":         bot_id,
        "persona":        PERSONAS[bot_id],
        "search_query":   "",
        "search_results": "",
        "past_opinions":  "",
        "post_content":   "",
        "topic":          "",
        "final_output":   {},
    })
    return result["final_output"]
