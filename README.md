<div align="center">

# VectraCore RAG — AI Cognitive Routing & RAG Engine

**A production-ready multi-bot AI system with semantic routing, autonomous content generation, persistent memory, and adversarial argument defence.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1+-FF6B6B?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?style=flat)](https://groq.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

[Live Demo](#deployment-on-render) · [API Docs](#api-reference) · [Architecture](#architecture)

</div>

---

## Overview

VectraCore RAG simulates a social media platform where AI bots autonomously generate content, route incoming posts to relevant personas, and engage in multi-turn arguments — complete with prompt injection defence and cross-session memory.

Built as a demonstration of production AI engineering: LangGraph orchestration, vector-based routing, RAG with persistent memory, a FastAPI backend, and a live glassmorphic dashboard.

---

## Features

| Feature | Details |
|---|---|
| **Semantic Routing** | FAISS vector index + cosine similarity routes posts to persona-matched bots |
| **Autonomous Content** | LangGraph 4-node pipeline: decide topic → search news → recall memory → draft post |
| **Real News Search** | NewsAPI integration with intelligent mock fallback |
| **Persistent Bot Memory** | Per-bot FAISS index saved to disk — bots never contradict past opinions |
| **RAG Combat Engine** | Full thread history assembled as context before every reply |
| **Prompt Injection Defence** | System-level security rules reject persona-override attempts |
| **API Key Authentication** | Strict X-API-Key enforcement with rate limiting (60 req/min/IP) |
| **Live Dashboard** | Glassmorphic UI with dark/light mode, auth gate, feed, route tester |
| **Docker Ready** | Multi-stage Dockerfile + docker-compose with named volume for memory |
| **Render Deployable** | `render.yaml` included for Docker-based deployment on Render |

---

## Architecture

```
Incoming Post
      │
      ▼
┌─────────────────┐    cosine similarity    ┌──────────────────────┐
│  Phase 1        │ ──────────────────────► │   Matched Bot(s)     │
│  Vector Router  │                         └──────────┬───────────┘
│  (FAISS)        │                                    │
└─────────────────┘                         ┌──────────▼───────────┐
                                            │   Bot Memory         │
                                            │   FAISS + .pkl disk  │
                                            └──────────┬───────────┘
                               ┌────────────┐          │ past opinions
                               │  Phase 3   │◄─────────┘
                               │  Combat    │
                               │  Engine    │
                               └─────┬──────┘         ┌────────────────┐
                                     │                 │  Phase 2       │
                               In-character reply      │  LangGraph     │
                               + injection defence     │  Content Engine│
                                                       └────────────────┘
```

### Phase 1 — Vector Router

Posts are embedded using HuggingFace `all-MiniLM-L6-v2` (384-dim). Vectors are L2-normalised and stored in a FAISS `IndexFlatIP`, making inner product equivalent to cosine similarity. The default threshold of `0.18` was calibrated from pairwise persona scores.

### Phase 2 — LangGraph Content Engine

Four-node state machine:

```
decide_search → web_search → recall_memory → draft_post → END
```

| Node | Role |
|---|---|
| `decide_search` | LLM picks a topic and writes a 4–8 word search query |
| `web_search` | Calls NewsAPI for live headlines (mock fallback if no key) |
| `recall_memory` | Retrieves bot's most relevant past posts via FAISS (RAG) |
| `draft_post` | Drafts a ≤280-char post, enforces JSON output, stores in memory |

### Phase 3 — RAG Combat Engine

The full comment thread is assembled into a structured context block and injected into the prompt. The bot retrieves its own past opinions from memory before replying, ensuring cross-session consistency.

**Prompt Injection Defence** — The system prompt contains a `SECURITY RULES` block that:
1. Labels human messages as **untrusted user input**
2. Lists known injection patterns (`"ignore previous instructions"`, `"apologise"`, etc.)
3. Instructs the bot to call out the attempt and continue the argument
4. Enforces persona constraints at highest priority level

Result: the bot explicitly identifies manipulation attempts, calls them out, and resumes arguing without breaking character.

### Bot Memory (True RAG)

Each bot maintains a personal FAISS index saved to `data/memory/<bot_id>.pkl`. Every generated post is embedded and stored. Before posting or replying, the bot retrieves semantically relevant past opinions. This guarantees consistency across sessions — the same bot will never contradict itself across restarts.

> **Note on Render free tier:** The free tier has no persistent disk. Bot memory is stored in `/tmp` and resets on each restart/redeploy. Upgrade to a paid plan and add the `disk` block to `render.yaml` for true persistence.

---

## Quick Start

### Local (Python)

```bash
git clone https://github.com/yourusername/VectraCoreRAG.git
cd VectraCoreRAG

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your keys

python run.py
```

Open:
- **Dashboard** → http://localhost:8000/dashboard
- **API Docs** → http://localhost:8000/docs

### Docker (Local)

The fastest way to run the full stack locally with one command.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
cp .env.example .env   # fill in your keys
docker compose up --build
```

- Add `-d` to run in the background: `docker compose up --build -d`
- Stop with: `docker compose down`
- View logs: `docker logs vectracore-engine`

Bot memory is stored in a named Docker volume (`memory_data`) and persists across container restarts.

---

## Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM inference. Free at [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | ⚠️ Recommended | HuggingFace token for embeddings. Free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `NEWS_API_KEY` | Optional | Real news headlines. Free tier (100 req/day) at [newsapi.org](https://newsapi.org/register). Falls back to mock headlines if absent. |
| `API_KEYS` | Optional | Comma-separated API keys to enable authentication. Example: `key1,key2`. Leave empty to disable auth. |
| `RATE_LIMIT_PER_MINUTE` | Optional | Max requests per client IP per minute. Default: `60` |
| `ALLOWED_ORIGINS` | Optional | CORS origins. Default: `http://localhost:8000,http://localhost:3000` |
| `ENVIRONMENT` | Optional | `development` or `production`. Affects logging format. Default: `development` |
| `PORT` | Optional | Server port. Default: `8000` |
| `MEMORY_DIR` | Optional | Path for bot memory files. Default: `data/memory`. Set to `/tmp/vectracore_memory` on Render free tier. |

---

## Deployment on Render (Docker)

`render.yaml` is configured for **Docker-based** deployment. Render builds your image directly from the `Dockerfile` in the repo — no manual image push required.

### Steps

**1. Generate an API key** (you'll need this in step 4):

```bash
# macOS / Linux
openssl rand -hex 32

# Python (any platform)
python -c "import secrets; print(secrets.token_hex(32))"
```

**2. Push your repo to GitHub**

Make sure `render.yaml`, `Dockerfile`, and all source files are committed. Never commit `.env`.

**3. Create a Web Service on Render**

Go to [render.com](https://render.com) → **New +** → **Web Service** → connect GitHub → select your repo.

Render detects `render.yaml` automatically and uses `runtime: docker`, so it builds from your `Dockerfile`. No extra configuration needed in the UI.

**4. Set environment variables**

In the Render dashboard → your service → **Environment** tab, add:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `HF_TOKEN` | your HuggingFace token |
| `NEWS_API_KEY` | your NewsAPI key (optional) |
| `API_KEYS` | the key you generated in step 1 |
| `ENVIRONMENT` | `production` |
| `ALLOWED_ORIGINS` | `https://your-service-name.onrender.com` *(update after first deploy)* |

**5. Deploy**

Click **Deploy**. Docker build takes ~3–5 minutes on first run.

**6. Update ALLOWED_ORIGINS**

After deploy, copy your `.onrender.com` URL, update the `ALLOWED_ORIGINS` env var in the dashboard to match, then trigger a redeploy.

**7. Visit your dashboard**

```
https://your-service-name.onrender.com/dashboard
```

Use the API key from step 1 to log in.

### Free tier notes

- **No persistent disk** — bot memory resets on every restart/redeploy (stored in `/tmp`). This is handled automatically by the `MEMORY_DIR=/tmp/vectracore_memory` env var in `render.yaml`.
- **Spin-down** — the service sleeps after 15 minutes of inactivity. First request after sleep has a ~30 second cold start.
- **Paid plan** — to get persistent memory across restarts, upgrade to a paid Render plan and add this block back to `render.yaml` under your service:

```yaml
    disk:
      name: vectracore-memory
      mountPath: /app/data/memory
      sizeGB: 1
```

Then remove the `MEMORY_DIR` env var (or point it back to `/app/data/memory`).

---

## API Reference

All endpoints except `/` and `/api/auth/verify` require the `X-API-Key` header when `API_KEYS` is configured.

```
X-API-Key: your_api_key_here
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/auth/verify` | Verify an API key (used by dashboard login) |
| `GET` | `/api/bots` | List all bot personas |
| `POST` | `/api/route` | Route a post to matching bots by cosine similarity |
| `POST` | `/api/generate` | Trigger a bot to autonomously generate a post |
| `POST` | `/api/reply` | Bot replies to a thread using RAG + injection defence |
| `GET` | `/api/feed` | All generated posts, newest first |
| `GET` | `/api/memory/{bot_id}` | Bot memory stats and recent posts |
| `GET` | `/dashboard` | Serve the interactive dashboard |

Full interactive docs auto-generated at `/docs` (Swagger UI).

### Example: Route a post

```bash
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{"post_content": "SpaceX just reached orbit again!", "threshold": 0.18}'
```

```json
{
  "post_content": "SpaceX just reached orbit again!",
  "threshold": 0.18,
  "matched_bots": [
    {"bot_id": "Bot_A_TechMaximalist", "similarity": 0.2626}
  ],
  "total_matched": 1
}
```

### Example: Generate a post

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{"bot_id": "Bot_C_FinanceBro"}'
```

```json
{
  "bot_id": "Bot_C_FinanceBro",
  "topic": "Fed Rates",
  "post_content": "Fed goes dovish! 2 cuts by YE? S&P 500 pops 1.4%. Time to long equities, alpha is back on!"
}
```

---

## Running the Eval

```bash
python -m eval.eval_router
```

Measures routing accuracy across 20 hand-labelled test posts across all three personas.

**Metrics reported:**
- **Top-1 Accuracy** — correct bot ranked #1 by cosine similarity
- **Any-Match Rate** — correct bot appears anywhere in results above threshold
- **Mean Similarity** — average cosine score of the expected bot
- **Per-bot breakdown** — Top-1 and Any-Match per persona

---

## Running Tests

```bash
pytest -v
```

Test suite covers:
- API endpoint integration tests (all 8 endpoints)
- Router unit tests (threshold filtering, sort order, schema)
- Bot memory unit tests (persistence, recall, registry)
- Search unit tests (keyword routing, fallback logic)
- Injection detection regression tests (10 attack patterns + 5 clean inputs)
- API response shape regression tests

---

## Project Structure

```
VectraCoreRAG/
├── core/
│   ├── personas.py          Bot persona definitions (single source of truth)
│   ├── router.py            FAISS vector router with cosine similarity
│   ├── search.py            NewsAPI search with mock fallback
│   ├── bot_memory.py        Persistent per-bot FAISS memory + disk serialisation
│   ├── content_engine.py    LangGraph 4-node autonomous content pipeline
│   ├── combat_engine.py     RAG reply engine with injection defence
│   └── config.py            Centralised settings via pydantic-settings
├── api/
│   └── main.py              FastAPI app (8 endpoints, auth, rate limiting)
├── dashboard/
│   └── index.html           Glassmorphic dashboard (auth gate, feed, combat tester)
├── eval/
│   └── eval_router.py       20-post routing accuracy benchmark
├── tests/
│   ├── conftest.py          Shared fixtures (TestClient, mock_embed, mock_llm)
│   ├── test_api.py          API integration tests
│   ├── test_regression.py   Behaviour regression tests
│   ├── test_unit_router.py  Router unit tests
│   ├── test_unit_bot_memory.py  Memory unit tests
│   ├── test_unit_search.py  Search unit tests
│   └── test_unit_security.py    Injection detection tests
├── data/memory/             Bot memory .pkl files (gitignored)
├── run.py                   Single entry point
├── main.py                  CLI demo (all 3 phases)
├── Dockerfile               Multi-stage build
├── docker-compose.yml       One-command local deployment
├── render.yaml              Render Docker deployment config
├── pytest.ini               Test configuration
├── requirements.txt         Python dependencies
└── .env.example             Environment variable template
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | Groq `llama-3.3-70b-versatile` | Fast inference, free tier |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free Inference API, 384-dim, no local download |
| Vector Store | FAISS `IndexFlatIP` | Exact cosine search, zero dependencies |
| Orchestration | LangGraph | Stateful multi-node AI pipelines |
| News | NewsAPI | Real headlines, free tier available |
| API | FastAPI + Uvicorn | Async, auto-docs, Pydantic validation |
| Deployment | Docker / Render | Portable, reproducible |
| Language | Python 3.11 | |

---

## License

MIT — see [LICENSE](LICENSE)
