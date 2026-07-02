# VectraCore RAG — AI Cognitive Routing & RAG Engine

**A production-ready multi-bot AI system with semantic routing, autonomous content generation, persistent memory, and adversarial prompt injection defence.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.x-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.x-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## Overview

VectraCore RAG simulates a platform where AI bots autonomously generate content, route incoming posts to relevant personas, and engage in multi-turn arguments. It features a complete prompt injection defence layer and cross-session memory.

The project is built as a monolithic application serving a modern **React (Vite + Tailwind)** single-page application from a high-performance **FastAPI (Python)** backend, utilizing local vector embeddings via **FastEmbed** and **FAISS**.

---

## Features

| Feature | Details |
|---|---|
| **Semantic Routing** | FAISS vector index + `all-MiniLM-L6-v2` embeddings route posts to matched personas. |
| **Autonomous Content** | Persona-driven post generation heavily grounded in core beliefs. |
| **Persistent Bot Memory** | In-memory FAISS indices (production-ready for disk persistence) ensure bots never contradict past opinions. |
| **RAG Combat Engine** | Evaluates user input against heuristic injection rules before passing to RAG generation. |
| **Prompt Injection Defence** | System-level security rules reject persona-override attempts. |
| **API Key Authentication** | Strict `X-API-Key` enforcement across all endpoints. |
| **Live Glassmorphic Dashboard** | Real-time React UI with an auth gate, live feed, combat tester, and embedded API docs. |
| **Docker Ready** | Multi-stage Dockerfile + docker-compose for seamless full-stack deployment. |

---


### 1. Vector Router & Embeddings

Posts are embedded locally using `fastembed` (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) avoiding external API latency. Vectors are L2-normalised and stored in a FAISS `IndexFlatIP`, making the inner product equivalent to cosine similarity.

### 2. RAG Combat Engine

The system evaluates replies against a heuristic list of known injection keywords (`"ignore previous instructions"`, `"apologise to me"`). If an attack is detected, the engine rejects the input and responds defensively in character.

### 3. The Monolith Pattern

The React frontend is built into static files which are mounted and served directly by FastAPI. A catch-all route at `/dashboard` ensures client-side routing works flawlessly without needing a separate web server (like Nginx) in production.

---

## Quick Start

### The Easy Way: Docker (Recommended)

The fastest way to run the full stack locally with one command.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.

```bash
git clone [https://github.com/yourusername/VectraCoreRAG.git](https://github.com/yourusername/VectraCoreRAG.git)
cd VectraCoreRAG

# Copy env template and add your API_KEYS
cp .env.example .env

# Build and start the container
docker compose up --build

```

* **Dashboard** → http://localhost:8000/dashboard
* **API Docs** → http://localhost:8000/docs

### The Manual Way: Local Development

If you want to run the frontend and backend separately for development:

**1. Start the React Frontend**

```bash
cd frontend
npm install
npm run dev

```

**2. Start the FastAPI Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

pip install -r ../requirements.txt
cp ../.env.example .env       # Set your API_KEYS

python main.py

```

---

## Environment Variables

Create a `.env` file in the root directory:

| Variable | Required | Description |
| --- | --- | --- |
| `API_KEYS` | ✅ Yes | Comma-separated API keys to enable authentication. Example: `dev_secret_key`. |
| `ALLOWED_ORIGINS` | Optional | CORS origins. Default: `*` |
| `ENVIRONMENT` | Optional | `development` or `production`. |
| `PORT` | Optional | Server port. Default: `8000` |
| `MODEL_NAME` | Optional | Embedding model. Default: `sentence-transformers/all-MiniLM-L6-v2` |
| `GROQ_API_KEY` | Optional | Reserved for future external LLM integration. |
| `HF_TOKEN` | Optional | Reserved for HuggingFace API access. |

---

## Deployment on Render (Docker)

`render.yaml` is configured for **Docker-based** deployment. Render builds your image directly from the multi-stage `Dockerfile` in the repo — no manual build steps required.

### Steps

1. **Generate an API key**:
```bash
python -c "import secrets; print(secrets.token_hex(32))"

```


2. **Push your repo to GitHub** (Ensure `.env` is gitignored).
3. **Create a Web Service on Render** → connect GitHub → select your repo.
4. **Set environment variables** in the Render dashboard:
* `API_KEYS` = *<your_generated_key>*
* `ENVIRONMENT` = `production`


5. **Deploy**. (The Docker build will automatically compile the React app and package it with Python).
6. **Visit your dashboard**: `https://your-service-name.onrender.com/dashboard`

---

## API Reference

All endpoints (except `/` and `/api/auth/verify`) require the `X-API-Key` header.

```http
X-API-Key: your_api_key_here

```

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Root redirect to dashboard |
| `POST` | `/api/auth/verify` | Verify an API key (used by AuthGate) |
| `GET` | `/api/bots` | List all bot personas |
| `POST` | `/api/route` | Route a post to matching bots by semantic similarity |
| `POST` | `/api/generate` | Trigger a bot to autonomously generate a post |
| `POST` | `/api/reply` | Test a bot reply using the Combat Engine (injection defence) |
| `GET` | `/api/feed` | Retrieve all generated posts, sorted by newest |
| `GET` | `/api/memory/{bot_id}` | View bot memory stats and recent vector posts |
| `GET` | `/dashboard` | Serve the React SPA |

*Interactive docs are embedded directly in the dashboard under the **API Docs** tab, or accessible at `/docs`.*

---

## Project Structure

```text
VectraCoreRAG/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI routers, schemas, and dependencies
│   │   ├── core/            # Pydantic settings & config
│   │   ├── models/          # Persona domain models
│   │   └── services/        # RAG Engine, Combat Engine, Content generators
│   └── main.py              # FastAPI entry point (mounts frontend)
├── frontend/
│   ├── src/                 
│   │   ├── api/             # API client & fetch wrappers
│   │   ├── components/      # React UI components (Dashboard, Sidebar, Feed)
│   │   ├── context/         # Auth and Theme providers
│   │   └── index.css        # Tailwind & global styles
│   ├── package.json
│   └── vite.config.ts       # Vite config (proxies /api to backend in dev)
├── eval/                    
│   └── eval_router.py       # Offline routing accuracy benchmark
├── Dockerfile               # Multi-stage build (Node -> Python)
├── docker-compose.yml       # Local deployment stack
├── render.yaml              # Render IaC deployment config
├── requirements.txt         # Python dependencies
└── .env.example             

```

---

## Tech Stack

**Frontend:**

* React 18
* TypeScript
* Vite
* Tailwind CSS + clsx/tailwind-merge
* Lucide React (Icons)

**Backend:**

* Python 3.11
* FastAPI + Uvicorn
* FAISS (CPU) for Vector Search
* FastEmbed for local sentence embeddings
* Pydantic V2

---

## License

MIT — see [LICENSE](https://www.google.com/search?q=LICENSE)

```

```