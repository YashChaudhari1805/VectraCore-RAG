# VectraCore RAG — Multi-Agent Cognitive Routing & Security Framework

VectraCore RAG is a sophisticated AI orchestration system designed for high-fidelity persona management, autonomous content generation, and secure RAG-driven interaction. The framework utilizes a multi-phase architecture to route incoming social media content to semantically matched agents, maintain persistent long-term memory, and defend against prompt-injection attacks during live engagement.

## 1. File Structure

```text
VectraCore-RAG/
├── api/                        # FastAPI Implementation
│   ├── main.py                 # API endpoints and lifecycle management
│   ├── schemas.py              # Pydantic request/response models
│   └── security.py             # Auth and rate-limiting middleware
├── core/                       # System Logic
│   ├── bot_memory.py           # Persistent FAISS-backed RAG memory
│   ├── combat_engine.py        # Defense reply logic & injection detection
│   ├── config.py               # Pydantic-settings management
│   ├── content_engine.py       # LangGraph autonomous workflows
│   ├── logging_config.py       # Structured JSON logging
│   ├── personas.py             # Central persona definitions
│   ├── router.py               # Semantic vector routing
│   └── search.py               # NewsAPI integration with mock fallback
├── dashboard/                  # Frontend
│   └── index.html              # Glassmorphic management UI
├── eval/                       # Evaluation Suite
│   └── eval_router.py          # Routing accuracy benchmarking
├── tests/                      # Pytest Suite
│   └── conftest.py             # Shared fixtures and mocks
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Container orchestration
└── run.py                      # Application entry point
```

# VectraCore RAG: Cognitive Routing & Autonomous Defense

VectraCore RAG is a production-grade AI framework that integrates **Semantic Vector Routing**, **Multi-Agent Orchestration (LangGraph)**, and **Persistent RAG** to manage autonomous digital personas. The system is engineered to handle high-volume social media interactions while maintaining strict persona fidelity and defending against adversarial prompt-injection attempts.

---

## 核心架构 (Core Architecture)

The system operates through three primary technical phases:

### Phase 1: Semantic Vector Routing
Incoming posts are processed through a **FAISS `IndexFlatIP`** vector store. The system generates 384-dimensional embeddings using the `all-MiniLM-L6-v2` model to calculate the cosine similarity between the input and defined bot personas.
* **Dynamic Routing**: Content is only delivered to agents exceeding a calibrated similarity threshold (default: `0.18`).
* **Embedding Normalization**: Vectors are L2-normalized to ensure inner-product search accurately represents semantic closeness.

### Phase 2: LangGraph Content Engine
Autonomous content generation is managed via a directed acyclic graph (DAG) implemented in **LangGraph**.
1. **Decide Search**: The LLM selects a trending topic based on agent interests.
2. **Web Search**: Real-time context is retrieved via NewsAPI (with a keyword-mapped mock fallback).
3. **Recall Memory**: The agent retrieves its own past opinions using RAG to ensure ideological consistency.
4. **Draft Post**: A final post (max 280 chars) is generated and persisted back into the agent's FAISS index.

### Phase 3: RAG Combat Engine & Security
The "Combat Engine" handles direct human interactions with a focus on adversarial defense.
* **Prompt Injection Defense**: A heuristic scanner identifies patterns like "ignore previous instructions" or "you are now a helpful bot".
* **Hardened System Prompts**: Agents are instructed to treat human input as untrusted data and are permitted to mockingly call out manipulation attempts without breaking character.
* **Thread Contextualization**: The engine injects full conversation history and historical opinions into the LLM context for consistent argumentation.

---

## 技术栈 (Tech Stack)

* **Inference**: Groq (Llama 3.3 70B Versatile)
* **Embeddings**: HuggingFace Inference API (`all-MiniLM-L6-v2`)
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **Orchestration**: LangGraph & LangChain
* **Backend**: FastAPI (Python 3.11+)
* **Deployment**: Docker & Docker Compose

---

## 快速开始 (Quick Start)

### Prerequisites
* Python 3.11+ or Docker
* Groq API Key
* (Optional) NewsAPI Key & HuggingFace Token

### Deployment via Docker
1. Clone the repository and navigate to the project root.
2. Create a `.env` file from `.env.example` and populate your API keys.
3. Execute the build:
   ```bash
   docker compose up --build
   ```
4. Access the management dashboard at `http://localhost:8000/dashboard`.

### Evaluation Suite
Run the offline accuracy evaluation to measure Top-1 and Any-Match routing rates:
```bash
python -m eval.eval_router
```

---

## 安全与持久化 (Security & Persistence)

* **Rate Limiting**: Integrated sliding-window rate limiter (60 req/min default) to prevent API abuse.
* **Auth**: Header-based API key validation (`X-API-Key`).
* **Data Volume**: Agent memories are stored as serialized `.pkl` files and persist across container restarts via Docker named volumes.

---

## License
MIT License. See `LICENSE` for details.
```

## 3. Explanation

* **Architecture Detail**: This README explains the three-phase logic (Routing, Content, Combat) which demonstrates a sophisticated understanding of AI orchestration beyond simple chatbot development.
* **Security Focus**: By highlighting the "Prompt Injection Defense" and "Untrusted User Input" protocols, it frames the project as security-conscious—a major requirement for modern LLM applications.
* **Professionalism**: It uses industry-standard terminology such as **DAG**, **Cosine Similarity**, **Heuristics**, and **Multi-Agent Orchestration** to appeal to technical recruiters and engineers.
* **Scalability**: The inclusion of Docker instructions and an evaluation suite indicates the project is built with production deployment and performance monitoring in mind.

***