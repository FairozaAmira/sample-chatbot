# Enterprise RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot designed for enterprise knowledge bases. The system ingests local document repositories (SharePoint substitutes), project emails, and QA records, builds a vector index, offers domain-aware semantic search with citations, and exposes an API surface for chatbot queries, data crawling, and response evaluation.

Key highlights:

- **LangChain-driven RAG** with domain filtering for knowledge, tender drafting, and financial queries.
- **Data crawler** that processes structured and unstructured files, generates synthetic Q&A pairs, and persists embeddings in ChromaDB.
- **Evaluation pipeline** leveraging `deepeval` when configured, with deterministic heuristics fallback.
- **FastAPI service** providing three composable endpoints and returning consistent metadata (IDs, timestamps, cost estimates).
- **Deployment ready** with Docker image and GitHub Actions CI pipeline.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Directory Layout](#directory-layout)
3. [Prerequisites](#prerequisites)
4. [Configuration (.env)](#configuration-env)
5. [Local Development](#local-development)
6. [API Endpoints](#api-endpoints)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Deployment (Docker)](#deployment-docker)
9. [Continuous Integration](#continuous-integration)
10. [Testing](#testing)

---

## Architecture

```
┌──────────────────────┐          ┌─────────────────────┐
│  Data Crawler        │          │  Vector Store       │
│  (utils/data_ing... )│ ───────▶ │  Chroma (persistent)│
└─────────┬────────────┘          └─────────┬───────────┘
          │                                 │
          ▼                                 ▼
┌──────────────────────┐          ┌─────────────────────┐
│  Generated Q&A cache │          │  Domain-aware RAG   │
│  (generated_qa/*.json)          │  (utils/rag.py)     │
└─────────┬────────────┘          └─────────┬───────────┘
          │                                 │
          ▼                                 ▼
┌────────────────────────────────────────────────────────┐
│                      FastAPI Service                    │
│  /api/crawler  /api/chatbot  /api/evaluation  /health   │
└────────────────────────────────────────────────────────┘
```

### Data ingestion
- Crawls the `data/` directory (proxy for SharePoint/project email storage).
- Supports `.txt`, `.md`, `.csv`, `.json`, `.pdf`, `.eml`, `.log` files.
- Produces document chunks and optional synthetic Q&A pairs that are indexed into Chroma.

### Retrieval & response generation
- Uses LangChain `RetrievalQA` with domain-aware filtering (`knowledge`, `tender`, `finance`).
- Provides citations by returning text snippets and source paths for top-`k` matches.
- Computes token-based cost estimates via a lightweight heuristic.

### Evaluation
- Invokes `deepeval` metrics (faithfulness, contextual precision, answer relevancy) when an API key is supplied.
- Falls back to deterministic overlap/coverage checks without external dependencies.

---

## Directory Layout

```
.
├── controllers/        # Request orchestration logic
├── routers/            # FastAPI routers for each endpoint
├── utils/              # Config, RAG, ingestion, evaluation, embeddings, etc.
├── tests/              # Pytest suite covering API flows
├── main.py             # FastAPI application factory
├── pyproject.toml      # Project dependencies and build metadata
├── Dockerfile          # Production-ready container image
├── .dockerignore       # Docker build context exclusions
└── .github/workflows/  # GitHub Actions CI pipeline
```

Local knowledge sources should be placed under `data/` (auto-created on first run). Generated Q&A artifacts are stored in `generated_qa/`, and Chroma persistence lives at `.chroma_store/` by default.

---

## Prerequisites

- Python **3.10**
- [`uv`](https://github.com/astral-sh/uv) package manager (`pip install uv`)
- Git (for cloning and running CI locally)

Optional (for full-featured evaluation and LLM responses):
- Access to a GPT-compatible model (e.g., OpenAI) and API key.

---

## Configuration (.env)

All runtime configuration is managed via environment variables loaded through Pydantic settings (`utils/config.py`). Create a `.env` file in the project root with the variables you require. The `.env` file **is already ignored** by Git via `.gitignore` to keep secrets out of version control.

```ini
# .env

# Application metadata
CHATBOT_APP_NAME="Enterprise RAG Chatbot"
CHATBOT_APP_ENV="development"

# LLM configuration
CHATBOT_LLM_API_KEY="sk-xxx"     # GPT key or equivalent
CHATBOT_LLM_MODEL="gpt5-mini"    # Default fallback model identifier
CHATBOT_LLM_COST_PER_1K_TOKENS=0.002  # Optional cost estimation
CHATBOT_ENABLE_DEEPEVAL=true          # Enable deepeval metrics when API key is present

# Storage locations (override if needed)
CHATBOT_CHROMA_PERSIST_DIRECTORY=.chroma_store
CHATBOT_DATA_DIRECTORY=data
CHATBOT_QA_CACHE_DIRECTORY=generated_qa
```

> **Note:** If you do not supply `CHATBOT_LLM_API_KEY`, the application uses a deterministic placeholder LLM and heuristic evaluation, allowing the stack to run fully offline.

---

## Local Development

1. **Install dependencies**
   ```bash
   uv pip install --system .[dev]
   ```

2. **Prepare data**
   - Place PDFs, emails (`.eml`), CSVs, etc. into `data/knowledge/`, `data/tender/`, or `data/finance/` to seed domain-specific content.

3. **Run the API locally**
   ```bash
   uvicorn main:app --reload
   ```
   The service listens at `http://127.0.0.1:8000`.

4. **Explore interactive docs**
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - Redoc: `http://127.0.0.1:8000/redoc`

   These documentation portals let you inspect request/response schemas and execute API calls (crawler, chatbot, evaluation, health) directly from the browser.

5. **Trigger ingestion** (optional before first chatbot query)
   ```bash
   curl -X POST http://127.0.0.1:8000/api/crawler -H "Content-Type: application/json" \
        -d '{"refresh_index": true, "max_qas_per_document": 2}'
   ```

---

## API Endpoints

### `GET /health`
Simple readiness probe returning `{ "status": "ok" }`.

### `POST /api/crawler`
Kicks off the ingestion pipeline.

```json
{
  "refresh_index": true,
  "max_qas_per_document": 2
}
```

Response includes documents indexed, QA pairs generated, and metadata with request ID and processing time.

### `POST /api/chatbot`
Retrieves an answer with citations from the RAG system.

```json
{
  "query": "What are the tender drafting guidelines?",
  "domain": "tender",
  "top_k": 4
}
```

Returns a structure containing:
- `answer`
- `citations` (file path, snippet, optional score)
- `cost` estimation (token counts, USD estimate)
- `id`, `created_at`, `processing_time_ms`

### `POST /api/evaluation`
Evaluates a question/answer pair against ground-truth references.

```json
{
  "question": "Summarise policy A",
  "answer": "Policy A mandates CFO approval for purchases.",
  "references": [{ "context": "Policy A requires CFO approval." }]
}
```

Responds with per-metric scores (reference overlap, question coverage, or deepeval metrics) plus an average.

---

## Evaluation Strategy

1. **deepeval metrics** *(default when API key & flag supplied)*
   - Faithfulness: detects hallucinations by comparing answer vs. retrieved context.
   - Contextual Precision: ensures answer leverages relevant knowledge snippets.
   - Answer Relevancy: measures direct alignment with the posed question.

2. **Heuristic fallback** *(offline mode)*
   - Reference overlap ratio to flag unsupported statements.
   - Question term coverage to detect incomplete answers.

These results are exposed via `/api/evaluation` for CI or human review workflows.

---

## Deployment (Docker)

Build and run using the provided Dockerfile:

```bash
docker build -t enterprise-chatbot .
docker run -p 8000:8000 --env-file .env enterprise-chatbot
```

The container installs dependencies with `uv`, bundles the application code, and starts Uvicorn on port 8000.

---

## Continuous Integration

The GitHub Actions workflow (`.github/workflows/ci.yml`) executes on pushes and pull requests to `main`:

1. Sets up Python 3.10 and installs `uv`.
2. Installs runtime plus dev dependencies via `uv pip install --system .[dev]`.
3. Runs the pytest suite.

---

## Testing

Execute the automated tests locally with:

```bash
pytest --maxfail=1 --disable-warnings -q
```

The tests spin up the FastAPI application, seed synthetic documents, and validate crawler, chatbot, and evaluation endpoints. Use these tests as a foundation for further coverage as your project evolves.

---

## Next Steps

- Integrate Azure-friendly secrets management (e.g., Key Vault) when deploying in production.
- Extend the crawler to connect directly to SharePoint APIs or email services.
- Instrument request/response logging and observability in line with enterprise compliance requirements.
