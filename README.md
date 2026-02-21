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
├── data/
│   ├── knowledge/      # Sample knowledge base markdown
│   ├── tender/         # Tender drafting guidance
│   ├── finance/        # Financial FAQs (CSV)
│   └── web_samples/    # Mock website HTML pages for crawler demo
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
- Access to a GPT-compatible model (e.g., OpenAI) and API key, **or**
- A local open-source model server via [Ollama](https://ollama.com/) (no API key required).

---

## Configuration (.env)

All runtime configuration is managed via environment variables loaded through Pydantic settings (`utils/config.py`). Create a `.env` file in the project root with the variables you require. The `.env` file **is already ignored** by Git via `.gitignore` to keep secrets out of version control.

```ini
# .env

# Application metadata
CHATBOT_APP_NAME="Enterprise RAG Chatbot"
CHATBOT_APP_ENV="development"

# LLM configuration
CHATBOT_LLM_PROVIDER="placeholder" # placeholder | openai | ollama
CHATBOT_LLM_API_KEY="sk-xxx"      # Required for openai provider
CHATBOT_LLM_MODEL="gpt5-mini"     # e.g. gpt-4o-mini (openai) or llama3.2 (ollama)
CHATBOT_OLLAMA_BASE_URL="http://127.0.0.1:11434"
CHATBOT_LLM_COST_PER_1K_TOKENS=0.002  # Optional cost estimation
CHATBOT_ENABLE_DEEPEVAL=true          # Enable deepeval metrics when API key is present

# Storage locations (override if needed)
CHATBOT_CHROMA_PERSIST_DIRECTORY=.chroma_store
CHATBOT_DATA_DIRECTORY=data
CHATBOT_QA_CACHE_DIRECTORY=generated_qa
```

> **Note:** If `CHATBOT_LLM_PROVIDER=ollama`, the app uses your local open-source model endpoint and does not require `CHATBOT_LLM_API_KEY`.
>
> If no provider is configured, the application uses a deterministic placeholder LLM and heuristic evaluation, allowing the stack to run fully offline.
>
> `.env.uat` and `.env.production` are committed as deployment templates for CI/CD. Keep real credentials out of these files (use GitHub Environment secrets for sensitive values).

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
  "max_qas_per_document": 2,
  "website_urls": [
    "https://www.iso.org/about-us.html",
    "https://support.microsoft.com/en-us/topic/purchase-approval-policy-sample-1234567890",
    "https://www.un.org/en/about-us"
  ]
}
```

Response includes documents indexed, QA pairs generated, and metadata with:
- `id`
- `createdAt`
- `timeTaken`
- `apiCost`

```json
{
  "id": "req_f3c90b4d5a2e4f0fbf7b33d01b7b2f41",
  "createdAt": "2026-02-21T07:30:00.000000+00:00",
  "timeTaken": 215.37,
  "apiCost": null,
  "cost": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0
  },
  "documents_indexed": 6,
  "qa_pairs_generated": 6,
  "qa_output_path": "generated_qa/qa_pairs_20260221T073000Z.json"
}
```

> ℹ️ **Tip:** The listed URLs are public pages with stable text content, making them suitable demos for external website ingestion.

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
- `id`
- `createdAt`
- `timeTaken`
- `apiCost`

```json
{
  "id": "req_0b9ad3c6f1c8428ab6d7098a994c1df3",
  "createdAt": "2026-02-21T07:31:12.120000+00:00",
  "timeTaken": 342.58,
  "apiCost": 0.0012,
  "cost": {
    "input_tokens": 180,
    "output_tokens": 220,
    "total_tokens": 400,
    "estimated_cost_usd": 0.0012
  },
  "answer": "Tender drafting should highlight compliance with Policy A, including CFO approval for spends above USD 10,000...",
  "citations": [
    {
      "source": "data/tender/tender_drafting_guide.md",
      "snippet": "Tender drafting should highlight compliance with policy A...",
      "score": 0.87
    },
    {
      "source": "data/web_samples/tender_hub.html",
      "snippet": "Include a compliance matrix referencing Policy A approval checkpoints...",
      "score": 0.79
    }
  ]
}
```

### `POST /api/evaluation`
Evaluates a question/answer pair against ground-truth references.

```json
{
  "question": "Summarise policy A",
  "answer": "Policy A mandates CFO approval for purchases.",
  "references": [{ "context": "Policy A requires CFO approval." }]
}
```

Responds with per-metric scores (reference overlap, question coverage, or deepeval metrics) plus an average. Metadata fields include `id`, `createdAt`, `timeTaken`, and `apiCost`.

```json
{
  "id": "req_58f921de4b7a4a6693f1f4a7a4cd9151",
  "createdAt": "2026-02-21T07:32:05.450000+00:00",
  "timeTaken": 129.44,
  "apiCost": null,
  "cost": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0
  },
  "metrics": [
    {
      "metric": "ReferenceOverlap",
      "score": 0.82,
      "passed": true,
      "feedback": "Checks for hallucination via reference overlap."
    },
    {
      "metric": "QuestionCoverage",
      "score": 0.74,
      "passed": true,
      "feedback": "Ensures answer addresses major terms in the question."
    }
  ],
  "average_score": 0.78
}
```

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

The container installs dependencies, bundles the application code, and starts Uvicorn on port 8000.

If you want open-source inference without GPT keys, run with Ollama mode enabled:

```bash
docker run \
  -p 8000:8000 \
  -p 11434:11434 \
  -e CHATBOT_LLM_PROVIDER=ollama \
  -e CHATBOT_LLM_MODEL=llama3.2 \
  -e CHATBOT_OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  enterprise-chatbot
```

In Ollama mode, container startup will:
- start the local Ollama server,
- pull `CHATBOT_LLM_MODEL` (for example `llama3.2`),
- then start the FastAPI app.

> **CI note:** Pulling an Ollama model can significantly increase startup time on first run.
> For CI pipelines, prefer `CHATBOT_LLM_PROVIDER=placeholder` (or a pre-warmed/cached Ollama setup) to keep test jobs fast and deterministic.

---

## Continuous Integration / Continuous Deployment

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on:
- Pushes and pull requests to any branch
- Manual dispatch
- Tags ending in `-uat` or `-prod` (trigger deployments)

### Jobs
- **build-and-test**: Sets up Python 3.10, installs `uv`, installs dependencies, and runs pytest.
- **deploy-uat**: Runs on tags like `0.0.1-uat`; builds Docker image and performs sample UAT deployment.
- **deploy-prod**: Runs on tags like `0.0.1-prod`; builds Docker image and performs sample production deployment.

Both deploy jobs reference GitHub environments (`uat`, `production`) for protection rules and secrets. Ensure these environments exist in the repository settings before tagging releases.

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
