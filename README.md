# COBOLedu

A RAG (Retrieval-Augmented Generation) system that makes the GnuCOBOL legacy codebase queryable through natural language. Ask questions about compiler internals, runtime behavior, COBOL test programs, and more — get answers with exact file paths and line numbers.

## Tech Stack

- **Embeddings:** Voyage Code 3 (1024-dim, code-optimized)
- **Reranking:** Voyage rerank-2.5 (cross-encoder re-scoring)
- **Vector DB:** Pinecone (serverless, cosine similarity)
- **Framework:** LlamaIndex (orchestration, node parsing, query engine)
- **LLM:** Claude Sonnet 4 via Anthropic API (with prompt caching)
- **Observability:** Langfuse (tracing, evaluation experiments)
- **API:** FastAPI + Uvicorn
- **Target Codebase:** [GnuCOBOL](https://github.com/OCamlPro/gnucobol) — open-source COBOL compiler (C source + COBOL test programs)

## Retrieval Pipeline

```
Query
  → COBOL-aware query expansion (19 keyword patterns + identifier decomposition)
  → Voyage Code 3 embedding (with LRU cache)
  → Multi-pass Pinecone retrieval:
      • Primary: top-20 dense similarity search
      • Secondary: language-filtered pass (COBOL/C routing)
      • File-hint pass: targeted retrieval for files referenced in expansions
      • Identifier pass: COBOL-style identifier search (e.g., CUSTOMER-RECORD)
  → Voyage rerank-2.5 cross-encoder re-scoring with file diversity (max 3 per file) → top-5
  → Claude Sonnet 4 answer generation (with prompt caching)
  → Response cached via TTLCache (1 hour TTL)
```

### Performance

| Metric | Baseline | Current |
|---|---|---|
| Precision@5 | 54% | **71%** |
| Term coverage | — | 93.8% |
| Codebase coverage | 100% | 100% |

Measured via Langfuse experiment tracking across a 20-item evaluation dataset (`coboledu-eval-v1`).

## Query Interface

The web UI at `/` provides a full-featured interface for exploring the codebase:

- **Natural language input** — free-text questions with mode tabs (Search, Explain, Dependencies, Patterns, Docs, Business Logic) and example query chips
- **Syntax-highlighted code snippets** — Prism.js with C and COBOL grammars applied to source previews and LLM answer code blocks
- **File paths and line numbers** — each result shows `file:start-end` badges with chunk type
- **Relevance scores** — Voyage rerank confidence score displayed per source
- **LLM-generated answers** — streaming markdown rendered in real time via SSE
- **Full file drill-down** — "Expand" button fetches surrounding context (+/- 50 lines) with line numbers and highlighted original range; "Collapse" to restore preview
- **Copy everywhere** — copy buttons on source cards, the full answer, and individual code blocks with "Copied!" feedback

## Code Understanding Features

Five specialized endpoints beyond basic search:

| Feature | Endpoint | Description |
|---|---|---|
| Code Explanation | `POST /api/explain` | Structured breakdown: purpose, I/O, logic, side effects |
| Dependency Mapping | `POST /api/dependencies` | Callers/callees via PERFORM/CALL analysis |
| Pattern Detection | `POST /api/patterns` | Semantic search for code matching a pattern description |
| Documentation Gen | `POST /api/docs` | Auto-generated markdown docs for functions/paragraphs |
| Business Logic | `POST /api/business-logic` | Extract business rules from COBOL paragraphs |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | Pinecone index statistics |
| `POST` | `/api/query` | Synchronous query with sources |
| `POST` | `/api/query/stream` | SSE streaming answer with sources |
| `POST` | `/api/explain` | Code explanation |
| `POST` | `/api/dependencies` | Dependency mapping |
| `POST` | `/api/patterns` | Pattern detection |
| `POST` | `/api/docs` | Documentation generation |
| `POST` | `/api/business-logic` | Business logic extraction |
| `GET` | `/api/file` | Full file content for drill-down (with line range context) |

## Caching

Three layers of caching reduce latency and cost:

- **Anthropic prompt caching** — system prompts use `cache_control: ephemeral` for ~90% cost reduction on cache hits
- **Response cache** — `TTLCache(maxsize=256, ttl=3600)` returns instant results for repeated queries
- **Embedding cache** — `CachedVoyageEmbedding` with LRU cache on query embeddings avoids redundant Voyage API calls

## Setup

### 1. Clone and create venv

```bash
git clone https://github.com/weeb3dev/COBOLedu.git
cd COBOLedu
python3.12 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file at the project root with:

| Variable | Source |
|---|---|
| `VOYAGE_API_KEY` | [dash.voyageai.com](https://dash.voyageai.com/) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io/) |
| `PINECONE_INDEX_NAME` | Your Pinecone index name (default: `coboledu`) |
| `PINECONE_HOST` | Pinecone index host URL |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| `LANGFUSE_SECRET_KEY` | [us.cloud.langfuse.com](https://us.cloud.langfuse.com/) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key |
| `LANGFUSE_BASE_URL` | `https://us.cloud.langfuse.com` |

### 4. Clone GnuCOBOL source

```bash
git clone https://github.com/OCamlPro/gnucobol.git gnucobol-source
```

### 5. Ingest the codebase

```bash
python -m src.ingestion.ingest
```

### 6. Run locally

```bash
uvicorn src.api.main:app --reload
```

Open `http://localhost:8000` for the web UI.

## Evaluation

Create the eval dataset in Langfuse, then run experiments:

```bash
python -m scripts.create_eval_dataset    # one-time: uploads 20 test items
python -m scripts.run_eval --name baseline-v1
```

Each run produces precision@5, term_coverage, and latency scores visible in the Langfuse experiment dashboard.

Run the local diagnostic for a per-item precision breakdown without Langfuse:

```bash
python -m scripts.eval_diagnostic                # full pipeline (retrieval + LLM)
python -m scripts.eval_diagnostic --retrieval-only  # skip LLM, just check retrieved files
```

## Project Structure

```
COBOLedu/
├── .env                              # API keys (gitignored)
├── .gitignore
├── README.md
├── requirements.txt
├── Procfile                          # Railway deployment
├── gnucobol-source/                  # GnuCOBOL repo (gitignored)
├── src/
│   ├── config.py                     # Env vars and constants
│   ├── observability.py              # Langfuse + OpenTelemetry setup
│   ├── cli.py                        # Interactive CLI for testing
│   ├── ingestion/
│   │   ├── discover.py               # File discovery & filtering
│   │   ├── preprocess.py             # .at file COBOL extraction
│   │   └── ingest.py                 # Full ingestion pipeline
│   ├── chunking/
│   │   ├── cobol_chunker.py          # COBOL paragraph-level splitting
│   │   ├── c_chunker.py              # C function-level splitting
│   │   ├── fallback.py               # Fixed-size fallback chunker
│   │   └── orchestrator.py           # Routes files to correct chunker
│   ├── retrieval/
│   │   ├── embeddings.py             # Voyage Code 3 with LRU cache
│   │   ├── vector_store.py           # Pinecone connection
│   │   ├── query.py                  # Query pipeline: expand → retrieve → rerank → generate
│   │   └── features.py              # 5 code understanding features
│   └── api/
│       ├── main.py                   # FastAPI app with 10 endpoints
│       └── static/
│           └── index.html            # Web UI with syntax highlighting, drill-down, and copy
└── scripts/
    ├── create_index.py               # One-time Pinecone index creation
    ├── create_eval_dataset.py        # Upload eval items to Langfuse
    ├── run_eval.py                   # Run evaluation experiments
    └── eval_diagnostic.py            # Local per-item precision diagnostic
```
