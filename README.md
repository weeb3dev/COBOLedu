# COBOLedu

A RAG (Retrieval-Augmented Generation) system that makes the GnuCOBOL legacy codebase queryable through natural language. Ask questions about compiler internals, runtime behavior, COBOL test programs, and more — get answers with exact file paths and line numbers.

## Tech Stack

- **Embeddings:** Voyage Code 3 (1024-dim, code-optimized)
- **Vector DB:** Pinecone (serverless, cosine similarity)
- **Framework:** LlamaIndex (orchestration, node parsing, query engine)
- **LLM:** Claude via Anthropic API
- **API:** FastAPI + Uvicorn
- **Target Codebase:** [GnuCOBOL](https://github.com/OCamlPro/gnucobol) — open-source COBOL compiler (C source + COBOL test programs)

## Setup

### 1. Clone and create venv

```bash
git clone <your-repo-url> COBOLedu
cd COBOLedu
python3.12 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the `.env` template and fill in your API keys:

```bash
cp .env .env  # already exists as a template
```

Required keys:

| Variable | Source |
|---|---|
| `VOYAGE_API_KEY` | [dash.voyageai.com](https://dash.voyageai.com/) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io/) |
| `PINECONE_INDEX_NAME` | Your Pinecone index name (default: `coboledu`) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |

### 4. Clone GnuCOBOL source

```bash
git clone https://github.com/OCamlPro/gnucobol.git gnucobol-source
```

## Project Structure

```
COBOLedu/
├── .env                          # API keys (gitignored)
├── .gitignore
├── README.md
├── requirements.txt
├── gnucobol-source/              # GnuCOBOL repo (gitignored)
├── src/
│   ├── config.py                 # Env vars and constants
│   ├── ingestion/                # File discovery & preprocessing
│   ├── chunking/                 # Syntax-aware code splitting
│   ├── retrieval/                # Embeddings, vector store, query engine
│   ├── api/                      # FastAPI web server
│   └── evaluation/               # Retrieval quality metrics
├── scripts/                      # One-off utility scripts
└── tests/                        # Test fixtures and query ground truth
```
