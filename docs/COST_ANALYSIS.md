# COBOLedu — AI Cost Analysis

## Development & Testing Costs

### Actual Spend During Development

Tracked via Langfuse (382 traces, 2,685 observations, 82 LLM generations).

| Cost Category | Service | Actual Usage | Unit Price | Total Cost |
|---|---|---|---|---|
| **Embeddings** | Voyage Code 3 | ~10M tokens (ingestion × 3–4 iterations + 1,045 query embeddings) | $0.06/1M tokens | **$0.00** (free tier: 200M) |
| **Reranking** | Voyage rerank-2.5 | ~4M tokens across eval + interactive queries | $0.05/1M tokens | **$0.00** (free tier: 200M) |
| **LLM (Generation)** | Claude Sonnet 4 | 222,086 input tokens + 51,211 output tokens across 82 generations | $3/1M input, $15/1M output | **$1.43** |
| **Vector Database** | Pinecone Serverless (Starter) | ~3,000–5,000 vectors, 1024 dims, 832 retrieval operations | $0 (free tier) | **$0.00** |
| **Observability** | Langfuse Cloud | 382 traces, 2,685 observations | Free tier | **$0.00** |
| **Hosting** | Railway | Deployment + demo hosting | $5/month | **$5.00** |
| | | | **API/Infra Subtotal** | **~$6.43** |
| **Dev Tooling** | Cursor (Ultra plan) | IDE + AI-assisted development | $200/month | **$200.00** |
| **Dev Tooling** | Claude Max (Desktop) | Architecture, planning, analysis | $100/month | **$100.00** |
| | | | **Total Dev Spend** | **~$306.43** |

*Dev tooling costs are fixed developer productivity subscriptions, not marginal costs of the COBOLedu system. They do not factor into production cost projections.*

### Breakdown Notes

**LLM costs (from Langfuse):** 82 Claude Sonnet 4 generations at an average cost of $0.0175 per query. Input cost totaled $0.67 and output cost totaled $0.77. Per-query cost ranged from $0.008 (short answers) to $0.026 (detailed explanations). The system prompt uses Anthropic's `cache_control: ephemeral` for prompt caching — Langfuse shows `cache_read` and `cache_write` token tracking on every generation.

**Embedding costs:** 1,045 embedding observations logged in Langfuse (487 via `CachedVoyageEmbedding`, reducing redundant API calls). Full ingestion passes of ~2.5M tokens each, plus eval and interactive query embeddings, all stayed within Voyage's 200M free token tier.

**Retrieval costs:** 832 Pinecone retrieval operations across multi-pass queries (primary + language-filtered + file-hint + identifier passes). At 2–4 reads per query, the Starter plan's 1M read units/month was never challenged.

**Eval experiment costs:** 9 experiment runs × 20 items = 180 eval traces. These account for the bulk of LLM spend, since each eval item triggers a full pipeline query. Interactive usage (38 queries across search, explain, dependencies, patterns, docs, and business-logic endpoints) was a small fraction.

---

## Production Cost Projections

### Assumptions

- **Queries per user per day:** 5
- **Actual avg cost per query:** $0.0175 (from Langfuse: $1.43 ÷ 82 generations)
- **Reranking per query:** ~10K tokens ($0.0005 at $0.05/1M) — based on `RETRIEVAL_K=20` candidates × `CHUNK_SIZE=400` tokens
- **Embedding per query:** ~50 tokens, often cached (negligible)
- **Multi-pass retrieval:** 2–4 Pinecone reads per query
- **Codebase re-indexing:** Weekly (~2.5M tokens, $0.15 per re-index)
- **Prompt cache hit rate:** ~80%+ under sustained load (system prompt identical across all queries)
- **Response cache hit rate:** ~30% (TTLCache, 1-hour TTL, identical queries)

### Monthly Cost by Scale

| Component | 100 Users | 1,000 Users | 10,000 Users | 100,000 Users |
|---|---|---|---|---|
| **LLM (per query at $0.0175 avg)** | 15K queries, ~$263 | 150K queries, ~$2,625 | 1.5M queries, ~$26,250 | 15M queries, ~$262,500 |
| **Reranking** | 150M tokens, ~$7.50 | 1.5B tokens, ~$75 | 15B tokens, ~$750 | 150B tokens, ~$7,500 |
| **Embedding (queries)** | 0.75M tokens, ~$0.05 | 7.5M tokens, ~$0.45 | 75M tokens, ~$4.50 | 750M tokens, ~$45 |
| **Embedding (re-index)** | 10M tokens, ~$0.60 | 10M tokens, ~$0.60 | 10M tokens, ~$0.60 | 10M tokens, ~$0.60 |
| **Vector DB (Pinecone)** | ~$0 (Starter) | ~$50 (Standard min) | ~$100 | ~$300 |
| **Observability (Langfuse)** | ~$0 (free) | ~$0 (free) | ~$59 (Pro) | ~$59+ (Pro) |
| **Hosting (Railway)** | ~$5 | ~$20 | ~$50 | ~$200 |
| **Total** | **~$276/mo** | **~$2,771/mo** | **~$27,214/mo** | **~$270,605/mo** |

*LLM cost dominates at 95%+ of total. The $0.0175/query figure comes directly from Langfuse data and already reflects prompt caching. Without any caching, per-query cost would be ~$0.025.*

### Key Observations

**LLM output costs dominate at every scale.** At the measured $0.0175 per query, the LLM accounts for 95% of the 100-user total. Output tokens ($15/1M) are the primary driver — input tokens are partially mitigated by prompt caching.

**Three caching layers are already implemented:**

1. **Anthropic prompt caching** (live) — system prompt uses `cache_control: ephemeral`, reducing input cost by ~90% on cache hits. Langfuse tracks cache_read and cache_write tokens per generation.
2. **Response cache** (live) — `TTLCache(maxsize=256, ttl=3600)` returns instant results for repeated identical queries within a 1-hour window, eliminating all API costs for cache hits.
3. **Embedding cache** (live) — `CachedVoyageEmbedding` with LRU cache (maxsize=512) on query embeddings. Langfuse shows 487 of 1,045 embedding calls went through the cached path.

**Additional cost reduction levers:**

1. **Model tiering:** Route simple queries to Claude Haiku 4.5 ($1/$5 per 1M tokens) and reserve Sonnet 4 for complex explanations. At 60–70% simple queries, blended per-query cost could drop to ~$0.008, reducing 100-user total to ~$125/mo.
2. **Batch API:** For non-real-time workloads (e.g., pre-generating docs via `/api/docs`), Anthropic's Batch API offers 50% off, bringing per-query cost to ~$0.009.
3. **Embedding quantization:** Voyage Code 3 supports int8 and binary output, reducing vector storage costs by 4–8× with minimal retrieval quality loss.
4. **Reranking optimization:** Reduce candidate pool from 20 to 10 for simple queries — halves reranking token cost with minimal precision impact.

**Break-even analysis:** At $20/user/month pricing, COBOLedu breaks even at approximately 14 users (at the measured $0.0175/query) or 7 users (with model tiering enabled).

---

## Cost Comparison: Build vs. Buy

| Approach | Monthly Cost (100 users) | Notes |
|---|---|---|
| **COBOLedu (current stack, measured)** | ~$276 | Full control, COBOL-optimized, 6 feature endpoints |
| **GitHub Copilot Enterprise** | ~$3,900 (100 × $39/seat) | General-purpose, not COBOL-optimized |
| **Manual code review** | ~$16,000+ (2 senior devs × $80K/yr) | Slow, doesn't scale |

COBOLedu offers a 14–58× cost advantage over alternatives for COBOL-specific code understanding.
