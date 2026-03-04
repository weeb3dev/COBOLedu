"""Voyage Code 3 embedding model configuration with query-level caching."""

from __future__ import annotations

from cachetools import LRUCache
from llama_index.embeddings.voyageai import VoyageEmbedding

from src.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, VOYAGE_API_KEY

_embed_cache: LRUCache = LRUCache(maxsize=512)


class CachedVoyageEmbedding(VoyageEmbedding):
    """VoyageEmbedding with an LRU cache on query embeddings."""

    def _get_query_embedding(self, query: str) -> list[float]:
        cached = _embed_cache.get(query)
        if cached is not None:
            return cached
        result = super()._get_query_embedding(query)
        _embed_cache[query] = result
        return result

    async def _aget_query_embedding(self, query: str) -> list[float]:
        cached = _embed_cache.get(query)
        if cached is not None:
            return cached
        result = await super()._aget_query_embedding(query)
        _embed_cache[query] = result
        return result


def get_embed_model() -> CachedVoyageEmbedding:
    """Return a configured CachedVoyageEmbedding instance (voyage-code-3, 1024-dim)."""
    return CachedVoyageEmbedding(
        model_name=EMBEDDING_MODEL,
        voyage_api_key=VOYAGE_API_KEY,
        output_dimension=EMBEDDING_DIMENSION,
        truncation=True,
        embed_batch_size=128,
    )
