"""Voyage Code 3 embedding model configuration."""

from __future__ import annotations

from llama_index.embeddings.voyageai import VoyageEmbedding

from src.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, VOYAGE_API_KEY


def get_embed_model() -> VoyageEmbedding:
    """Return a configured VoyageEmbedding instance (voyage-code-3, 1024-dim)."""
    return VoyageEmbedding(
        model_name=EMBEDDING_MODEL,
        voyage_api_key=VOYAGE_API_KEY,
        output_dimension=EMBEDDING_DIMENSION,
        truncation=True,
        embed_batch_size=128,
    )
