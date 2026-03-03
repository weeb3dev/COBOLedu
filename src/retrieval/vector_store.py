"""Pinecone vector store wiring for LlamaIndex."""

from __future__ import annotations

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME


def get_vector_store() -> PineconeVectorStore:
    """Connect to the existing Pinecone index and return a LlamaIndex wrapper."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(host=PINECONE_HOST) if PINECONE_HOST else pc.Index(name=PINECONE_INDEX_NAME)
    return PineconeVectorStore(pinecone_index=pinecone_index)


def get_index(vector_store: PineconeVectorStore) -> VectorStoreIndex:
    """Build a VectorStoreIndex from an already-populated vector store (read path for queries)."""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )
