"""Idempotent Pinecone index creation script.

The coboledu index was created via the Pinecone dashboard, so this script
serves as a safety-net for fresh environments. It checks whether the index
already exists and only creates it if missing.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "coboledu")

    if not api_key:
        print("ERROR: PINECONE_API_KEY not set in .env")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]

    if index_name in existing:
        print(f"Index '{index_name}' already exists — skipping creation.")
        idx = pc.Index(index_name)
        stats = idx.describe_index_stats()
        print(f"  Vectors:    {stats.total_vector_count}")
        print(f"  Dimension:  {stats.dimension}")
        print(f"  Namespaces: {list(stats.namespaces.keys()) or ['(default)']}")
        return

    print(f"Creating index '{index_name}' …")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created successfully.")


if __name__ == "__main__":
    main()
