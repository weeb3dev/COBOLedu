"""Ingestion pipeline orchestrator — discovery, extraction, chunking, embedding, and vector upsert."""

from __future__ import annotations

import logging
import time
from collections import Counter

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode

from src.chunking.orchestrator import chunk_all_files
from src.config import GNUCOBOL_SOURCE_DIR
from src.ingestion.discover import discover_files
from src.ingestion.preprocess import extract_cobol_from_at, normalize_source
from src.retrieval.embeddings import get_embed_model
from src.retrieval.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingestion() -> None:
    t0 = time.perf_counter()

    # --- Phase 1a: discover files ---
    logger.info("Scanning %s …", GNUCOBOL_SOURCE_DIR)
    files = discover_files(GNUCOBOL_SOURCE_DIR)

    # --- Phase 1b: extract COBOL from .at test files ---
    at_files = [f for f in files if f.language == "COBOL_TEST"]
    logger.info("Extracting COBOL programs from %d .at files …", len(at_files))

    all_programs = []
    for f in at_files:
        programs = extract_cobol_from_at(f.path)
        all_programs.extend(programs)

    # --- Phase 1c: normalize all source content (dry-run preview) ---
    source_files = [f for f in files if f.language in ("C", "COBOL", "YACC", "LEX")]
    logger.info(
        "Discovered %d direct source files + %d extracted COBOL programs",
        len(source_files),
        len(all_programs),
    )

    _print_extraction_summary(all_programs)

    elapsed_p1 = time.perf_counter() - t0
    logger.info("Phase 1 ingestion completed in %.2fs", elapsed_p1)

    # --- Phase 2: chunking ---
    t1 = time.perf_counter()
    logger.info("Chunking all files …")
    chunks = chunk_all_files(files, all_programs)
    elapsed_p2 = time.perf_counter() - t1
    logger.info("Phase 2 chunking completed: %d chunks in %.2fs", len(chunks), elapsed_p2)

    # --- Phase 3: embed + upsert into Pinecone ---
    t2 = time.perf_counter()
    logger.info("Embedding %d chunks and upserting into Pinecone …", len(chunks))

    Settings.embed_model = get_embed_model()
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = _embed_with_retry(chunks, storage_context)

    elapsed_p3 = time.perf_counter() - t2
    logger.info("Phase 3 embedding completed: %d vectors in %.2fs", len(chunks), elapsed_p3)

    total = time.perf_counter() - t0
    logger.info("Full pipeline completed in %.2fs", total)

    return files, all_programs, chunks, index


EMBED_BATCH = 512
MAX_RETRIES = 6
BASE_DELAY = 15.0


def _embed_with_retry(
    chunks: list[TextNode],
    storage_context: StorageContext,
) -> VectorStoreIndex:
    """Embed and upsert in batches with exponential backoff on rate-limit errors."""
    total = len(chunks)
    offset = 0

    while offset < total:
        batch = chunks[offset : offset + EMBED_BATCH]
        for attempt in range(MAX_RETRIES):
            try:
                if offset == 0:
                    index = VectorStoreIndex(
                        nodes=batch,
                        storage_context=storage_context,
                        show_progress=True,
                    )
                else:
                    index.insert_nodes(batch, show_progress=True)
                break
            except Exception as exc:
                if "rate" not in str(exc).lower() and "429" not in str(exc):
                    raise
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited on batch %d-%d (attempt %d/%d), sleeping %.0fs …",
                    offset, offset + len(batch), attempt + 1, MAX_RETRIES, delay,
                )
                time.sleep(delay)
        else:
            raise RuntimeError(
                f"Failed to embed batch at offset {offset} after {MAX_RETRIES} retries"
            )
        offset += EMBED_BATCH
        logger.info("Embedded %d / %d chunks", min(offset, total), total)
        if offset < total:
            time.sleep(2)

    return index


def _print_extraction_summary(programs):
    ext_counts = Counter(p.extension.lower() for p in programs)
    total_lines = sum(p.cobol_source.count("\n") + 1 for p in programs)

    print(f"\n{'=' * 50}")
    print("COBOL Extraction Summary")
    print(f"{'=' * 50}")
    print(f"Total extracted programs:  {len(programs)}")
    print(f"Total extracted lines:     {total_lines:,}")
    print(f"{'-' * 50}")
    for ext, count in ext_counts.most_common():
        print(f"  {ext:<10} {count:>5} programs")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    run_ingestion()
