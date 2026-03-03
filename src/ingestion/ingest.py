"""Ingestion pipeline orchestrator — ties discovery, extraction, and (later) chunking + embedding."""

from __future__ import annotations

import logging
import time
from collections import Counter

from src.chunking.orchestrator import chunk_all_files
from src.config import GNUCOBOL_SOURCE_DIR
from src.ingestion.discover import discover_files
from src.ingestion.preprocess import extract_cobol_from_at, normalize_source

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

    # --- Phase 3 stub (embedding) ---
    # TODO: embed and upsert into Pinecone

    return files, all_programs, chunks


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
