"""Chunking orchestrator — routes files to the appropriate chunker and collects TextNodes."""

from __future__ import annotations

import logging
from collections import Counter
from typing import List

from llama_index.core.schema import TextNode

from src.chunking.c_chunker import chunk_c_file
from src.chunking.cobol_chunker import chunk_cobol
from src.chunking.fallback import chunk_generic
from src.ingestion.discover import FileInfo
from src.ingestion.preprocess import ExtractedProgram, normalize_source, read_file_safe

logger = logging.getLogger(__name__)

C_LIKE_LANGUAGES = frozenset({"C", "YACC", "LEX"})


def chunk_all_files(
    files: List[FileInfo],
    extracted_cobol: List[ExtractedProgram],
) -> List[TextNode]:
    """Route every discovered file and extracted COBOL program to its chunker."""
    all_nodes: List[TextNode] = []

    for fi in files:
        if fi.language == "COBOL_TEST":
            continue

        try:
            content = read_file_safe(fi.path)
            content = normalize_source(content, fi.language)
        except Exception:
            logger.warning("Failed to read %s, skipping", fi.relative_path, exc_info=True)
            continue

        if not content.strip():
            continue

        try:
            if fi.language == "COBOL":
                nodes = chunk_cobol(content, fi.relative_path)
            elif fi.language in C_LIKE_LANGUAGES:
                nodes = chunk_c_file(content, fi.relative_path)
            else:
                nodes = chunk_generic(content, fi.relative_path, language=fi.language)

            all_nodes.extend(nodes)
        except Exception:
            logger.warning("Failed to chunk %s, skipping", fi.relative_path, exc_info=True)

    for ep in extracted_cobol:
        identifier = f"{ep.source_file}::{ep.program_name}"
        try:
            source = normalize_source(ep.cobol_source, "COBOL")
            nodes = chunk_cobol(source, identifier, line_offset=ep.line_offset)
            all_nodes.extend(nodes)
        except Exception:
            logger.warning("Failed to chunk extracted program %s, skipping", identifier, exc_info=True)

    _print_stats(all_nodes)
    return all_nodes


def _print_stats(nodes: List[TextNode]) -> None:
    lang_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    sizes: List[int] = []

    for n in nodes:
        lang_counts[n.metadata.get("language", "?")] += 1
        type_counts[n.metadata.get("chunk_type", "?")] += 1
        sizes.append(len(n.text))

    avg_size = sum(sizes) / len(sizes) if sizes else 0

    print(f"\n{'=' * 50}")
    print("Chunking Summary")
    print(f"{'=' * 50}")
    print(f"Total chunks:   {len(nodes)}")
    print(f"Avg chunk size: {avg_size:.0f} chars (~{avg_size / 4:.0f} tokens)")
    if sizes:
        print(f"Min / Max:      {min(sizes)} / {max(sizes)} chars")
    print(f"{'-' * 50}")
    print("By language:")
    for lang, count in lang_counts.most_common():
        print(f"  {lang:<15} {count:>5} chunks")
    print(f"{'-' * 50}")
    print("By chunk type:")
    for ctype, count in type_counts.most_common():
        print(f"  {ctype:<20} {count:>5} chunks")
    print(f"{'=' * 50}\n")
