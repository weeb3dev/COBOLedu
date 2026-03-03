"""Generic fixed-size chunker with line-boundary awareness and overlap."""

from __future__ import annotations

from typing import List

from llama_index.core.schema import TextNode


def chunk_generic(
    source: str,
    file_path: str,
    language: str = "unknown",
    chunk_size: int = 1600,
    overlap: int = 200,
) -> List[TextNode]:
    """Split source into fixed-size chunks at line boundaries with overlap."""
    if not source.strip():
        return []

    lines = source.split("\n")
    nodes: List[TextNode] = []
    chunk_start = 0

    while chunk_start < len(lines):
        char_count = 0
        chunk_end = chunk_start
        while chunk_end < len(lines) and char_count + len(lines[chunk_end]) + 1 <= chunk_size:
            char_count += len(lines[chunk_end]) + 1
            chunk_end += 1

        if chunk_end == chunk_start:
            chunk_end = chunk_start + 1

        chunk_text = "\n".join(lines[chunk_start:chunk_end])
        ls = chunk_start + 1
        le = chunk_end
        nodes.append(
            TextNode(
                text=chunk_text,
                id_=f"{file_path}:{ls}-{le}",
                metadata={
                    "file_path": file_path,
                    "line_start": ls,
                    "line_end": le,
                    "language": language,
                    "chunk_type": "generic",
                },
            )
        )

        overlap_chars = 0
        overlap_start = chunk_end
        while overlap_start > chunk_start and overlap_chars < overlap:
            overlap_start -= 1
            overlap_chars += len(lines[overlap_start]) + 1

        chunk_start = max(overlap_start, chunk_start + 1) if chunk_end < len(lines) else len(lines)

    return nodes
