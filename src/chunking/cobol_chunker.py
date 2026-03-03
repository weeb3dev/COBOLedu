"""COBOL syntax-aware chunker — splits at divisions, sections, paragraphs, and data items."""

from __future__ import annotations

import re
from typing import List

from llama_index.core.schema import TextNode

MAX_CHUNK_CHARS = 2000
SUB_CHUNK_OVERLAP = 200

DIVISION_RE = re.compile(
    r"^\s{0,11}(IDENTIFICATION|ENVIRONMENT|DATA|PROCEDURE)\s+DIVISION",
    re.IGNORECASE | re.MULTILINE,
)

SECTION_RE = re.compile(
    r"^\s{0,11}([\w-]+)\s+SECTION\s*\.",
    re.IGNORECASE | re.MULTILINE,
)

LEVEL_01_RE = re.compile(
    r"^\s{6}\s?01\s+",
    re.MULTILINE,
)

COBOL_VERBS = frozenset(
    w.upper()
    for w in [
        "MOVE", "PERFORM", "DISPLAY", "COMPUTE", "IF", "EVALUATE", "READ",
        "WRITE", "OPEN", "CLOSE", "ACCEPT", "ADD", "SUBTRACT", "MULTIPLY",
        "DIVIDE", "STRING", "STOP", "GO", "SET", "CALL", "EXIT", "INITIALIZE",
        "INSPECT", "SEARCH", "SORT", "MERGE", "RELEASE", "RETURN", "REWRITE",
        "DELETE", "START", "CONTINUE", "GOBACK", "ALTER", "ENABLE", "DISABLE",
        "SEND", "RECEIVE", "GENERATE", "INITIATE", "TERMINATE", "UNSTRING",
        "WHEN", "ELSE", "END-IF", "END-EVALUATE", "END-PERFORM", "END-READ",
        "END-WRITE", "END-SEARCH", "END-CALL", "END-COMPUTE", "END-STRING",
        "END-UNSTRING", "END-MULTIPLY", "END-DIVIDE", "END-ADD", "END-SUBTRACT",
        "NOT", "THEN", "ALSO", "OTHER", "THRU", "THROUGH", "VARYING",
        "UNTIL", "WITH", "INTO", "FROM", "GIVING", "USING", "BY",
    ]
)

PARAGRAPH_RE = re.compile(
    r"^(\s{0,3}\s{4})([\w][\w-]*)\s*\.\s*$",
    re.MULTILINE,
)


def chunk_cobol(source: str, file_path: str, line_offset: int = 0) -> List[TextNode]:
    """Split COBOL source into semantically meaningful chunks."""
    lines = source.split("\n")
    divisions = _split_divisions(lines)
    nodes: List[TextNode] = []

    for div_name, div_lines, div_start in divisions:
        div_text = "\n".join(div_lines)

        if div_name in ("IDENTIFICATION", "ENVIRONMENT"):
            nodes.extend(
                _make_nodes(div_text, file_path, div_start + line_offset, "division", div_name)
            )
        elif div_name == "DATA":
            nodes.extend(_chunk_data_division(div_lines, file_path, div_start + line_offset, div_name))
        elif div_name == "PROCEDURE":
            nodes.extend(_chunk_procedure_division(div_lines, file_path, div_start + line_offset, div_name))
        else:
            nodes.extend(
                _make_nodes(div_text, file_path, div_start + line_offset, "division", div_name)
            )

    if not divisions:
        nodes.extend(_make_nodes(source, file_path, line_offset, "division", "UNKNOWN"))

    return nodes


def _split_divisions(lines: List[str]) -> List[tuple[str, List[str], int]]:
    """Return list of (division_name, lines, start_line_0based)."""
    boundaries: List[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = DIVISION_RE.match(line)
        if m:
            boundaries.append((m.group(1).upper(), i))

    if not boundaries:
        return []

    result: List[tuple[str, List[str], int]] = []
    for idx, (name, start) in enumerate(boundaries):
        end = boundaries[idx + 1][1] if idx + 1 < len(boundaries) else len(lines)
        result.append((name, lines[start:end], start))
    return result


def _chunk_data_division(
    lines: List[str], file_path: str, base_offset: int, division_name: str,
) -> List[TextNode]:
    """Split DATA DIVISION at 01-level items, keeping subordinates grouped."""
    boundaries: List[int] = []
    for i, line in enumerate(lines):
        if LEVEL_01_RE.match(line):
            boundaries.append(i)

    if not boundaries:
        text = "\n".join(lines)
        return _make_nodes(text, file_path, base_offset, "division", division_name)

    nodes: List[TextNode] = []

    if boundaries[0] > 0:
        preamble = "\n".join(lines[: boundaries[0]])
        if preamble.strip():
            nodes.extend(
                _make_nodes(preamble, file_path, base_offset, "section", division_name)
            )

    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        chunk_lines = lines[start:end]
        text = "\n".join(chunk_lines)

        name_match = re.match(r"\s*01\s+([\w-]+)", lines[start])
        item_name = name_match.group(1) if name_match else "FILLER"

        nodes.extend(
            _make_nodes(
                text, file_path, base_offset + start, "data_item", division_name,
                paragraph_name=item_name,
            )
        )

    return nodes


def _chunk_procedure_division(
    lines: List[str], file_path: str, base_offset: int, division_name: str,
) -> List[TextNode]:
    """Split PROCEDURE DIVISION at paragraph boundaries."""
    boundaries: List[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = PARAGRAPH_RE.match(line)
        if m:
            name = m.group(2).upper()
            if name not in COBOL_VERBS and not name.startswith("END-"):
                boundaries.append((m.group(2), i))

    section_matches: List[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = SECTION_RE.match(line)
        if m:
            section_matches.append((m.group(1), i))

    all_boundaries = sorted(
        [(n, i, "paragraph") for n, i in boundaries]
        + [(n, i, "section") for n, i in section_matches],
        key=lambda x: x[1],
    )

    if not all_boundaries:
        text = "\n".join(lines)
        return _make_nodes(text, file_path, base_offset, "division", division_name)

    nodes: List[TextNode] = []

    if all_boundaries[0][1] > 0:
        preamble = "\n".join(lines[: all_boundaries[0][1]])
        if preamble.strip():
            nodes.extend(
                _make_nodes(preamble, file_path, base_offset, "division", division_name)
            )

    for idx, (name, start, kind) in enumerate(all_boundaries):
        end = all_boundaries[idx + 1][1] if idx + 1 < len(all_boundaries) else len(lines)
        chunk_lines = lines[start:end]
        text = "\n".join(chunk_lines)

        nodes.extend(
            _make_nodes(
                text, file_path, base_offset + start, kind, division_name,
                paragraph_name=name,
            )
        )

    return nodes


def _make_nodes(
    text: str,
    file_path: str,
    line_start: int,
    chunk_type: str,
    division_name: str,
    section_name: str = "",
    paragraph_name: str = "",
) -> List[TextNode]:
    """Create one or more TextNodes, sub-chunking if text exceeds the limit."""
    if not text.strip():
        return []

    line_count = text.count("\n") + 1

    if len(text) <= MAX_CHUNK_CHARS:
        return [
            _build_node(
                text, file_path, line_start, line_start + line_count - 1,
                chunk_type, division_name, section_name, paragraph_name,
            )
        ]

    return _sub_chunk(
        text, file_path, line_start, chunk_type, division_name,
        section_name, paragraph_name,
    )


def _sub_chunk(
    text: str,
    file_path: str,
    line_start: int,
    chunk_type: str,
    division_name: str,
    section_name: str,
    paragraph_name: str,
) -> List[TextNode]:
    """Split oversized text into overlapping sub-chunks at line boundaries."""
    lines = text.split("\n")
    nodes: List[TextNode] = []
    chunk_start = 0

    while chunk_start < len(lines):
        char_count = 0
        chunk_end = chunk_start
        while chunk_end < len(lines) and char_count + len(lines[chunk_end]) + 1 <= MAX_CHUNK_CHARS:
            char_count += len(lines[chunk_end]) + 1
            chunk_end += 1

        if chunk_end == chunk_start:
            chunk_end = chunk_start + 1

        chunk_text = "\n".join(lines[chunk_start:chunk_end])
        nodes.append(
            _build_node(
                chunk_text, file_path,
                line_start + chunk_start, line_start + chunk_end - 1,
                chunk_type, division_name, section_name, paragraph_name,
            )
        )

        overlap_chars = 0
        overlap_start = chunk_end
        while overlap_start > chunk_start and overlap_chars < SUB_CHUNK_OVERLAP:
            overlap_start -= 1
            overlap_chars += len(lines[overlap_start]) + 1

        chunk_start = max(overlap_start, chunk_start + 1) if chunk_end < len(lines) else len(lines)

    return nodes


def _build_node(
    text: str,
    file_path: str,
    line_start: int,
    line_end: int,
    chunk_type: str,
    division_name: str,
    section_name: str = "",
    paragraph_name: str = "",
) -> TextNode:
    ls = line_start + 1
    le = line_end + 1
    return TextNode(
        text=text,
        id_=f"{file_path}:{ls}-{le}",
        metadata={
            "file_path": file_path,
            "line_start": ls,
            "line_end": le,
            "language": "COBOL",
            "chunk_type": chunk_type,
            "division_name": division_name,
            "section_name": section_name,
            "paragraph_name": paragraph_name,
        },
    )
