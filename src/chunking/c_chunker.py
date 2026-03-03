"""C source-aware chunker — splits at function definitions, structs, and typedefs."""

from __future__ import annotations

import re
from typing import List

from llama_index.core.schema import TextNode

MAX_CHUNK_CHARS = 2000
SUB_CHUNK_OVERLAP = 200

FUNC_DEF_RE = re.compile(
    r"""
    ^                                   # start of line
    (?!(?:if|else|for|while|switch|return|case|typedef|struct|enum|union)\b)
    ([\w][\w\s*]*?)                     # return type (non-greedy)
    \s+
    (\*?\s*\w+)                         # function name (optionally pointer)
    \s*
    \(([^)]*)\)                         # parameter list
    \s*\{                               # opening brace
    """,
    re.MULTILINE | re.VERBOSE,
)

STRUCT_RE = re.compile(
    r"^\s*(?:typedef\s+)?(?:struct|enum|union)\s+(\w+)?\s*\{",
    re.MULTILINE,
)

TYPEDEF_SIMPLE_RE = re.compile(
    r"^typedef\s+.+?;\s*$",
    re.MULTILINE,
)

BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

PROTOTYPE_RE = re.compile(
    r"""
    ^
    (?:extern\s+)?
    [\w][\w\s*]*?          # return type
    \s+\*?\s*\w+           # function name
    \s*\([^)]*\)           # params
    \s*;                   # semicolon — prototype, not definition
    """,
    re.MULTILINE | re.VERBOSE,
)


def chunk_c_file(source: str, file_path: str) -> List[TextNode]:
    """Split C/YACC/LEX source into function-level (or struct-level) chunks."""
    is_header = file_path.endswith(".h")
    lines = source.split("\n")

    if is_header:
        return _chunk_header(lines, file_path)

    return _chunk_source(lines, file_path)


def _chunk_source(lines: List[str], file_path: str) -> List[TextNode]:
    """Chunk a .c / .y / .l file at function boundaries."""
    spans = _find_function_spans(lines)

    if not spans:
        text = "\n".join(lines)
        if not text.strip():
            return []
        return _make_nodes(text, file_path, 0, "function", "unknown")

    nodes: List[TextNode] = []

    if spans[0][1] > 0:
        preamble = "\n".join(lines[: spans[0][1]])
        if preamble.strip():
            nodes.extend(_make_nodes(preamble, file_path, 0, "preamble", "preamble"))

    for func_name, start, end in spans:
        comment_start = _find_preceding_comment(lines, start)
        chunk_text = "\n".join(lines[comment_start:end])
        nodes.extend(_make_nodes(chunk_text, file_path, comment_start, "function", func_name))

    return nodes


def _find_function_spans(lines: List[str]) -> List[tuple[str, int, int]]:
    """Return (func_name, start_line, end_line_exclusive) for each function definition."""
    full_text = "\n".join(lines)
    spans: List[tuple[str, int, int]] = []

    for m in FUNC_DEF_RE.finditer(full_text):
        func_name = m.group(2).strip().lstrip("*").strip()
        brace_pos = m.end() - 1
        start_line = full_text[:m.start()].count("\n")
        close_pos = _match_brace(full_text, brace_pos)
        if close_pos == -1:
            end_line = len(lines)
        else:
            end_line = full_text[:close_pos + 1].count("\n") + 1
        spans.append((func_name, start_line, end_line))

    spans.sort(key=lambda s: s[1])
    return _merge_overlapping(spans)


def _merge_overlapping(spans: List[tuple[str, int, int]]) -> List[tuple[str, int, int]]:
    if not spans:
        return spans
    merged: List[tuple[str, int, int]] = [spans[0]]
    for name, start, end in spans[1:]:
        prev_name, prev_start, prev_end = merged[-1]
        if start < prev_end:
            merged[-1] = (prev_name, prev_start, max(prev_end, end))
        else:
            merged.append((name, start, end))
    return merged


def _match_brace(text: str, open_pos: int) -> int:
    """Find closing } matching the { at open_pos, respecting nesting and strings."""
    depth = 1
    i = open_pos + 1
    length = len(text)
    in_string = False
    string_char = ""
    in_line_comment = False
    in_block_comment = False

    while i < length:
        ch = text[i]

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and i + 1 < length and text[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == "\\" and i + 1 < length:
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue

        if ch == "/" and i + 1 < length:
            nxt = text[i + 1]
            if nxt == "/":
                in_line_comment = True
                i += 2
                continue
            if nxt == "*":
                in_block_comment = True
                i += 2
                continue

        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


def _find_preceding_comment(lines: List[str], func_start: int) -> int:
    """Walk backwards from func_start to include any immediately preceding comment block."""
    i = func_start - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.endswith("*/") or stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
            i -= 1
        else:
            break
    return i + 1


def _chunk_header(lines: List[str], file_path: str) -> List[TextNode]:
    """Chunk a header file — individual structs/enums/typedefs, batched prototypes."""
    full_text = "\n".join(lines)
    nodes: List[TextNode] = []
    consumed: set[int] = set()

    for m in STRUCT_RE.finditer(full_text):
        brace_pos = m.end() - 1
        close_pos = _match_brace(full_text, brace_pos)
        if close_pos == -1:
            continue

        semi_pos = full_text.find(";", close_pos)
        if semi_pos == -1:
            semi_pos = close_pos

        start_line = full_text[:m.start()].count("\n")
        comment_start = _find_preceding_comment(lines, start_line)
        end_line = full_text[:semi_pos + 1].count("\n") + 1

        chunk_text = "\n".join(lines[comment_start:end_line])
        name = m.group(1) or "anonymous"

        kind = "struct"
        if "enum" in m.group(0):
            kind = "enum"
        elif "union" in m.group(0):
            kind = "struct"

        nodes.extend(_make_nodes(chunk_text, file_path, comment_start, kind, name))
        for ln in range(comment_start, end_line):
            consumed.add(ln)

    proto_batch: List[str] = []
    proto_start: int | None = None
    proto_end: int = 0

    for i, line in enumerate(lines):
        if i in consumed:
            continue
        if PROTOTYPE_RE.match(line):
            if proto_start is None:
                proto_start = i
            proto_batch.append(line)
            proto_end = i
        else:
            if proto_batch and len(proto_batch) >= 3:
                text = "\n".join(proto_batch)
                nodes.extend(_make_nodes(text, file_path, proto_start or 0, "prototype_group", "prototypes"))
                proto_batch = []
                proto_start = None

    if proto_batch and len(proto_batch) >= 3:
        text = "\n".join(proto_batch)
        nodes.extend(_make_nodes(text, file_path, proto_start or 0, "prototype_group", "prototypes"))

    if not nodes:
        text = "\n".join(lines)
        if text.strip():
            nodes.extend(_make_nodes(text, file_path, 0, "header", file_path.rsplit("/", 1)[-1]))

    return nodes


def _make_nodes(
    text: str, file_path: str, line_start: int, chunk_type: str, func_name: str,
) -> List[TextNode]:
    if not text.strip():
        return []

    line_count = text.count("\n") + 1

    if len(text) <= MAX_CHUNK_CHARS:
        return [
            _build_node(text, file_path, line_start, line_start + line_count - 1, chunk_type, func_name)
        ]

    return _sub_chunk(text, file_path, line_start, chunk_type, func_name)


def _sub_chunk(
    text: str, file_path: str, line_start: int, chunk_type: str, func_name: str,
) -> List[TextNode]:
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
                chunk_type, func_name,
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
    text: str, file_path: str, line_start: int, line_end: int,
    chunk_type: str, func_name: str,
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
            "language": "C",
            "chunk_type": chunk_type,
            "function_name": func_name,
        },
    )
