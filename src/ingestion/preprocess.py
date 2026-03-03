"""COBOL extraction from .at test files and source normalization."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

COBOL_EXTENSIONS = {".cob", ".cbl", ".cpy", ".copy", ".inc", ".COB", ".CPY"}

AT_DATA_START_RE = re.compile(r"^AT_DATA\(\[([^\]]+)\],\s*\[", re.MULTILINE)


@dataclass
class ExtractedProgram:
    source_file: str
    program_name: str
    cobol_source: str
    line_offset: int
    extension: str


def _find_matching_close(text: str, start: int) -> int:
    """Find the position of `])` that closes the AT_DATA content bracket.

    Starts scanning from `start` (which should point just past the opening `[`).
    Tracks bracket depth so nested `[`/`]` pairs inside COBOL content are handled.
    """
    depth = 1
    i = start
    length = len(text)

    while i < length:
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1

    return -1


def extract_cobol_from_at(file_path: str | Path) -> List[ExtractedProgram]:
    """Parse an .at file and extract embedded COBOL programs from AT_DATA blocks."""
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []

    programs: List[ExtractedProgram] = []
    line_offsets = _build_line_offsets(content)

    for match in AT_DATA_START_RE.finditer(content):
        filename = match.group(1)
        ext = Path(filename).suffix

        if ext.lower() not in {e.lower() for e in COBOL_EXTENSIONS}:
            continue

        bracket_open_end = match.end()
        close_pos = _find_matching_close(content, bracket_open_end)
        if close_pos == -1:
            logger.warning(
                "Unmatched bracket in %s at offset %d for %s",
                path,
                match.start(),
                filename,
            )
            continue

        cobol_source = content[bracket_open_end:close_pos]

        # Strip leading/trailing blank line that the macro format introduces
        if cobol_source.startswith("\n"):
            cobol_source = cobol_source[1:]
        if cobol_source.endswith("\n"):
            cobol_source = cobol_source[:-1]

        line_offset = _offset_to_line(line_offsets, bracket_open_end)

        programs.append(
            ExtractedProgram(
                source_file=str(path),
                program_name=filename,
                cobol_source=cobol_source,
                line_offset=line_offset,
                extension=ext,
            )
        )

    logger.info("Extracted %d COBOL programs from %s", len(programs), path.name)
    return programs


def _build_line_offsets(text: str) -> List[int]:
    """Return a list where index i is the character offset of line i+1."""
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def _offset_to_line(offsets: List[int], char_offset: int) -> int:
    """Convert a character offset to a 1-based line number."""
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= char_offset:
            lo = mid + 1
        else:
            hi = mid - 1
    return lo


def normalize_source(content: str, language: str) -> str:
    """Normalize encoding, line endings, and whitespace for a source string."""
    if isinstance(content, bytes):
        for enc in ("utf-8", "latin-1"):
            try:
                content = content.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            content = content.decode("utf-8", errors="replace")

    content = content.replace("\r\n", "\n").replace("\r", "\n")

    lines = content.split("\n")
    lines = [line.rstrip() for line in lines]

    if language in ("COBOL", "COBOL_TEST"):
        # Preserve fixed-format column structure — don't truncate at col 72
        # since free-format COBOL may be present too. Just clean trailing ws.
        pass

    return "\n".join(lines)


def read_file_safe(path: Path) -> str:
    """Read a file with encoding fallback."""
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")
