"""File discovery module for the GnuCOBOL source tree."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.config import GNUCOBOL_SOURCE_DIR

logger = logging.getLogger(__name__)

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".c": "C",
    ".h": "C",
    ".cob": "COBOL",
    ".cbl": "COBOL",
    ".cpy": "COBOL",
    ".at": "COBOL_TEST",
    ".conf": "CONFIG",
    ".words": "CONFIG",
    ".y": "YACC",
    ".l": "LEX",
}

SKIP_DIRS: set[str] = {
    "_build",
    "autom4te.cache",
    ".git",
    "__pycache__",
    "build_aux",
    "build_windows",
    "po",
    ".github",
}

SKIP_EXTENSIONS: set[str] = {
    ".o",
    ".lo",
    ".la",
    ".pdf",
    ".png",
    ".jpg",
    ".vcxproj",
    ".vcproj",
    ".sln",
    ".filters",
    ".user",
    ".sample",
}


@dataclass
class FileInfo:
    path: Path
    relative_path: str
    language: str
    extension: str
    size_bytes: int
    line_count: int


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("rb"))
    except (OSError, UnicodeDecodeError):
        return 0


def discover_files(root_dir: str | Path | None = None) -> List[FileInfo]:
    """Recursively scan the GnuCOBOL source tree and return classified FileInfo objects."""
    root = Path(root_dir) if root_dir else GNUCOBOL_SOURCE_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {root}")

    files: List[FileInfo] = []

    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue

        if any(skip in item.parts for skip in SKIP_DIRS):
            continue

        ext = item.suffix.lower()
        if ext in SKIP_EXTENSIONS:
            continue

        language = EXTENSION_TO_LANGUAGE.get(ext)
        if language is None:
            continue

        files.append(
            FileInfo(
                path=item,
                relative_path=str(item.relative_to(root)),
                language=language,
                extension=ext,
                size_bytes=item.stat().st_size,
                line_count=_count_lines(item),
            )
        )

    _log_summary(files)
    return files


def _log_summary(files: List[FileInfo]) -> None:
    lang_counts: Counter[str] = Counter(f.language for f in files)
    total_lines = sum(f.line_count for f in files)
    total_bytes = sum(f.size_bytes for f in files)

    print(f"\n{'=' * 50}")
    print(f"File Discovery Summary")
    print(f"{'=' * 50}")
    print(f"Total files:  {len(files)}")
    print(f"Total lines:  {total_lines:,}")
    print(f"Total size:   {total_bytes / 1024:.1f} KB")
    print(f"{'-' * 50}")
    for lang, count in lang_counts.most_common():
        lang_lines = sum(f.line_count for f in files if f.language == lang)
        print(f"  {lang:<15} {count:>4} files  {lang_lines:>8,} lines")
    print(f"{'=' * 50}\n")
