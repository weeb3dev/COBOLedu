"""Advanced code understanding features for COBOLedu.

Each feature reuses the existing Pinecone retriever + Claude generation
and is decorated with @observe() for automatic Langfuse tracing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import anthropic
from langfuse import observe
from llama_index.core import Settings

from src.config import ANTHROPIC_API_KEY, TOP_K
from src.retrieval.embeddings import get_embed_model
from src.retrieval.query import (
    SourceInfo,
    _detect_language_filter,
    _extract_sources,
    _merged_retrieve,
    _scrub_answer_paths,
    normalize_path,
    preprocess_query,
    rerank_nodes,
)
from src.retrieval.vector_store import get_index, get_vector_store

logger = logging.getLogger(__name__)

_CLAUDE_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 4096


# ── Shared helpers ────────────────────────────────────────────────────────

def _get_index():
    Settings.embed_model = get_embed_model()
    vs = get_vector_store()
    return get_index(vs)


def _retrieve(query: str, top_k: int = TOP_K) -> tuple[list, list[SourceInfo]]:
    """Over-retrieve from Pinecone, rerank with Voyage, return top-k."""
    expanded = preprocess_query(query)
    lang = _detect_language_filter(query)
    idx = _get_index()
    nodes = _merged_retrieve(idx, expanded, lang)
    nodes = rerank_nodes(query, nodes, top_k=top_k)
    return nodes, _extract_sources(nodes)


def _llm_generate(system: str, user: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    cached_system = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    resp = client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=_MAX_TOKENS,
        system=cached_system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


async def _aretrieve(query: str, top_k: int = TOP_K) -> tuple[list, list[SourceInfo]]:
    """Async wrapper around _retrieve() for use in streaming generators."""
    return await asyncio.to_thread(_retrieve, query, top_k)


async def _llm_stream(
    system: str, user: str,
) -> AsyncGenerator[str, None]:
    """Async generator that streams text chunks from Claude."""
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    cached_system = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    async with client.messages.stream(
        model=_CLAUDE_MODEL,
        max_tokens=_MAX_TOKENS,
        system=cached_system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


def _context_from_nodes(nodes) -> str:
    parts = []
    for n in nodes:
        meta = n.metadata or {}
        fp = normalize_path(meta.get("file_path", "unknown"))
        ls = meta.get("line_start", "?")
        le = meta.get("line_end", "?")
        parts.append(f"--- {fp}:{ls}-{le} ---\n{n.get_content()}")
    return _scrub_answer_paths("\n\n".join(parts))


# ── 1. Code Explanation ───────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    explanation: str
    sources: list[SourceInfo]


@observe(name="coboledu-explain")
def explain_code(query: str) -> ExplanationResult:
    """Retrieve relevant code and generate a structured explanation."""
    nodes, sources = _retrieve(query)
    context = _context_from_nodes(nodes)

    system = (
        "You are a code analysis assistant for the GnuCOBOL project. "
        "When given code snippets, produce a structured explanation with these sections:\n"
        "## Purpose\n## Inputs / Outputs\n## Key Logic\n## Side Effects\n## Complexity Notes\n\n"
        "Always cite file paths and line numbers (format: filename:line_start-line_end)."
    )
    user = f"Code context:\n{context}\n\nExplain: {query}"

    answer = _llm_generate(system, user)
    return ExplanationResult(explanation=answer, sources=sources)


@observe(name="coboledu-stream-explain")
async def stream_explain(
    query: str,
) -> AsyncGenerator[tuple[str, str | list[SourceInfo]], None]:
    """Stream explanation tokens then emit sources."""
    try:
        nodes, sources = await _aretrieve(query)
        context = _context_from_nodes(nodes)

        system = (
            "You are a code analysis assistant for the GnuCOBOL project. "
            "When given code snippets, produce a structured explanation with these sections:\n"
            "## Purpose\n## Inputs / Outputs\n## Key Logic\n## Side Effects\n## Complexity Notes\n\n"
            "Always cite file paths and line numbers (format: filename:line_start-line_end)."
        )
        user = f"Code context:\n{context}\n\nExplain: {query}"

        async for text in _llm_stream(system, user):
            yield ("token", text)
        yield ("sources", sources)
    except Exception as exc:
        logger.exception("stream_explain failed")
        yield ("error", str(exc))


# ── 2. Dependency Mapping ─────────────────────────────────────────────────

@dataclass
class DependencyResult:
    target: str
    callers: list[str]
    callees: list[str]
    analysis: str
    sources: list[SourceInfo]


@observe(name="coboledu-dependencies")
def find_dependencies(name: str, direction: str = "both") -> DependencyResult:
    """Find what calls *name* and what *name* calls."""
    search_query = f"PERFORM {name} CALL {name} function {name}"
    nodes, sources = _retrieve(search_query, top_k=10)
    context = _context_from_nodes(nodes)

    system = (
        "You are a dependency analysis assistant for the GnuCOBOL project.\n"
        "Given code snippets, identify:\n"
        "1. CALLERS — code that calls/PERFORMs the target\n"
        "2. CALLEES — code that the target calls/PERFORMs\n\n"
        "For COBOL: look for PERFORM <name>, CALL <name>.\n"
        "For C: look for function_name(...) calls.\n\n"
        "Return your analysis in this format:\n"
        "## Callers\n- list each with file:line citation\n"
        "## Callees\n- list each with file:line citation\n"
        "## Summary\n- brief dependency overview\n\n"
        "If no callers or callees are found, say so explicitly."
    )
    user = f"Code context:\n{context}\n\nAnalyse dependencies for: {name}"

    answer = _llm_generate(system, user)

    callers = _extract_names(answer, "Callers")
    callees = _extract_names(answer, "Callees")

    return DependencyResult(
        target=name,
        callers=callers,
        callees=callees,
        analysis=answer,
        sources=sources,
    )


def _extract_names(text: str, section: str) -> list[str]:
    """Best-effort extraction of names from a markdown list under a heading."""
    pattern = rf"##\s*{section}\s*\n((?:[-*]\s*.+\n?)+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return []
    names = []
    for line in match.group(1).strip().split("\n"):
        line = line.strip().lstrip("-* ").strip()
        if line and "none" not in line.lower() and "no " not in line.lower():
            token = line.split(":")[0].split("(")[0].split(" ")[0].strip("`")
            if token:
                names.append(token)
    return names


@observe(name="coboledu-stream-dependencies")
async def stream_dependencies(
    name: str, direction: str = "both",
) -> AsyncGenerator[tuple[str, str | list[SourceInfo] | dict], None]:
    """Stream dependency analysis tokens, then emit metadata + sources."""
    try:
        search_query = f"PERFORM {name} CALL {name} function {name}"
        nodes, sources = await _aretrieve(search_query, top_k=10)
        context = _context_from_nodes(nodes)

        system = (
            "You are a dependency analysis assistant for the GnuCOBOL project.\n"
            "Given code snippets, identify:\n"
            "1. CALLERS — code that calls/PERFORMs the target\n"
            "2. CALLEES — code that the target calls/PERFORMs\n\n"
            "For COBOL: look for PERFORM <name>, CALL <name>.\n"
            "For C: look for function_name(...) calls.\n\n"
            "Return your analysis in this format:\n"
            "## Callers\n- list each with file:line citation\n"
            "## Callees\n- list each with file:line citation\n"
            "## Summary\n- brief dependency overview\n\n"
            "If no callers or callees are found, say so explicitly."
        )
        user = f"Code context:\n{context}\n\nAnalyse dependencies for: {name}"

        full_parts: list[str] = []
        async for text in _llm_stream(system, user):
            full_parts.append(text)
            yield ("token", text)

        full_answer = "".join(full_parts)
        callers = _extract_names(full_answer, "Callers")
        callees = _extract_names(full_answer, "Callees")
        yield ("metadata", {"target": name, "callers": callers, "callees": callees})
        yield ("sources", sources)
    except Exception as exc:
        logger.exception("stream_dependencies failed")
        yield ("error", str(exc))


# ── 3. Pattern Detection ─────────────────────────────────────────────────

@dataclass
class PatternResult:
    description: str
    matches: list[SourceInfo]


@observe(name="coboledu-patterns")
def find_patterns(description: str) -> PatternResult:
    """Semantic search for code matching a pattern description."""
    nodes, sources = _retrieve(description, top_k=10)
    return PatternResult(description=description, matches=sources)


# ── 4. Documentation Generation ──────────────────────────────────────────

@dataclass
class DocsResult:
    documentation: str
    sources: list[SourceInfo]


@observe(name="coboledu-docs")
def generate_docs(name: str, language: str = "auto") -> DocsResult:
    """Generate markdown documentation for a function or paragraph."""
    search_query = f"{name} function definition paragraph"
    if language != "auto":
        search_query += f" {language}"

    nodes, sources = _retrieve(search_query)
    context = _context_from_nodes(nodes)

    system = (
        "You are a documentation generator for the GnuCOBOL project.\n"
        "Given code snippets, produce clean markdown documentation with:\n"
        "# <name>\n"
        "## Description\n## Parameters / Data Items\n## Return Value\n"
        "## Usage Examples\n## Related Functions\n\n"
        "Cite file paths and line numbers. If the code is COBOL, "
        "document paragraphs and data items; if C, document functions."
    )
    user = f"Code context:\n{context}\n\nGenerate documentation for: {name}"

    doc = _llm_generate(system, user)
    return DocsResult(documentation=doc, sources=sources)


@observe(name="coboledu-stream-docs")
async def stream_docs(
    name: str, language: str = "auto",
) -> AsyncGenerator[tuple[str, str | list[SourceInfo]], None]:
    """Stream documentation tokens then emit sources."""
    try:
        search_query = f"{name} function definition paragraph"
        if language != "auto":
            search_query += f" {language}"

        nodes, sources = await _aretrieve(search_query)
        context = _context_from_nodes(nodes)

        system = (
            "You are a documentation generator for the GnuCOBOL project.\n"
            "Given code snippets, produce clean markdown documentation with:\n"
            "# <name>\n"
            "## Description\n## Parameters / Data Items\n## Return Value\n"
            "## Usage Examples\n## Related Functions\n\n"
            "Cite file paths and line numbers. If the code is COBOL, "
            "document paragraphs and data items; if C, document functions."
        )
        user = f"Code context:\n{context}\n\nGenerate documentation for: {name}"

        async for text in _llm_stream(system, user):
            yield ("token", text)
        yield ("sources", sources)
    except Exception as exc:
        logger.exception("stream_docs failed")
        yield ("error", str(exc))


# ── 5. Business Logic Extraction ─────────────────────────────────────────

@dataclass
class BusinessLogicResult:
    logic: str
    sources: list[SourceInfo]


@observe(name="coboledu-business-logic")
def extract_business_logic(name: str) -> BusinessLogicResult:
    """Extract business rules from a COBOL paragraph or section."""
    search_query = f"{name} COBOL business logic paragraph"
    nodes, sources = _retrieve(search_query, top_k=8)
    context = _context_from_nodes(nodes)

    system = (
        "You are a business analyst for the GnuCOBOL project.\n"
        "Given COBOL code snippets, extract the embedded business rules.\n"
        "Structure your output as:\n"
        "## Business Rules\n- numbered list of rules with conditions and actions\n"
        "## Data Transformations\n- what data is read, computed, written\n"
        "## Edge Cases / Validations\n- boundary checks, error conditions\n"
        "## Plain English Summary\n- one-paragraph explanation\n\n"
        "Cite file paths and line numbers."
    )
    user = f"Code context:\n{context}\n\nExtract business logic for: {name}"

    logic = _llm_generate(system, user)
    return BusinessLogicResult(logic=logic, sources=sources)


@observe(name="coboledu-stream-business-logic")
async def stream_business_logic(
    name: str,
) -> AsyncGenerator[tuple[str, str | list[SourceInfo]], None]:
    """Stream business logic tokens then emit sources."""
    try:
        search_query = f"{name} COBOL business logic paragraph"
        nodes, sources = await _aretrieve(search_query, top_k=8)
        context = _context_from_nodes(nodes)

        system = (
            "You are a business analyst for the GnuCOBOL project.\n"
            "Given COBOL code snippets, extract the embedded business rules.\n"
            "Structure your output as:\n"
            "## Business Rules\n- numbered list of rules with conditions and actions\n"
            "## Data Transformations\n- what data is read, computed, written\n"
            "## Edge Cases / Validations\n- boundary checks, error conditions\n"
            "## Plain English Summary\n- one-paragraph explanation\n\n"
            "Cite file paths and line numbers."
        )
        user = f"Code context:\n{context}\n\nExtract business logic for: {name}"

        async for text in _llm_stream(system, user):
            yield ("token", text)
        yield ("sources", sources)
    except Exception as exc:
        logger.exception("stream_business_logic failed")
        yield ("error", str(exc))
