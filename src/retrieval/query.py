"""Query engine: retrieval from Pinecone + answer generation via Claude."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import anthropic
import voyageai
from cachetools import TTLCache
from langfuse import observe
from llama_index.core import Settings, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, RERANK_MODEL, RETRIEVAL_K, TOP_K
from src.retrieval.embeddings import get_embed_model
from src.retrieval.vector_store import get_index, get_vector_store

logger = logging.getLogger(__name__)

_REPO_DIR_MARKER = "gnucobol-source/"
_PATH_PREFIX_RE = re.compile(r"[^\s\"']*?gnucobol-source/")

_vo_client: voyageai.Client | None = None
_query_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)


def _get_voyage_client() -> voyageai.Client:
    global _vo_client
    if _vo_client is None:
        _vo_client = voyageai.Client()
    return _vo_client


def rerank_nodes(
    question: str,
    nodes,
    top_k: int = TOP_K,
    max_per_file: int = 3,
):
    """Re-rank with Voyage rerank-2.5, enforce file diversity, return top-k."""
    if not nodes:
        return nodes
    vo = _get_voyage_client()
    docs = [n.get_content() for n in nodes]
    reranked = vo.rerank(
        question, docs, model=RERANK_MODEL, top_k=min(len(docs), top_k * 3),
    )

    result: list = []
    file_counts: dict[str, int] = {}
    for r in reranked.results:
        node = nodes[r.index]
        fp = normalize_path(node.metadata.get("file_path", "unknown"))
        count = file_counts.get(fp, 0)
        if count < max_per_file:
            node.score = r.relevance_score
            result.append(node)
            file_counts[fp] = count + 1
            if len(result) >= top_k:
                break
    return result


def normalize_path(raw: str) -> str:
    """Strip everything up to and including 'gnucobol-source/' from a path."""
    idx = raw.find(_REPO_DIR_MARKER)
    if idx == -1:
        return raw
    return raw[idx + len(_REPO_DIR_MARKER) :]


def _scrub_answer_paths(text: str) -> str:
    """Remove absolute-path prefixes ending in gnucobol-source/ from prose."""
    return _PATH_PREFIX_RE.sub("", text)


# ── Query preprocessing / expansion ──────────────────────────────────────

_COBOL_EXPANSIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"file\s+i/?o", re.I), "READ WRITE OPEN CLOSE FILE fileio.c"),
    (re.compile(r"error\s+handl", re.I), "error exception cob_runtime_error common.c"),
    (re.compile(r"entry\s+point", re.I), "main argc argv cobc.c"),
    (re.compile(r"\bdata\s+type", re.I), "PIC PICTURE field type tree.c field.c"),
    (re.compile(r"\bloop\b", re.I), "PERFORM VARYING UNTIL"),
    (re.compile(r"\bdependenc", re.I), "PERFORM CALL COPY"),
    (re.compile(r"\bnumeric\b", re.I), "cob_decimal add subtract numeric.c"),
    (re.compile(r"\bmemory\s+manag", re.I), "cob_malloc alloc free common.c"),
    (re.compile(r"\bstring\s+oper", re.I), "STRING INSPECT UNSTRING strings.c"),
    (re.compile(r"\bMOVE\s+statement", re.I), "cob_move move.c libcob"),
    (re.compile(r"\bscanner|lexer\b", re.I), "scanner.l token lex"),
    (re.compile(r"\bcode\s*gen", re.I), "codegen.c output generate cb_"),
    (re.compile(r"\btype\s+check", re.I), "typeck.c cb_validate"),
    (re.compile(r"\bparser\b", re.I), "parser.y cobc YACC grammar syntax rule"),
    (re.compile(r"\bdialect", re.I), "config/ default.conf standard"),
    (re.compile(r"\bcompiler\s+flag", re.I), "cobc option -f help.c"),
    (re.compile(r"\bCOPY\b.*\bREPLACE\b", re.I), "COPY REPLACE copybook"),
    (re.compile(r"\bEVALUATE\b", re.I), "EVALUATE WHEN"),
    (re.compile(r"\btest\s+program", re.I), "tests/ .at COBOL test"),
]

_FILE_HINT_RE = re.compile(r"\b([a-zA-Z]\w*\.(?:c|h|l|y|conf))\b")
_COBOL_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")


def extract_file_hints(text: str) -> list[str]:
    """Extract filename hints like move.c, parser.y from text."""
    return list(dict.fromkeys(_FILE_HINT_RE.findall(text)))


def extract_cobol_identifiers(text: str) -> list[str]:
    """Extract COBOL-style hyphenated identifiers like CUSTOMER-RECORD."""
    return list(dict.fromkeys(_COBOL_ID_RE.findall(text)))


def preprocess_query(question: str) -> str:
    """Expand a query with COBOL/GnuCOBOL-specific synonyms for better retrieval."""
    extras: list[str] = []
    for pattern, expansion in _COBOL_EXPANSIONS:
        if pattern.search(question):
            extras.append(expansion)

    for cid in extract_cobol_identifiers(question):
        parts = cid.split("-")
        if len(parts) >= 2:
            extras.append(f"{cid} {' '.join(parts)}")

    if not extras:
        return question
    return f"{question} {' '.join(extras)}"


# ── Metadata-based language classification ───────────────────────────────

_COBOL_SIGNALS = re.compile(
    r"\bCOBOL\b|PERFORM\b|EVALUATE\b|CUSTOMER-RECORD|CALCULATE-|MODULE-"
    r"|\btest\s+program|\b\.at\s+file|\bparagraph\b|\bCOPY\b|\bcopybook\b",
    re.I,
)
_C_SIGNALS = re.compile(
    r"\bruntime\b|\blibcob\b|\bcobc\b|\bcompiler\b|\bparser\b"
    r"|\bscanner\b|\blexer\b|\bcodegen\b|\btypeck\b|\bcob_\w+",
    re.I,
)
_IMPL_OVERRIDE = re.compile(
    r"\bdefined\b|\bimplemented\b|\bimplementation\b|\bconfigur",
    re.I,
)


def _detect_language_filter(question: str) -> str | None:
    """Return 'COBOL' or 'C' if the query strongly targets one language, else None."""
    cobol_hits = len(_COBOL_SIGNALS.findall(question))
    c_hits = len(_C_SIGNALS.findall(question))

    if _IMPL_OVERRIDE.search(question):
        return None

    if cobol_hits >= 1 and c_hits == 0:
        return "COBOL"
    if c_hits >= 2 and cobol_hits == 0:
        return "C"
    return None


def _merged_retrieve(
    index,
    expanded_query: str,
    language: str | None,
    file_hints: list[str] | None = None,
    cobol_ids: list[str] | None = None,
):
    """Primary retrieval + language/file-hint/identifier passes, deduplicated."""
    primary = index.as_retriever(similarity_top_k=RETRIEVAL_K)
    nodes = primary.retrieve(expanded_query)
    seen_ids = {n.node_id for n in nodes}

    if language:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="language", value=language)]
        )
        secondary = index.as_retriever(
            similarity_top_k=10, filters=filters,
        )
        for n in secondary.retrieve(expanded_query):
            if n.node_id not in seen_ids:
                nodes.append(n)
                seen_ids.add(n.node_id)

    for hint in (file_hints or [])[:3]:
        hint_retriever = index.as_retriever(similarity_top_k=5)
        for n in hint_retriever.retrieve(f"implementation in {hint}"):
            if n.node_id not in seen_ids:
                nodes.append(n)
                seen_ids.add(n.node_id)

    for cid in (cobol_ids or [])[:2]:
        parts = cid.split("-")
        id_query = f"{cid} {' '.join(parts)} COBOL paragraph section"
        id_retriever = index.as_retriever(similarity_top_k=5)
        for n in id_retriever.retrieve(id_query):
            if n.node_id not in seen_ids:
                nodes.append(n)
                seen_ids.add(n.node_id)

    return nodes


CODE_QA_SYSTEM = (
    "You are a code analysis assistant for the GnuCOBOL project — an open-source "
    "COBOL compiler written in C with COBOL test programs.\n\n"
    "Given code snippets retrieved from the codebase, answer the user's question. "
    "Always cite specific file paths and line numbers "
    "(format: filename:line_start-line_end). If the snippets don't contain enough "
    "information, say so clearly and suggest what to search for instead."
)

CODE_QA_USER_TMPL = """\
Retrieved code:
{context_str}

Question: {query_str}

Answer:
"""

CODE_QA_PROMPT_TMPL = f"{CODE_QA_SYSTEM}\n\n{CODE_QA_USER_TMPL}"
CODE_QA_PROMPT = PromptTemplate(CODE_QA_PROMPT_TMPL)

_CACHED_SYSTEM_BLOCK = [
    {
        "type": "text",
        "text": CODE_QA_SYSTEM,
        "cache_control": {"type": "ephemeral"},
    }
]


@dataclass
class SourceInfo:
    file_path: str
    line_start: int
    line_end: int
    score: float
    preview: str
    chunk_type: str


@dataclass
class QueryResult:
    answer: str
    sources: list[SourceInfo] = field(default_factory=list)


def get_llm() -> Anthropic:
    return Anthropic(
        model="claude-sonnet-4-20250514",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=4096,
    )


def create_query_engine() -> RetrieverQueryEngine:
    """Wire Pinecone retriever + Claude LLM into a RetrieverQueryEngine."""
    Settings.embed_model = get_embed_model()
    Settings.llm = get_llm()

    vector_store = get_vector_store()
    index = get_index(vector_store)

    retriever = index.as_retriever(similarity_top_k=RETRIEVAL_K)

    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=CODE_QA_PROMPT,
    )
    return engine


@observe(name="coboledu-query")
def query(engine: RetrieverQueryEngine, question: str) -> QueryResult:
    """Retrieve top-20 (+filtered pass), rerank to top-5, generate answer."""
    cached = _query_cache.get(question)
    if cached is not None:
        return cached

    index = engine._retriever._index
    expanded = preprocess_query(question)
    lang = _detect_language_filter(question)
    file_hints = extract_file_hints(expanded)
    cobol_ids = extract_cobol_identifiers(question)
    nodes = _merged_retrieve(index, expanded, lang, file_hints, cobol_ids)
    nodes = rerank_nodes(expanded, nodes, top_k=TOP_K)
    sources = _extract_sources(nodes)

    context_str = "\n\n".join(node.get_content() for node in nodes)
    context_str = _scrub_answer_paths(context_str)
    user_msg = CODE_QA_USER_TMPL.format(context_str=context_str, query_str=question)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=_CACHED_SYSTEM_BLOCK,
        messages=[{"role": "user", "content": user_msg}],
    )
    answer = _scrub_answer_paths(resp.content[0].text)
    result = QueryResult(answer=answer, sources=sources)
    _query_cache[question] = result
    return result


def _extract_sources(nodes) -> list[SourceInfo]:
    """Build SourceInfo list from retrieved NodeWithScore objects."""
    sources: list[SourceInfo] = []
    for node in nodes:
        meta = node.metadata or {}
        text = node.get_content()
        sources.append(
            SourceInfo(
                file_path=normalize_path(meta.get("file_path", "unknown")),
                line_start=meta.get("line_start", 0),
                line_end=meta.get("line_end", 0),
                score=round(node.score or 0.0, 4),
                preview=text[:500],
                chunk_type=meta.get("chunk_type", "unknown"),
            )
        )
    return sources


@observe(name="coboledu-stream-query")
async def stream_query(
    engine: RetrieverQueryEngine,
    question: str,
) -> AsyncGenerator[tuple[str, str | list[SourceInfo]], None]:
    """Async generator that streams answer tokens then emits sources.

    Yields tuples of (event_type, data):
      ("token", <str>)   — a chunk of the answer text
      ("sources", <list>) — SourceInfo list (sent once, after all tokens)
      ("error", <str>)    — if something goes wrong mid-stream
    """
    cached = _query_cache.get(question)
    if cached is not None:
        yield ("token", cached.answer)
        yield ("sources", cached.sources)
        return

    index = engine._retriever._index

    expanded = preprocess_query(question)
    lang = _detect_language_filter(question)
    file_hints = extract_file_hints(expanded)
    cobol_ids = extract_cobol_identifiers(question)
    nodes = await asyncio.to_thread(
        _merged_retrieve, index, expanded, lang, file_hints, cobol_ids,
    )
    nodes = rerank_nodes(expanded, nodes, top_k=TOP_K)
    sources = _extract_sources(nodes)

    context_str = "\n\n".join(node.get_content() for node in nodes)
    context_str = _scrub_answer_paths(context_str)
    user_msg = CODE_QA_USER_TMPL.format(context_str=context_str, query_str=question)

    full_answer_parts: list[str] = []
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    try:
        async with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=_CACHED_SYSTEM_BLOCK,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            async for text in stream.text_stream:
                full_answer_parts.append(text)
                yield ("token", text)
    except Exception as exc:
        logger.exception("Streaming generation failed")
        yield ("error", str(exc))
        return

    _query_cache[question] = QueryResult(
        answer="".join(full_answer_parts), sources=sources,
    )
    yield ("sources", sources)
