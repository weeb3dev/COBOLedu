"""Query engine: retrieval from Pinecone + answer generation via Claude."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llama_index.core import Settings, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, TOP_K
from src.retrieval.embeddings import get_embed_model
from src.retrieval.vector_store import get_index, get_vector_store

logger = logging.getLogger(__name__)

CODE_QA_PROMPT_TMPL = """\
You are a code analysis assistant for the GnuCOBOL project — an open-source COBOL \
compiler written in C with COBOL test programs.

Given the following code snippets retrieved from the codebase, answer the user's \
question. Always cite specific file paths and line numbers \
(format: filename:line_start-line_end). If the snippets don't contain enough \
information, say so clearly and suggest what to search for instead.

Retrieved code:
{context_str}

Question: {query_str}

Answer:
"""

CODE_QA_PROMPT = PromptTemplate(CODE_QA_PROMPT_TMPL)


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

    retriever = index.as_retriever(similarity_top_k=TOP_K)

    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=CODE_QA_PROMPT,
    )
    return engine


def query(engine: RetrieverQueryEngine, question: str) -> QueryResult:
    """Run a question through the engine and return structured results."""
    response = engine.query(question)

    sources: list[SourceInfo] = []
    for node in response.source_nodes:
        meta = node.metadata or {}
        text = node.get_content()
        sources.append(
            SourceInfo(
                file_path=meta.get("file_path", "unknown"),
                line_start=meta.get("line_start", 0),
                line_end=meta.get("line_end", 0),
                score=round(node.score or 0.0, 4),
                preview=text[:200],
                chunk_type=meta.get("chunk_type", "unknown"),
            )
        )

    return QueryResult(answer=str(response), sources=sources)
