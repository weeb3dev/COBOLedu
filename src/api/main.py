"""FastAPI server for the COBOLedu RAG system."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pinecone import Pinecone
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src.config import PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME
from src.observability import init_observability, langfuse_client
from src.retrieval.features import (
    explain_code,
    extract_business_logic,
    find_dependencies,
    find_patterns,
    generate_docs,
)
from src.retrieval.query import create_query_engine, query, stream_query

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceResponse(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    score: float
    preview: str
    chunk_type: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    status: str


class ExplainRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class DependencyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    direction: str = Field(default="both", pattern="^(both|callers|callees)$")


class PatternRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=2000)


class DocsRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    language: str = Field(default="auto")


class BusinessLogicRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


# ---------------------------------------------------------------------------
# App state managed via lifespan
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_observability()
    logger.info("Initialising query engine …")
    _state["engine"] = create_query_engine()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    _state["pinecone_index"] = (
        pc.Index(host=PINECONE_HOST) if PINECONE_HOST else pc.Index(name=PINECONE_INDEX_NAME)
    )
    logger.info("Query engine ready.")
    yield
    langfuse_client.flush()
    _state.clear()


app = FastAPI(
    title="COBOLedu",
    description="RAG-powered GnuCOBOL code explorer",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API routes (under /api prefix)
# ---------------------------------------------------------------------------

api = APIRouter(prefix="/api")


@api.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@api.get("/stats")
async def stats():
    try:
        idx = _state.get("pinecone_index")
        if idx is None:
            raise HTTPException(status_code=503, detail="Index not initialised")
        raw = idx.describe_index_stats()
        return raw.to_dict() if hasattr(raw, "to_dict") else raw
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@api.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    engine = _state.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialised")

    try:
        result = query(engine, req.question)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        langfuse_client.flush()

    return QueryResponse(
        answer=result.answer,
        sources=[SourceResponse(**asdict(s)) for s in result.sources],
    )


@api.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    engine = _state.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialised")

    async def event_generator():
        try:
            async for event_type, data in stream_query(engine, req.question):
                if event_type == "token":
                    yield f"event: token\ndata: {json.dumps(data)}\n\n"
                elif event_type == "sources":
                    payload = [asdict(s) for s in data]
                    yield f"event: sources\ndata: {json.dumps(payload)}\n\n"
                elif event_type == "error":
                    yield f"event: error\ndata: {json.dumps(data)}\n\n"
            yield "event: done\ndata: \n\n"
        except Exception as exc:
            logger.exception("Stream endpoint error")
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
        finally:
            langfuse_client.flush()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Feature endpoints
# ---------------------------------------------------------------------------


@api.post("/explain")
async def explain_endpoint(req: ExplainRequest):
    try:
        result = explain_code(req.query)
    except Exception as exc:
        logger.exception("Explain failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        langfuse_client.flush()
    return {
        "explanation": result.explanation,
        "sources": [asdict(s) for s in result.sources],
    }


@api.post("/dependencies")
async def dependencies_endpoint(req: DependencyRequest):
    try:
        result = find_dependencies(req.name, req.direction)
    except Exception as exc:
        logger.exception("Dependency analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        langfuse_client.flush()
    return {
        "target": result.target,
        "callers": result.callers,
        "callees": result.callees,
        "analysis": result.analysis,
        "sources": [asdict(s) for s in result.sources],
    }


@api.post("/patterns")
async def patterns_endpoint(req: PatternRequest):
    try:
        result = find_patterns(req.description)
    except Exception as exc:
        logger.exception("Pattern search failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        langfuse_client.flush()
    return {
        "description": result.description,
        "matches": [asdict(s) for s in result.matches],
    }


@api.post("/docs")
async def docs_endpoint(req: DocsRequest):
    try:
        result = generate_docs(req.name, req.language)
    except Exception as exc:
        logger.exception("Doc generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        langfuse_client.flush()
    return {
        "documentation": result.documentation,
        "sources": [asdict(s) for s in result.sources],
    }


@api.post("/business-logic")
async def business_logic_endpoint(req: BusinessLogicRequest):
    try:
        result = extract_business_logic(req.name)
    except Exception as exc:
        logger.exception("Business logic extraction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        langfuse_client.flush()
    return {
        "logic": result.logic,
        "sources": [asdict(s) for s in result.sources],
    }


app.include_router(api)

# Static files mounted last so /api/* routes take priority
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
