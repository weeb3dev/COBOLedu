"""Run a COBOLedu evaluation experiment against the Langfuse dataset.

Usage:
    python -m scripts.run_eval                      # default name "baseline-v1"
    python -m scripts.run_eval --name with-reranking-v1
"""

from __future__ import annotations

import argparse
import time

from src.config import *  # noqa: F401,F403  — loads .env
from src.observability import init_observability, langfuse_client

init_observability()

from langfuse import Evaluation, get_client  # noqa: E402
from src.retrieval.query import create_query_engine, query  # noqa: E402

DATASET_NAME = "coboledu-eval-v1"


# ── Task function ─────────────────────────────────────────────────────────

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_query_engine()
    return _engine


def coboledu_task(*, item, **kwargs):
    """Run a single query through the COBOLedu pipeline."""
    engine = _get_engine()
    question = item.input["query"]

    t0 = time.time()
    result = query(engine, question)
    latency = time.time() - t0

    source_files = [s.file_path for s in result.sources]
    source_scores = [s.score for s in result.sources]

    return {
        "answer": result.answer,
        "source_files": source_files,
        "source_scores": source_scores,
        "latency_s": round(latency, 2),
    }


# ── Item-level evaluators ────────────────────────────────────────────────

def precision_evaluator(*, output, expected_output, **kwargs) -> Evaluation:
    """Precision@5: fraction of top-5 retrieved files matching expected files."""
    retrieved = output.get("source_files", [])[:5]
    expected = expected_output.get("expected_files", [])

    if not retrieved:
        return Evaluation(name="precision_at_5", value=0.0, comment="No files retrieved")

    hits = 0
    for rf in retrieved:
        for ef in expected:
            if ef in rf:
                hits += 1
                break

    precision = hits / len(retrieved)
    return Evaluation(
        name="precision_at_5",
        value=round(precision, 3),
        comment=f"{hits}/{len(retrieved)} hits | retrieved={retrieved} expected={expected}",
    )


def term_coverage_evaluator(*, output, expected_output, **kwargs) -> Evaluation:
    """Fraction of expected terms that appear (case-insensitive) in the answer."""
    answer = (output.get("answer") or "").lower()
    terms = expected_output.get("expected_terms", [])

    if not terms:
        return Evaluation(name="term_coverage", value=1.0, comment="No terms to check")

    found = sum(1 for t in terms if t.lower() in answer)
    coverage = found / len(terms)
    missing = [t for t in terms if t.lower() not in answer]

    return Evaluation(
        name="term_coverage",
        value=round(coverage, 3),
        comment=f"{found}/{len(terms)} terms found"
        + (f" | missing: {missing}" if missing else ""),
    )


def latency_evaluator(*, output, **kwargs) -> Evaluation:
    """Track per-query latency (target: <3s)."""
    latency = output.get("latency_s", 0)
    return Evaluation(
        name="latency_s",
        value=latency,
        comment=f"{'PASS' if latency < 3 else 'SLOW'} — {latency}s",
    )


# ── Run-level evaluators ─────────────────────────────────────────────────

def avg_precision_run(*, item_results, **kwargs) -> Evaluation:
    vals = [
        e.value
        for r in item_results
        for e in r.evaluations
        if e.name == "precision_at_5" and e.value is not None
    ]
    if not vals:
        return Evaluation(name="avg_precision_at_5", value=None, comment="No data")
    avg = sum(vals) / len(vals)
    return Evaluation(
        name="avg_precision_at_5",
        value=round(avg, 3),
        comment=f"Mean precision@5 across {len(vals)} items: {avg:.1%}",
    )


def avg_term_coverage_run(*, item_results, **kwargs) -> Evaluation:
    vals = [
        e.value
        for r in item_results
        for e in r.evaluations
        if e.name == "term_coverage" and e.value is not None
    ]
    if not vals:
        return Evaluation(name="avg_term_coverage", value=None, comment="No data")
    avg = sum(vals) / len(vals)
    return Evaluation(
        name="avg_term_coverage",
        value=round(avg, 3),
        comment=f"Mean term coverage across {len(vals)} items: {avg:.1%}",
    )


def avg_latency_run(*, item_results, **kwargs) -> Evaluation:
    vals = [
        e.value
        for r in item_results
        for e in r.evaluations
        if e.name == "latency_s" and e.value is not None
    ]
    if not vals:
        return Evaluation(name="avg_latency_s", value=None, comment="No data")
    avg = sum(vals) / len(vals)
    return Evaluation(
        name="avg_latency_s",
        value=round(avg, 2),
        comment=f"Mean latency across {len(vals)} items: {avg:.2f}s",
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run COBOLedu eval experiment")
    parser.add_argument("--name", default="baseline-v1", help="Experiment name")
    args = parser.parse_args()

    langfuse = get_client()
    dataset = langfuse.get_dataset(DATASET_NAME)

    print(f"Running experiment '{args.name}' on dataset '{DATASET_NAME}' …")

    result = dataset.run_experiment(
        name=args.name,
        task=coboledu_task,
        evaluators=[precision_evaluator, term_coverage_evaluator, latency_evaluator],
        run_evaluators=[avg_precision_run, avg_term_coverage_run, avg_latency_run],
    )

    print(result.format())
    langfuse_client.flush()


if __name__ == "__main__":
    main()
