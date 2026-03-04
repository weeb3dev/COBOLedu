"""Local eval diagnostic — runs all 20 queries and prints per-item precision breakdown.

Usage:
    python -m scripts.eval_diagnostic
    python -m scripts.eval_diagnostic --retrieval-only   # skip LLM, just show retrieved files
"""

from __future__ import annotations

import argparse
import time

from src.config import *  # noqa: F401,F403  — loads .env
from src.observability import init_observability

init_observability()

from scripts.create_eval_dataset import ALL_ITEMS  # noqa: E402
from src.retrieval.query import (  # noqa: E402
    create_query_engine,
    normalize_path,
    preprocess_query,
    rerank_nodes,
    extract_file_hints,
    extract_cobol_identifiers,
    _detect_language_filter,
    _merged_retrieve,
    query,
)
from src.config import TOP_K  # noqa: E402


def _precision(retrieved: list[str], expected: list[str]) -> tuple[int, float]:
    if not retrieved:
        return 0, 0.0
    hits = 0
    for rf in retrieved:
        for ef in expected:
            if ef in rf:
                hits += 1
                break
    return hits, hits / len(retrieved)


def run_diagnostic(retrieval_only: bool = False) -> None:
    engine = create_query_engine()
    index = engine._retriever._index

    total_precision = 0.0
    results = []

    for i, item in enumerate(ALL_ITEMS, 1):
        q = item["input"]["query"]
        expected_files = item["expected_output"]["expected_files"]
        expected_terms = item["expected_output"].get("expected_terms", [])

        t0 = time.time()

        expanded = preprocess_query(q)
        lang = _detect_language_filter(q)
        file_hints = extract_file_hints(expanded)
        cobol_ids = extract_cobol_identifiers(q)
        nodes = _merged_retrieve(index, expanded, lang, file_hints, cobol_ids)
        nodes = rerank_nodes(expanded, nodes, top_k=TOP_K)

        retrieved = [normalize_path(n.metadata.get("file_path", "?")) for n in nodes]
        scores = [round(n.score or 0, 4) for n in nodes]

        hits, prec = _precision(retrieved, expected_files)
        total_precision += prec
        latency = time.time() - t0

        answer_excerpt = ""
        term_found = []
        term_missing = []

        if not retrieval_only:
            result = query(engine, q)
            answer_lower = result.answer.lower()
            for t in expected_terms:
                if t.lower() in answer_lower:
                    term_found.append(t)
                else:
                    term_missing.append(t)
            answer_excerpt = result.answer[:120].replace("\n", " ")

        results.append({
            "idx": i,
            "query": q,
            "expected_files": expected_files,
            "retrieved": retrieved,
            "scores": scores,
            "hits": hits,
            "precision": prec,
            "latency": latency,
            "lang_filter": lang,
            "expanded_extra": expanded[len(q):].strip() if expanded != q else "",
            "term_found": term_found,
            "term_missing": term_missing,
            "answer_excerpt": answer_excerpt,
        })

    print("\n" + "=" * 100)
    print("EVAL DIAGNOSTIC REPORT")
    print("=" * 100)

    for r in results:
        status = "PASS" if r["precision"] >= 0.6 else "WARN" if r["precision"] >= 0.4 else "FAIL"
        print(f"\n[{r['idx']:2d}] [{status}] P@5={r['precision']:.1%}  ({r['hits']}/5)  {r['latency']:.1f}s")
        print(f"     Q: {r['query']}")
        if r["lang_filter"]:
            print(f"     Lang filter: {r['lang_filter']}")
        if r["expanded_extra"]:
            print(f"     Expanded: +{r['expanded_extra'][:80]}")
        print(f"     Expected: {r['expected_files']}")
        print(f"     Got:      {r['retrieved']}")
        print(f"     Scores:   {r['scores']}")
        if r["term_missing"]:
            print(f"     Terms OK:      {r['term_found']}")
            print(f"     Terms MISSING: {r['term_missing']}")

    avg = total_precision / len(ALL_ITEMS)
    print(f"\n{'=' * 100}")
    print(f"AVG PRECISION@5: {avg:.1%}  ({avg:.3f})")
    print(f"{'=' * 100}")

    failing = [r for r in results if r["precision"] < 0.4]
    if failing:
        print(f"\n--- {len(failing)} items with precision < 0.4 ---")
        for r in failing:
            print(f"  [{r['idx']:2d}] P@5={r['precision']:.1%}  {r['query'][:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="COBOLedu eval diagnostic")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip LLM generation, only check retrieval precision")
    args = parser.parse_args()
    run_diagnostic(retrieval_only=args.retrieval_only)


if __name__ == "__main__":
    main()
