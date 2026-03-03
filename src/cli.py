"""Interactive CLI for querying the COBOLedu RAG system."""

from __future__ import annotations

import sys

from src.retrieval.query import QueryResult, create_query_engine, query

PROMPT = "\n\033[1;36mcoboledu>\033[0m "


def _print_result(result: QueryResult) -> None:
    print(f"\n\033[1m{result.answer}\033[0m")

    if result.sources:
        print(f"\n\033[90m{'─' * 60}\033[0m")
        print("\033[1;33mSources:\033[0m")
        for src in result.sources:
            loc = f"{src.file_path}:{src.line_start}-{src.line_end}"
            print(f"  \033[94m{loc}\033[0m  (score: {src.score})  [{src.chunk_type}]")


def main() -> None:
    print("COBOLedu — GnuCOBOL Code Explorer")
    print("Initialising query engine …")
    engine = create_query_engine()
    print("Ready. Type a question, or 'quit' to exit.\n")

    while True:
        try:
            question = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Bye.")
            break

        try:
            result = query(engine, question)
            _print_result(result)
        except Exception as exc:
            print(f"\n\033[31mError: {exc}\033[0m", file=sys.stderr)


if __name__ == "__main__":
    main()
