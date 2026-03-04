"""Create the COBOLedu evaluation dataset in Langfuse.

Run once:  python -m scripts.create_eval_dataset
"""

from __future__ import annotations

from src.config import *  # noqa: F401,F403  — loads .env
from langfuse import get_client


DATASET_NAME = "coboledu-eval-v1"

# ── 6 grader test queries (verbatim) ─────────────────────────────────────
GRADER_ITEMS = [
    {
        "input": {"query": "Where is the main entry point of this program?"},
        "expected_output": {
            "expected_files": ["cobc/cobc.c"],
            "expected_terms": ["main", "argc", "argv"],
        },
    },
    {
        "input": {"query": "What functions modify the CUSTOMER-RECORD?"},
        "expected_output": {
            "expected_files": ["tests/"],
            "expected_terms": ["MOVE", "CUSTOMER-RECORD", "WRITE"],
        },
    },
    {
        "input": {"query": "Explain what the CALCULATE-INTEREST paragraph does"},
        "expected_output": {
            "expected_files": ["tests/"],
            "expected_terms": ["CALCULATE", "INTEREST", "COMPUTE"],
        },
    },
    {
        "input": {"query": "Find all file I/O operations"},
        "expected_output": {
            "expected_files": ["libcob/fileio.c", "tests/"],
            "expected_terms": ["READ", "WRITE", "OPEN", "CLOSE", "FILE"],
        },
    },
    {
        "input": {"query": "What are the dependencies of MODULE-X?"},
        "expected_output": {
            "expected_files": ["tests/", "cobc/"],
            "expected_terms": ["PERFORM", "CALL", "MODULE"],
        },
    },
    {
        "input": {"query": "Show me error handling patterns in this codebase"},
        "expected_output": {
            "expected_files": ["libcob/common.c", "cobc/cobc.c"],
            "expected_terms": ["error", "exception", "cob_runtime_error"],
        },
    },
]

# ── Additional items by category ─────────────────────────────────────────

COMPILER_ITEMS = [
    {
        "input": {"query": "How does the GnuCOBOL parser handle PERFORM statements?"},
        "expected_output": {
            "expected_files": ["cobc/parser.y", "cobc/codegen.c"],
            "expected_terms": ["perform", "PERFORM"],
        },
    },
    {
        "input": {"query": "What does the scanner/lexer do in the compiler?"},
        "expected_output": {
            "expected_files": ["cobc/scanner.l"],
            "expected_terms": ["scanner", "token", "lex"],
        },
    },
    {
        "input": {"query": "How does code generation work in GnuCOBOL?"},
        "expected_output": {
            "expected_files": ["cobc/codegen.c"],
            "expected_terms": ["codegen", "output", "generate", "cb_"],
        },
    },
    {
        "input": {"query": "Where is type checking implemented in the compiler?"},
        "expected_output": {
            "expected_files": ["cobc/typeck.c"],
            "expected_terms": ["type", "check", "cb_validate"],
        },
    },
]

RUNTIME_ITEMS = [
    {
        "input": {"query": "How does GnuCOBOL handle numeric operations?"},
        "expected_output": {
            "expected_files": ["libcob/numeric.c"],
            "expected_terms": ["numeric", "cob_decimal", "add", "subtract"],
        },
    },
    {
        "input": {"query": "How does the MOVE statement work at the runtime level?"},
        "expected_output": {
            "expected_files": ["libcob/move.c"],
            "expected_terms": ["move", "cob_move"],
        },
    },
    {
        "input": {"query": "How does GnuCOBOL handle string operations?"},
        "expected_output": {
            "expected_files": ["libcob/strings.c"],
            "expected_terms": ["string", "STRING", "INSPECT"],
        },
    },
    {
        "input": {"query": "How does memory management work in the runtime?"},
        "expected_output": {
            "expected_files": ["libcob/common.c"],
            "expected_terms": ["alloc", "free", "memory", "cob_malloc"],
        },
    },
]

COBOL_TEST_ITEMS = [
    {
        "input": {"query": "Show examples of PERFORM VARYING loops in test programs"},
        "expected_output": {
            "expected_files": ["tests/"],
            "expected_terms": ["PERFORM", "VARYING", "UNTIL"],
        },
    },
    {
        "input": {"query": "How are EVALUATE and IF statements tested?"},
        "expected_output": {
            "expected_files": ["tests/"],
            "expected_terms": ["EVALUATE", "WHEN", "IF", "ELSE"],
        },
    },
    {
        "input": {"query": "Show COPY and REPLACE usage in test files"},
        "expected_output": {
            "expected_files": ["tests/"],
            "expected_terms": ["COPY", "REPLACE"],
        },
    },
]

CONFIG_ITEMS = [
    {
        "input": {"query": "What COBOL dialect configurations are supported?"},
        "expected_output": {
            "expected_files": ["config/"],
            "expected_terms": ["dialect", "conf", "standard"],
        },
    },
    {
        "input": {"query": "What compiler flags and options are available?"},
        "expected_output": {
            "expected_files": ["cobc/cobc.c", "cobc/help.c", "config/"],
            "expected_terms": ["flag", "option", "-f", "cobc"],
        },
    },
    {
        "input": {"query": "How are COBOL data types defined in GnuCOBOL?"},
        "expected_output": {
            "expected_files": ["cobc/tree.c", "cobc/tree.h", "cobc/field.c"],
            "expected_terms": ["PIC", "PICTURE", "field", "type"],
        },
    },
]

ALL_ITEMS = GRADER_ITEMS + COMPILER_ITEMS + RUNTIME_ITEMS + COBOL_TEST_ITEMS + CONFIG_ITEMS


def main() -> None:
    langfuse = get_client()

    langfuse.create_dataset(
        name=DATASET_NAME,
        description="COBOLedu evaluation dataset — grader queries + supplementary items",
        metadata={"version": "v1", "total_items": len(ALL_ITEMS)},
    )
    print(f"Created dataset '{DATASET_NAME}'")

    for i, item in enumerate(ALL_ITEMS, 1):
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=item["input"],
            expected_output=item["expected_output"],
            metadata={"index": i},
        )
        print(f"  [{i}/{len(ALL_ITEMS)}] {item['input']['query'][:60]}…")

    langfuse.flush()
    print(f"\nDone — {len(ALL_ITEMS)} items uploaded to '{DATASET_NAME}'")


if __name__ == "__main__":
    main()
