"""Langfuse observability setup for COBOLedu.

Import this module AFTER src.config (which calls load_dotenv) so that
LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and LANGFUSE_BASE_URL are
already in the environment when the Langfuse client initialises.
"""

from __future__ import annotations

import logging

from langfuse import get_client, observe  # noqa: F401 – re-exported
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

logger = logging.getLogger(__name__)


def init_observability() -> None:
    """Instrument LlamaIndex and verify Langfuse connectivity."""
    LlamaIndexInstrumentor().instrument()

    client = get_client()
    if client.auth_check():
        logger.info("Langfuse client authenticated — tracing active")
    else:
        logger.warning(
            "Langfuse auth check failed — traces will NOT be recorded. "
            "Verify LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_BASE_URL."
        )


langfuse_client = get_client()
