from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GNUCOBOL_SOURCE_DIR = PROJECT_ROOT / "gnucobol-source"

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "coboledu")
PINECONE_HOST = os.getenv("PINECONE_HOST")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

EMBEDDING_MODEL = "voyage-code-3"
EMBEDDING_DIMENSION = 1024
TOP_K = 5
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
