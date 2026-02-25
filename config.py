from pathlib import Path
from os import getenv

ROOT = Path(__file__).parent

# Data paths
BOOKS_DIR         = ROOT / "data" / "books"
PARSED_DIR        = ROOT / "data" / "parsed"

# Embedding
EMBED_MODEL       = "BAAI/bge-large-en-v1.5"
QDRANT_PATH       = ROOT / "data" / "qdrant"
QDRANT_COLLECTION = "bookparser"

# Chunking
CHUNK_MAX_TOKENS  = 512
CHUNK_OVERLAP_PCT = 0.12

# Graph
GRAPH_PATH        = ROOT / "data" / "graph.gpickle"
GRAPH_NER_TEXT_CAP = 2000   # max chars per section fed to NER
GRAPH_ENTITY_CAP  = 50      # max entities returned per book

# RAG / LLM
OLLAMA_MODEL      = "phi3:mini"
OLLAMA_BASE_URL   = "http://localhost:11434"
TOP_K_RETRIEVAL   = 5
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RELEVANCE_THRESHOLD = -2.0  # CrossEncoder min score to attempt LLM answer
RERANK_MIN_SCORE  = -1.0    # CrossEncoder min score to include chunk in context
CHUNK_CONTEXT_CAP = 500     # max chars of chunk text sent to LLM

# Flashcards
FLASHCARDS_DIR         = ROOT / "data" / "flashcards"
FLASHCARDS_PER_SECTION = 3   # Q&A pairs to generate per section

# API
API_URL           = "http://localhost:8000"
API_PORT          = 8000
