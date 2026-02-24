from pathlib import Path

ROOT = Path(__file__).parent

# Data paths
BOOKS_DIR       = ROOT / "data" / "books"
PARSED_DIR      = ROOT / "data" / "parsed"

# Embedding
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
QDRANT_PATH    = ROOT / "data" / "qdrant"
QDRANT_COLLECTION = "bookparser"

# Chunking
CHUNK_MAX_TOKENS  = 512
CHUNK_OVERLAP_PCT = 0.12

# Graph (Neo4j)
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"

# Relation extraction
REBEL_MODEL    = "Babelscape/rebel-large"

# RAG / LLM
OLLAMA_MODEL   = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K_RETRIEVAL = 5
