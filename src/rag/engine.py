"""
Phase 4 — RAG reference engine.

Pipeline:
  query -> embed -> ChromaDB top-k -> BM25 hybrid -> CrossEncoder rerank -> Ollama LLM -> cited answer
"""


def retrieve(query: str, chroma_dir, collection_name: str, top_k: int) -> list[dict]:
    """Dense vector retrieval from ChromaDB. Returns chunks with metadata."""
    raise NotImplementedError


def hybrid_retrieve(query: str, chroma_dir, collection_name: str, top_k: int) -> list[dict]:
    """Combine dense (ChromaDB) + sparse (BM25) retrieval, deduplicate results."""
    raise NotImplementedError


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-rank chunks using cross-encoder/ms-marco-MiniLM-L-6-v2."""
    raise NotImplementedError


def generate_answer(query: str, chunks: list[dict], ollama_model: str, base_url: str) -> dict:
    """Send query + retrieved context to Ollama LLM.
    Returns { answer: str, citations: list[str] }
    Citation format: '[Book, Chapter X, Section Y, Page Z]'"""
    raise NotImplementedError


def ask(query: str) -> dict:
    """End-to-end entry point: query -> cited answer. Loads config internally."""
    raise NotImplementedError
