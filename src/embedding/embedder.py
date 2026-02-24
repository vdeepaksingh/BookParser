"""
Phase 2 — Section-aware chunking and embedding into ChromaDB.

Each chunk stored with metadata:
  { book, author, chapter_title, chapter_index, section_title, page_start, page_end }
"""
from pathlib import Path


def chunk_book(book_json: dict, max_tokens: int, overlap_pct: float) -> list[dict]:
    """Split a parsed book JSON into overlapping chunks at section boundaries.
    Returns list of { text, metadata } dicts."""
    raise NotImplementedError


def embed_and_store(chunks: list[dict], collection_name: str, chroma_dir: Path) -> None:
    """Embed chunks with BAAI/bge-large-en-v1.5 and upsert into ChromaDB."""
    raise NotImplementedError


def embed_all(parsed_dir: Path, chroma_dir: Path) -> None:
    """Load all parsed JSONs, chunk, embed, and store. Entry point for M2."""
    raise NotImplementedError
