"""
Phase 2 — Section-aware chunking and embedding into Qdrant (local).

Each chunk stored with metadata:
  { book, author, chapter_title, chapter_index, section_title, page_start, page_end }
"""
import json
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, QDRANT_PATH, QDRANT_COLLECTION, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_PCT

VECTOR_SIZE = 1024  # BAAI/bge-large-en-v1.5 output dim


def chunk_book(book_json: dict, max_tokens: int = CHUNK_MAX_TOKENS, overlap_pct: float = CHUNK_OVERLAP_PCT) -> list[dict]:
    """Split a parsed book JSON into overlapping chunks at section boundaries.
    Returns list of { text, metadata } dicts."""
    chunks = []
    words_overlap = int(max_tokens * overlap_pct)

    for chapter in book_json["chapters"]:
        for section in chapter["sections"]:
            text = section["text"].strip()
            if not text:
                continue

            metadata = {
                "book": book_json["book"],
                "author": book_json["author"],
                "source_file": book_json["source_file"],
                "chapter_title": chapter["title"],
                "chapter_index": chapter["index"],
                "section_title": section["title"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            }

            words = text.split()
            if len(words) <= max_tokens:
                chunks.append({"text": text, "metadata": metadata})
                continue

            start = 0
            while start < len(words):
                end = min(start + max_tokens, len(words))
                chunks.append({"text": " ".join(words[start:end]), "metadata": metadata})
                if end == len(words):
                    break
                start += max_tokens - words_overlap

    return chunks


def _get_client(qdrant_path: Path) -> QdrantClient:
    qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(qdrant_path))


def _ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def embed_and_store(chunks: list[dict], qdrant_path: Path = QDRANT_PATH) -> None:
    """Embed chunks with BAAI/bge-large-en-v1.5 and upsert into Qdrant."""
    model = SentenceTransformer(EMBED_MODEL)
    client = _get_client(qdrant_path)
    _ensure_collection(client)

    texts = [c["text"] for c in chunks]
    points = []
    for i in range(0, len(texts), 64):
        batch_texts = texts[i:i + 64]
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        for emb, chunk in zip(embeddings, chunks[i:i + 64]):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"text": chunk["text"], **chunk["metadata"]},
            ))

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"  Stored {len(points)} chunks in Qdrant collection '{QDRANT_COLLECTION}'")


def _book_already_embedded(client: QdrantClient, book_title: str) -> bool:
    """Return True if any chunk for this book title exists in Qdrant."""
    result = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter={"must": [{"key": "book", "match": {"value": book_title}}]},
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(result[0]) > 0


def embed_all(parsed_dir: Path, qdrant_path: Path = QDRANT_PATH, force: bool = False) -> None:
    """Load all parsed JSONs, chunk, embed, and store. Skips already-embedded books unless force=True."""
    json_files = sorted(parsed_dir.glob("*.json"))
    if not json_files:
        print("No parsed JSON files found. Run 'python main.py ingest' first.")
        return

    model = SentenceTransformer(EMBED_MODEL)
    client = _get_client(qdrant_path)
    _ensure_collection(client)

    for json_path in json_files:
        book_json = json.loads(json_path.read_text(encoding="utf-8"))
        if not force and _book_already_embedded(client, book_json["book"]):
            print(f"[SKIP] {json_path.name} — already embedded")
            continue

        chunks = chunk_book(book_json)
        if not chunks:
            print(f"[SKIP] {json_path.name} — no chunks produced")
            continue

        points = []
        texts = [c["text"] for c in chunks]
        for i in range(0, len(texts), 64):
            batch_texts = texts[i:i + 64]
            embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
            for emb, chunk in zip(embeddings, chunks[i:i + 64]):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={"text": chunk["text"], **chunk["metadata"]},
                ))

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"[OK] {json_path.name} — {len(points)} chunks embedded")


def _get_book_vector(client: QdrantClient, book_title: str) -> np.ndarray | None:
    """Mean-pool all chunk vectors for a book into a single representative vector."""
    vectors = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="book", match=MatchValue(value=book_title))]),
            limit=256,
            offset=offset,
            with_vectors=True,
            with_payload=False,
        )
        vectors.extend(r.vector for r in results)
        if offset is None:
            break
    if not vectors:
        return None
    arr = np.array(vectors, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else None


def recommend_books(book_title: str, top_k: int = 5, qdrant_path: Path = QDRANT_PATH) -> list[dict]:
    """Return top_k most similar books to book_title by cosine similarity of mean embeddings."""
    client = _get_client(qdrant_path)

    # collect all distinct book titles
    all_books: set[str] = set()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=["book"],
            with_vectors=False,
        )
        all_books.update(r.payload["book"] for r in results if r.payload)
        if offset is None:
            break

    query_vec = _get_book_vector(client, book_title)
    if query_vec is None:
        return []

    scores = []
    for title in all_books:
        if title == book_title:
            continue
        vec = _get_book_vector(client, title)
        if vec is not None:
            scores.append((title, float(np.dot(query_vec, vec))))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [{"book": t, "score": round(s, 4)} for t, s in scores[:top_k]]
