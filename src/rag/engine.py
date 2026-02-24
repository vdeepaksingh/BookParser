"""
Phase 4 — RAG reference engine.

Pipeline:
  query -> embed (BGE) -> Qdrant top-k -> CrossEncoder rerank -> Ollama LLM -> cited answer
"""
import sys
from pathlib import Path

import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    EMBED_MODEL, QDRANT_PATH, QDRANT_COLLECTION,
    OLLAMA_MODEL, OLLAMA_BASE_URL, TOP_K_RETRIEVAL
)

_embed_model = None
_rerank_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _rerank_model


def _load_all_chunks(client: QdrantClient) -> list[dict]:
    """Load all chunks from Qdrant for BM25 indexing."""
    results, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        results.extend(batch)
        if offset is None:
            break
    return [{"text": r.payload["text"], "metadata": {k: v for k, v in r.payload.items() if k != "text"}} for r in results]


def _rrf(dense: list[dict], sparse: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion of two ranked lists."""
    scores: dict[int, float] = {}
    index: dict[int, dict] = {}
    for rank, chunk in enumerate(dense):
        uid = id(chunk)
        scores[uid] = scores.get(uid, 0) + 1 / (k + rank + 1)
        index[uid] = chunk
    for rank, chunk in enumerate(sparse):
        uid = id(chunk)
        scores[uid] = scores.get(uid, 0) + 1 / (k + rank + 1)
        index[uid] = chunk
    return [index[uid] for uid in sorted(scores, key=scores.__getitem__, reverse=True)]


def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL * 3) -> list[dict]:
    """Hybrid retrieval: dense (BGE) + sparse (BM25) fused with RRF."""
    # Dense
    model = _get_embed_model()
    query_vec = model.encode(query, normalize_embeddings=True).tolist()
    client = QdrantClient(path=str(QDRANT_PATH))
    dense_results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points
    dense = [{"text": r.payload["text"], "metadata": {k: v for k, v in r.payload.items() if k != "text"}, "score": r.score} for r in dense_results]

    # Sparse (BM25)
    all_chunks = _load_all_chunks(client)
    tokenized = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse = [all_chunks[i] for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]]

    return _rrf(dense, sparse)[:top_k]


def rerank(query: str, chunks: list[dict], top_k: int = TOP_K_RETRIEVAL, min_score: float = 0.0) -> list[dict]:
    """Re-rank chunks using CrossEncoder, return top_k."""
    model = _get_rerank_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    seen, result = set(), []
    for score, c in ranked:
        if score < min_score:
            break
        key = (c["metadata"].get("chapter_title"), c["metadata"].get("page_start"))
        if key not in seen:
            seen.add(key)
            result.append(c)
        if len(result) == top_k:
            break
    return result


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Send query + retrieved context to Ollama. Returns {answer, citations}."""
    context_parts = []
    citations = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk["metadata"]
        section = m.get('section_title', '?')
        if section == '__intro__':
            section = '-'
        citation = f"[{m.get('book', '?')}, {m.get('chapter_title', '?')}, {section}, p.{m.get('page_start', '?')}]"
        citations.append(citation)
        context_parts.append(f"[{i}] {chunk['text'][:500]}\nSource: {citation}")

    context = "\n\n".join(context_parts)
    prompt = (
        f"Using ONLY the numbered sources below, answer in 3-5 sentences. "
        f"Cite only as [1], [2], etc. Do not invent sources, add extra references, or ask follow-up questions.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    answer_parts = []
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "options": {"stop": ["Follow-up", "Follow up", "Question:"]}},
        timeout=(10, 300),
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                token = __import__("json").loads(line).get("response", "")
                print(token, end="", flush=True)
                answer_parts.append(token)
    print()

    import re
    answer = "".join(answer_parts).strip()
    # Strip full and partial inline citations copied from context
    for citation in citations:
        answer = answer.replace(citation, '')
    answer = re.sub(r'\[[^\]]{30,}\]', '', answer)
    # Strip invalid numeric citations
    valid = set(str(i) for i in range(1, len(chunks) + 1))
    answer = re.sub(r'\[(\d+)\]', lambda m: f'[{m.group(1)}]' if m.group(1) in valid else '', answer)
    answer = re.sub(r'\[\][-–]?\[\]|\[\]-\[\]', '', answer).strip()

    return {"answer": answer, "citations": citations}


def ask(query: str) -> dict:
    """End-to-end: query -> cited answer."""
    chunks = retrieve(query)
    chunks = rerank(query, chunks, min_score=-1.0)
    return generate_answer(query, chunks)
