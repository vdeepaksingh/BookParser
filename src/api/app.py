"""Phase 5 — FastAPI REST API."""
import json
import logging
import time
from contextlib import asynccontextmanager
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.rag.engine import _get_embed_model, _get_rerank_model
    log.info("Pre-loading embedding and rerank models...")
    _get_embed_model()
    _get_rerank_model()
    log.info("Models ready.")
    yield

app = FastAPI(title="BookParser API", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(body: AskRequest):
    """RAG query endpoint. Returns answer + citations."""
    log.info("POST /ask query=%r", body.query)
    t0 = time.time()
    from src.rag.engine import ask
    try:
        result = ask(body.query)
        log.info("POST /ask completed in %.1fs, %d citations", time.time() - t0, len(result.get("citations", [])))
        return result
    except Exception as e:
        log.error("POST /ask failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search_endpoint(query: str, top_k: int = 5):
    """Semantic search — returns top-k matching chunks with metadata."""
    log.info("GET /search query=%r top_k=%d", query, top_k)
    t0 = time.time()
    from src.rag.engine import retrieve, rerank
    try:
        chunks = retrieve(query, top_k=top_k * 3)
        chunks = rerank(query, chunks, top_k=top_k)
        log.info("GET /search completed in %.1fs, %d results", time.time() - t0, len(chunks))
        return {"results": [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]}
    except Exception as e:
        log.error("GET /search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/section")
def get_section(book: str, chapter: str, section: str):
    """Return full text of a specific section from parsed JSON."""
    from config import PARSED_DIR
    for path in PARSED_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("book") != book:
            continue
        for ch in data.get("chapters", []):
            if ch["title"] != chapter:
                continue
            for sec in ch.get("sections", []):
                if sec["title"] == section:
                    return {"book": book, "chapter": chapter, "section": section,
                            "text": sec["text"], "page_start": sec.get("page_start"), "page_end": sec.get("page_end")}
    raise HTTPException(status_code=404, detail="Section not found")


@app.get("/flashcards")
def flashcards_endpoint(book: str):
    """Return generated flashcards for a book (by title)."""
    from src.flashcards.generator import list_flashcard_books, load_flashcards
    from pathlib import Path
    from config import FLASHCARDS_DIR
    # match by title
    for path in sorted(FLASHCARDS_DIR.glob("*.json")):
        try:
            import json as _json
            data = _json.loads(path.read_text(encoding="utf-8"))
            if data.get("book") == book:
                return data
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Flashcards not found for this book. Run flashcards command first.")


@app.get("/flashcards/books")
def flashcard_books_endpoint():
    """List books that have generated flashcards."""
    from src.flashcards.generator import list_flashcard_books
    return {"books": list_flashcard_books()}



def list_books():
    from src.graph.knowledge_graph import _load_graph
    g = _load_graph()
    return {"books": [n for n, d in g.nodes(data=True) if d.get("type") == "book"]}


@app.get("/graph/entity/{name}")
def get_entity_endpoint(name: str):
    from src.graph.knowledge_graph import get_entity
    result = get_entity(name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/graph/book/{book_title}")
def get_book_graph_endpoint(book_title: str):
    from src.graph.knowledge_graph import get_book_graph
    result = get_book_graph(book_title)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
