"""Phase 5 — FastAPI REST API."""
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).parents[2]))

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


@app.get("/graph/entity/{name}")
def get_entity(name: str):
    raise HTTPException(status_code=501, detail="Phase 3 not yet implemented")


@app.get("/graph/book/{book_title}")
def get_book_graph(book_title: str):
    raise HTTPException(status_code=501, detail="Phase 3 not yet implemented")
