"""Phase 5 — FastAPI REST API."""
from fastapi import FastAPI

app = FastAPI(title="BookParser API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(query: str):
    """RAG query endpoint. Returns answer + citations."""
    raise NotImplementedError


@app.get("/search")
def search_endpoint(query: str, top_k: int = 5):
    """Semantic search — returns top-k matching chunks with metadata."""
    raise NotImplementedError


@app.get("/graph/entity/{name}")
def get_entity(name: str):
    """Return all relations for a given entity from the knowledge graph."""
    raise NotImplementedError


@app.get("/graph/book/{book_title}")
def get_book_graph(book_title: str):
    """Return structural graph (chapters/sections) for a book."""
    raise NotImplementedError
