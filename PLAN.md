# BookParser — ML Knowledge Base System: Plan

## Problem Summary

Convert a local collection of PDF books into a multi-purpose intelligent knowledge system that supports:
- **Semantic search** across diverse topics
- **Knowledge graph** with structured hierarchy (subject → chapter → topic)
- **Reference/RAG engine** for fact retrieval and contextual grounding

---

## Architecture Overview

```
PDFs
 └─► Ingestion & Parsing
       └─► Chunking & Embedding
             ├─► Vector Store          → Semantic Search + RAG
             ├─► Knowledge Graph (NetworkX) → Structured Navigation
             └─► Metadata Index        → Filtering, faceted search
```

---

## Phase 1 — Ingestion & Parsing ✅

**Goal:** Extract clean, structured text from PDFs.

### Tools
- `PyMuPDF (fitz)` — primary extraction for text-based PDFs
- OCR (`pytesseract` + `pdf2image`) — stubbed interface, out of current scope

### Steps
1. Extract text page-by-page with PyMuPDF, preserving font size/weight metadata
2. Use regex + font heuristics to detect structural boundaries (no LLM needed):
   - Book title, Author
   - Part / Chapter / Section / Subsection
3. Output: structured JSON per book
   ```json
   {
     "book": "...", "author": "...",
     "chapters": [
       { "title": "...", "sections": [ { "title": "...", "text": "..." } ] }
     ]
   }
   ```

---

## Phase 2 — Chunking & Embedding ✅

**Goal:** Produce semantically meaningful chunks ready for vector indexing.

### Strategy
- Chunk at the **section level** (not fixed token windows) to preserve context
- Overlap: 10–15% token overlap between adjacent chunks
- Attach metadata to every chunk: `book`, `chapter`, `section`, `page_range`

### Stack
- **Embeddings:** `BAAI/bge-large-en-v1.5` (local)
- **Vector Store:** Qdrant (local file mode)

---

## Phase 3 — Knowledge Graph Construction ✅

**Goal:** Build a navigable graph of concepts, topics, and their relationships.

### Approach A — Structural Graph ✅
- Nodes: `Book → Chapter → Section`
- Edges: `contains`
- Built directly from parsed JSON hierarchy
- `__intro__` sections skipped
- Always starts fresh (`nx.DiGraph()`) on `graph-struct`

### Approach B — Entity Graph ✅
- NER via `stanza` (replaces spaCy — incompatible with Python 3.14 due to pydantic v1)
- Filtered entity types: `PERSON, ORG, GPE, WORK_OF_ART, EVENT, LAW, PRODUCT`
- Entity nodes linked to sections via `MENTIONS` / `APPEARS_IN` edges
- Loads existing structural graph and enriches it (`graph-entity`)
- Physics/abstract concepts (e.g. Vector, Momentum) are NOT named entities — use Search tab

### Tools
- `NetworkX` — in-memory graph, persisted to `data/graph.gpickle`
- `stanza` — NER (works on Python 3.14)

### Config
- `GRAPH_PATH`, `GRAPH_NER_TEXT_CAP=2000`, `GRAPH_ENTITY_CAP=50` in `config.py`

### Run Order
```
python main.py graph-struct   # always starts fresh
python main.py graph-entity   # enriches existing graph
```

---

## Phase 4 — RAG Reference Engine ✅

**Goal:** Answer questions grounded in the book content.

### Pipeline
```
User Query
  └─► Embed query
        └─► Vector similarity search (top-k chunks)
              └─► Re-rank with CrossEncoder (RELEVANCE_THRESHOLD, RERANK_MIN_SCORE)
                    └─► LLM generates answer with citations (streaming)
```

### Components
- **Retriever:** Dense vector search via Qdrant
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- **LLM:** Ollama + `phi3:mini` (local)
  - Note: `exit status 2` crash = corrupted model blob → fix with `ollama pull phi3:mini`
- **Config:** All thresholds/model names in `config.py` — no hardcoded values

### Citation Format
Every answer includes: `[Book Title, Chapter X, Section Y]`

---

## Phase 5 — Search Interface ✅

**Goal:** Expose the system via a usable interface.

### Completed
- ✅ **CLI** — `python main.py ask "query"` with cited answers
- ✅ **REST API** — FastAPI (`src/api/app.py`)
  - `POST /ask` — RAG query, returns answer + citations
  - `GET /search` — hybrid retrieval, returns top-k chunks
  - `GET /graph/books` — returns full graph data
  - `GET /graph/entity/{name}` — entity lookup
  - `GET /section` — reads full section text from parsed JSON
  - Model preloading via `asynccontextmanager` lifespan (replaces deprecated `@app.on_event`)
  - Request logging with timing
- ✅ **Chat UI** — Streamlit (`src/ui/app.py`)
  - Ask tab: question → cited answer, Enter key submits via `st.form`
  - Search tab: semantic search with top-k slider
  - Graph tab:
    - Book Structure: pyvis interactive graph (`cdn_resources="in_line"` — embeds vis.js, no CDN) + section browser with nested expanders
    - Entity Search: expandable sections showing full text via `/section`
  - `python main.py serve-ui` with hot reload (`--server.runOnSave true`)

### Notes
- pyvis `cdn_resources="in_line"` (underscore) required — `"cdn_resources"` with CDN triggers Fivetran webhook
- Streamlit uses WebSocket (`ws://localhost:8501`), not HTTP — API calls invisible to browser network tab
- Qdrant: single client instance must be reused (file lock)

---

## Project Infrastructure ✅

- **`pyproject.toml`** — `pip install -e .` registers project root via `.pth` file; eliminates all `sys.path.insert` hacks
- **`config.py`** — single source of truth for all config: model names, paths, thresholds, caps, ports
- **`requirements.txt`** — `networkx>=3.3`, `stanza>=1.9.0`, `pyvis>=0.3.2`; removed neo4j, transformers, torch

---

## Phased Delivery Milestones

| Milestone | Deliverable |
|---|---|
| M1 | PDF ingestion → structured JSON for all books ✅ |
| M2 | Embeddings + Qdrant vector store live ✅ |
| M3 | Basic RAG CLI: ask a question, get cited answer ✅ |
| M4 | Structural knowledge graph (NetworkX) ✅ |
| M5 | Entity graph: stanza NER merged into NetworkX ✅ |
| M6 | FastAPI + Streamlit UI with Graph tab ✅ |
| M7 | Incremental ingestion — skip already-parsed/embedded books ✅ |
| M8 | Flashcard / quiz generation — CLI + API + UI tab ✅ |
| M9 | Book recommendation — cosine similarity over book embeddings |
| M10 | Topic clustering — UMAP + HDBSCAN over all chunk embeddings |

---

## Phase 7 — Incremental Ingestion

**Goal:** Skip already-parsed/embedded books so re-running `ingest` or `embed` is safe and fast.

### Approach
- `ingest`: skip PDF if `data/parsed/<stem>.json` already exists
- `embed`: skip book if all its chunks are already in Qdrant (check by book name metadata)
- CLI flag `--force` to override and reprocess

---

## Phase 8 — Flashcard / Quiz Generation

**Goal:** Generate Q&A pairs per section using the local LLM.

### Approach
- New CLI command: `python main.py flashcards <book>`
- Prompt Ollama to produce N question/answer pairs per section
- Output: `data/flashcards/<book>.json`
- New API endpoint: `GET /flashcards?book=...`
- New UI tab: Flashcards — pick book/chapter, browse Q&A cards

---

## Phase 9 — Book Recommendation

**Goal:** Given a book, find the most similar books in the library.

### Approach
- Aggregate chunk embeddings per book (mean pooling)
- Cosine similarity between book vectors
- New API endpoint: `GET /recommend?book=...&top_k=5`
- Surfaced in UI: sidebar or dedicated tab

---

## Phase 10 — Topic Clustering

**Goal:** Auto-discover thematic clusters across all chunks.

### Approach
- Fetch all chunk embeddings from Qdrant
- Reduce dimensions: UMAP
- Cluster: HDBSCAN
- Label clusters: LLM names each cluster from its top terms
- Output: `data/clusters.json` + interactive scatter plot in UI
- New deps: `umap-learn`, `hdbscan`

---

## Extended Capabilities (Deferred)

| Capability | How |
|---|---|
| **Concept drift tracking** | Same concept across books from different eras → timeline view |
| **Podcast/audio summary** | LLM summarizes chapters → TTS (e.g., Coqui, ElevenLabs) |
| **Multi-modal** | Extract and index figures/diagrams via vision models |
| **Annotation layer** | Store user notes linked to specific chunks in the graph |

---

## Design Decisions — Resolved

| Decision | Choice | Rationale |
|---|---|---|
| PDF type | Text-based | PyMuPDF only; pdfplumber planned as fallback but never needed — all books are text-based PDFs |
| Language | English only | `BAAI/bge-large-en-v1.5` is optimal |
| Graph DB | NetworkX (not Neo4j) | Zero infra, sufficient for current scale |
| NER | stanza (not spaCy) | spaCy incompatible with Python 3.14 (pydantic v1) |
| Relation extraction | NER only (not Rebel) | Rebel/transformers removed; stanza sufficient |
| Local LLM | Ollama + phi3:mini | Fully local, zero cost |
| Graph persistence | `data/graph.gpickle` | Survives restarts; struct always fresh, entity enriches |
