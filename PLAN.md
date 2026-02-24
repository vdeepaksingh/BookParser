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
             ├─► Knowledge Graph (Neo4j/NetworkX) → Structured Navigation
             └─► Metadata Index        → Filtering, faceted search
```

---

## Phase 1 — Ingestion & Parsing

**Goal:** Extract clean, structured text from PDFs.

### Tools
- `PyMuPDF (fitz)` — primary extraction for text-based PDFs
- `pdfplumber` — fallback for table/layout-heavy pages
- OCR (`pytesseract` + `pdf2image`) — stubbed interface, out of current scope

### Steps
1. Extract text page-by-page with PyMuPDF, preserving font size/weight metadata
2. Use regex + font heuristics to detect structural boundaries (no LLM needed):
   - Book title, Author
   - Part / Chapter / Section / Subsection
4. Output: structured JSON per book
   ```json
   {
     "book": "...", "author": "...",
     "chapters": [
       { "title": "...", "sections": [ { "title": "...", "text": "..." } ] }
     ]
   }
   ```

---

## Phase 2 — Chunking & Embedding

**Goal:** Produce semantically meaningful chunks ready for vector indexing.

### Strategy
- Chunk at the **section level** (not fixed token windows) to preserve context
- Overlap: 10–15% token overlap between adjacent chunks
- Attach metadata to every chunk: `book`, `chapter`, `section`, `page_range`

### Embedding Models (choose one)
| Model | Notes |
|---|---|
| `text-embedding-3-small` (OpenAI) | Best quality/cost ratio, hosted |
| `BAAI/bge-large-en-v1.5` | Best open-source, run locally |
| `sentence-transformers/all-MiniLM-L6-v2` | Lightweight, fast, good baseline |

### Vector Store Options
| Store | Best For |
|---|---|
| **ChromaDB** | Local, zero-infra, great for prototyping |
| **Qdrant** | Production-grade, local or cloud |
| **Weaviate** | Built-in hybrid search (BM25 + vector) |
| **pgvector** | If you already use PostgreSQL |

---

## Phase 3 — Knowledge Graph Construction

**Goal:** Build a navigable graph of concepts, topics, and their relationships.

### Approach A — Structural Graph (fast, deterministic)
- Nodes: `Book → Chapter → Section → Topic`
- Edges: `contains`, `references`, `related_to`
- Built directly from the parsed JSON hierarchy

### Approach B — Semantic Graph (richer, LLM-assisted)
- Use an LLM (e.g., GPT-4o, Llama 3) to extract:
  - Named entities (people, concepts, theories, events)
  - Relationships between entities per chunk
- Merge into a graph: `(ConceptA) -[RELATES_TO]-> (ConceptB)`

### Tools
- `NetworkX` — lightweight, in-memory, good for exploration
- `Neo4j` — production graph DB, Cypher queries, great visualization
- `LlamaIndex` — has built-in `KnowledgeGraphIndex`
- `spaCy` + `Rebel` model — open-source relation extraction

### Output
- Graph queryable by: "What topics are covered in Chapter 3 of Book X?"
- Cross-book concept linking: same concept appearing in multiple books gets merged

---

## Phase 4 — RAG Reference Engine

**Goal:** Answer questions grounded in the book content.

### Pipeline
```
User Query
  └─► Embed query
        └─► Vector similarity search (top-k chunks)
              └─► Optional: re-rank with CrossEncoder
                    └─► LLM generates answer with citations
```

### Components
- **Retriever:** Dense (vector) + optional Sparse (BM25) hybrid
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, fast)
- **LLM:** GPT-4o / Claude 3.5 / Llama 3 (local via Ollama)
- **Framework:** `LlamaIndex` or `LangChain` (LlamaIndex preferred for document-centric RAG)

### Citation Format
Every answer includes: `[Book Title, Chapter X, Section Y, Page Z]`

---

## Phase 5 — Search Interface

**Goal:** Expose the system via a usable interface.

### Options
- **CLI tool** — quick queries from terminal (Phase 5a, minimal)
- **REST API** — FastAPI, enables integration with other tools (Phase 5b)
- **Chat UI** — Streamlit or Gradio frontend (Phase 5c)
- **Obsidian plugin / VS Code extension** — for researchers (Phase 5d, advanced)

---

## Extended Capabilities (Beyond Core)

| Capability | How |
|---|---|
| **Topic clustering** | Embed all chunks → UMAP + HDBSCAN → auto-discover topic clusters |
| **Book recommendation** | Cosine similarity between book embeddings |
| **Concept drift tracking** | Same concept across books from different eras → timeline view |
| **Flashcard / quiz generation** | LLM generates Q&A pairs per section |
| **Podcast/audio summary** | LLM summarizes chapters → TTS (e.g., Coqui, ElevenLabs) |
| **Multi-modal** | Extract and index figures/diagrams via vision models (GPT-4o vision) |
| **Annotation layer** | Store user notes linked to specific chunks in the graph |
| **Incremental ingestion** | Watch folder for new PDFs, auto-process and merge |

---

## Recommended Tech Stack (Pragmatic Starting Point)

```
Parsing:           PyMuPDF (text PDFs); pytesseract stubbed, out of current scope
Chunking:          Custom section-aware splitter
Embeddings:        BAAI/bge-large-en-v1.5 (local)
Vector Store:      ChromaDB
Graph DB:          Neo4j (local via Docker)
Relation Extract:  Rebel (Babelscape/rebel-large) — local, no API cost
RAG LLM:           Ollama + Llama 3.1 8B (local) or GPT-4o (cloud fallback)
RAG Framework:     LlamaIndex
API:               FastAPI
UI:                Streamlit
```

---

## Phased Delivery Milestones

| Milestone | Deliverable |
|---|---|
| M1 | PDF ingestion → structured JSON for all books |
| M2 | Embeddings + ChromaDB vector store live |
| M3 | Basic RAG CLI: ask a question, get cited answer |
| M4 | Structural knowledge graph in Neo4j |
| M5 | Semantic graph: Rebel-extracted triplets merged into Neo4j |
| M6 | FastAPI + Streamlit UI |
| M7 | Extended features (clustering, quizzes, etc.) |

---

## Design Decisions — Resolved

| Decision | Choice | Rationale |
|---|---|---|
| Scale | ~100 books | ChromaDB sufficient; no need for Qdrant yet |
| PDF type | Text-based | PyMuPDF only; OCR pipeline stubbed but out of scope |
| Language | English only | `BAAI/bge-large-en-v1.5` is optimal |
| Graph depth | Full semantic relation extraction | LLM/Rebel-based triplet extraction per chunk |
| Local vs Cloud LLM | TBD | See LLM Usage section below |

---

## LLM Usage — Where and Why

| Phase | LLM Used? | Purpose | Alternatives |
|---|---|---|---|
| Phase 1 — Parsing | No | Regex + heuristics sufficient for text PDFs | — |
| Phase 3 — Knowledge Graph | Yes (core) | Extract entity-relation triplets per chunk: `(A)-[REL]->(B)` | `Rebel` model (local, no API cost) |
| Phase 4 — RAG | Yes (core) | Synthesize retrieved chunks into cited answers | Cannot be avoided |
| Extended — Flashcards/Summaries | Yes (optional) | Generate Q&A pairs, chapter summaries | — |

### Phase 3 LLM Options
- **`Rebel` (recommended for cost)** — HuggingFace `Babelscape/rebel-large`, fine-tuned BART for relation triplet extraction. Runs locally, no API cost. Best for bulk processing 100 books.
- **GPT-4o / Claude 3.5** — Higher quality relations, better at abstract concepts. Use if Rebel quality is insufficient.
- **Llama 3 via Ollama** — Good middle ground: local, free, decent quality.

### Phase 4 LLM Options
- **Ollama + Llama 3.1 8B** — fully local, good quality, zero cost
- **GPT-4o** — best answer quality, pay-per-token
- Recommendation: start with Ollama locally, switch to GPT-4o if answer quality is lacking
