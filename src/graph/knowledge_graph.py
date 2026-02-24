"""
Phase 3 — Knowledge graph construction.

Two layers:
  A) Structural: Book -> Chapter -> Section nodes (deterministic, from parsed JSON)
  B) Entity:     stanza NER -> Entity nodes linked to sections + cross-book APPEARS_IN edges
"""
import json
import pickle
from pathlib import Path

import networkx as nx
import stanza

from config import GRAPH_PATH, GRAPH_NER_TEXT_CAP, GRAPH_ENTITY_CAP

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        stanza.download("en", processors="tokenize,ner", verbose=False)
        _nlp = stanza.Pipeline("en", processors="tokenize,ner", verbose=False)
    return _nlp


def _load_graph() -> nx.DiGraph:
    if GRAPH_PATH.exists():
        with open(GRAPH_PATH, "rb") as f:
            return pickle.load(f)
    return nx.DiGraph()


def _save_graph(g: nx.DiGraph) -> None:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(g, f)


# --- Structural graph (M4) ---

def build_structural_graph(book_json: dict, g: nx.DiGraph) -> nx.DiGraph:
    """Add Book/Chapter/Section nodes and CONTAINS edges for one book."""
    book = book_json["book"]
    g.add_node(book, type="book", author=book_json.get("author", ""))

    for ch in book_json.get("chapters", []):
        ch_id = f"{book}::{ch['title']}"
        g.add_node(ch_id, type="chapter", title=ch["title"], book=book)
        g.add_edge(book, ch_id, rel="CONTAINS")

        for sec in ch.get("sections", []):
            if sec["title"] == "__intro__":
                continue
            sec_id = f"{ch_id}::{sec['title']}"
            g.add_node(sec_id, type="section", title=sec["title"],
                       book=book, chapter=ch["title"],
                       page_start=sec.get("page_start"), page_end=sec.get("page_end"))
            g.add_edge(ch_id, sec_id, rel="CONTAINS")

    return g


def build_all_structural(parsed_dir: Path) -> nx.DiGraph:
    """Build structural graph from all parsed JSONs."""
    g = nx.DiGraph()  # always start fresh
    for path in sorted(parsed_dir.glob("*.json")):
        book_json = json.loads(path.read_text(encoding="utf-8"))
        build_structural_graph(book_json, g)
        print(f"  structural: {book_json['book']}")
    _save_graph(g)
    print(f"Graph saved: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


# --- Entity graph (M5) ---

def build_entity_graph(parsed_dir: Path) -> nx.DiGraph:
    """Run spaCy NER over all sections, add Entity nodes + MENTIONS/APPEARS_IN edges."""
    g = _load_graph()
    nlp = _get_nlp()
    entity_books: dict[str, set] = {}  # entity_id -> set of books

    for path in sorted(parsed_dir.glob("*.json")):
        book_json = json.loads(path.read_text(encoding="utf-8"))
        book = book_json["book"]
        print(f"  entities: {book}")

        for ch in book_json.get("chapters", []):
            for sec in ch.get("sections", []):
                sec_id = f"{book}::{ch['title']}::{sec['title']}"
                doc = nlp(sec["text"][:GRAPH_NER_TEXT_CAP])
                for ent in doc.entities:
                    if ent.type not in {"PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT", "LAW", "PRODUCT"}:
                        continue
                    ent_id = f"entity::{ent.type}::{ent.text.strip().lower()}"
                    if not g.has_node(ent_id):
                        g.add_node(ent_id, type="entity", label=ent.type, name=ent.text.strip())
                    if g.has_node(sec_id):
                        g.add_edge(sec_id, ent_id, rel="MENTIONS")
                    entity_books.setdefault(ent_id, set()).add(book)

    # Cross-book APPEARS_IN edges
    for ent_id, books in entity_books.items():
        for book in books:
            if g.has_node(book):
                g.add_edge(ent_id, book, rel="APPEARS_IN")

    _save_graph(g)
    print(f"Graph saved: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


# --- Query helpers (used by API) ---

def get_book_graph(book_title: str) -> dict:
    """Return chapters and top entities for a book."""
    g = _load_graph()
    if book_title not in g:
        return {"error": f"Book '{book_title}' not found in graph"}

    chapters = [
        {"chapter": g.nodes[n]["title"],
         "sections": [g.nodes[s]["title"] for s in g.successors(n) if g.nodes[s].get("type") == "section"]}
        for n in g.successors(book_title) if g.nodes[n].get("type") == "chapter"
    ]
    entities = [
        {"name": g.nodes[n]["name"], "label": g.nodes[n]["label"]}
        for n in g.predecessors(book_title) if g.nodes[n].get("type") == "entity"
    ]
    return {"book": book_title, "chapters": chapters, "entities": entities[:GRAPH_ENTITY_CAP]}


def get_entity(name: str) -> dict:
    """Return all books and sections that mention an entity."""
    g = _load_graph()
    ent_id = next((n for n in g.nodes if g.nodes[n].get("type") == "entity"
                   and g.nodes[n].get("name", "").lower() == name.lower()), None)
    if not ent_id:
        return {"error": f"Entity '{name}' not found"}

    books = [g.nodes[n]["book"] if "book" in g.nodes[n] else n
             for n in g.successors(ent_id) if g.nodes[n].get("type") in {"book", None}]
    sections = [{"section": g.nodes[n]["title"], "book": g.nodes[n].get("book"),
                 "chapter": g.nodes[n].get("chapter")}
                for n in g.predecessors(ent_id) if g.nodes[n].get("type") == "section"]
    return {"entity": name, "appears_in_books": list(set(books)), "sections": sections}
