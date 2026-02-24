"""
Phase 3 — Knowledge graph construction in Neo4j.

Two layers:
  A) Structural: Book -> Chapter -> Section nodes (deterministic)
  B) Semantic:   Entity -[RELATION]-> Entity triplets via Rebel model
"""
from pathlib import Path


# --- Structural graph (M4) ---

def build_structural_graph(book_json: dict, driver) -> None:
    """Create Book/Chapter/Section nodes and CONTAINS edges in Neo4j."""
    raise NotImplementedError


def build_all_structural(parsed_dir: Path, driver) -> None:
    """Load all parsed JSONs and populate structural graph. Entry point for M4."""
    raise NotImplementedError


# --- Semantic graph (M5) ---

def extract_triplets(text: str, model, tokenizer) -> list[tuple[str, str, str]]:
    """Run Rebel model on text, return list of (subject, relation, object) tuples."""
    raise NotImplementedError


def merge_triplets_to_graph(triplets: list[tuple], chunk_metadata: dict, driver) -> None:
    """Upsert Entity nodes and typed RELATION edges into Neo4j.
    Merges duplicate entities across books by normalized name."""
    raise NotImplementedError


def build_semantic_graph(parsed_dir: Path, driver) -> None:
    """Extract triplets from all chunks and merge into Neo4j. Entry point for M5."""
    raise NotImplementedError


# --- Neo4j helpers ---

def get_driver(uri: str, user: str, password: str):
    """Return a Neo4j driver instance."""
    raise NotImplementedError


def close_driver(driver) -> None:
    raise NotImplementedError
