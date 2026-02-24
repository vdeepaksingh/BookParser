"""
Phase 1 — PDF ingestion and structural parsing.

Output schema per book:
{
  "book": str, "author": str, "source_file": str,
  "chapters": [
    { "title": str, "index": int,
      "sections": [ { "title": str, "text": str, "page_start": int, "page_end": int } ]
    }
  ]
}
"""
from pathlib import Path


def parse_pdf(pdf_path: Path) -> dict:
    """Extract structured content from a single text-based PDF."""
    raise NotImplementedError


def parse_all(books_dir: Path, output_dir: Path) -> list[Path]:
    """Parse all PDFs in books_dir, write one JSON per book to output_dir.
    Returns list of written JSON paths."""
    raise NotImplementedError


def _detect_headings(page_blocks: list[dict]) -> list[dict]:
    """Heuristic: classify blocks as heading/body using font size + weight."""
    raise NotImplementedError


def _ocr_fallback(pdf_path: Path) -> dict:
    """Stub — OCR pipeline for scanned PDFs. Not implemented in current scope."""
    raise NotImplementedError("OCR support is out of current scope.")
