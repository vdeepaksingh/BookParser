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
import json
import re
from pathlib import Path

import fitz  # PyMuPDF


def parse_pdf(pdf_path: Path) -> dict:
    """Extract structured content from a single text-based PDF."""
    doc = fitz.open(str(pdf_path))
    all_blocks = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:  # skip images
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    all_blocks.append({
                        "text": text,
                        "size": round(span["size"], 1),
                        "flags": span["flags"],  # bold=16, italic=2
                        "page": page_num,
                    })

    doc.close()

    classified = _detect_headings(all_blocks)
    book_title, author = _extract_title_author(classified)
    if book_title == "Unknown":
        book_title = pdf_path.stem
    chapters = _build_structure(classified)

    return {
        "book": book_title,
        "author": author,
        "source_file": pdf_path.name,
        "chapters": chapters,
    }


def parse_all(books_dir: Path, output_dir: Path) -> list[Path]:
    """Parse all PDFs in books_dir, write one JSON per book to output_dir.
    Returns list of written JSON paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pdf_path in sorted(books_dir.glob("*.pdf")):
        try:
            result = parse_pdf(pdf_path)
            out_path = output_dir / (pdf_path.stem + ".json")
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(out_path)
            print(f"[OK] {pdf_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"[ERR] {pdf_path.name}: {e}")
    return written


def _detect_headings(blocks: list[dict]) -> list[dict]:
    """Classify each block as h1/h2/h3/body using font size + frequency heuristics."""
    from collections import Counter

    sizes = [b["size"] for b in blocks]
    if not sizes:
        return blocks

    counts = Counter(sizes)
    body_size = sorted(sizes)[len(sizes) // 2]  # median = dominant body size

    # Only consider sizes that appear >= 3 times AND are above body size as headings
    # This filters out decorative/cover one-off large fonts
    heading_sizes = sorted(
        [s for s, c in counts.items() if s > body_size and c >= 3],
        reverse=True
    )[:3]  # at most 3 heading levels

    size_to_level = {s: f"h{i + 1}" for i, s in enumerate(heading_sizes)}

    for block in blocks:
        s = block["size"]
        is_bold = bool(block["flags"] & 16)
        if s in size_to_level:
            block["level"] = size_to_level[s]
        elif is_bold and s > body_size and counts[s] >= 3:
            block["level"] = "h3"
        else:
            block["level"] = "body"

    return blocks


def _extract_title_author(blocks: list[dict]) -> tuple[str, str]:
    """Best-effort title/author from first-page h1/h2 blocks."""
    first_page = [b for b in blocks if b["page"] == 0]
    h1_blocks = [b["text"] for b in first_page if b.get("level") == "h1"]
    h2_blocks = [b["text"] for b in first_page if b.get("level") == "h2"]

    title = h1_blocks[0] if h1_blocks else "Unknown"
    # Author heuristic: h2 block containing common author keywords or second h1
    author = "Unknown"
    for text in h2_blocks:
        if re.search(r"(by |author|edited)", text, re.IGNORECASE):
            author = re.sub(r"(?i)^by\s+", "", text).strip()
            break
    if author == "Unknown" and len(h1_blocks) > 1:
        author = h1_blocks[1]

    return title, author


def _merge_heading_spans(blocks: list[dict]) -> list[dict]:
    """Merge consecutive same-level heading spans on the same page into one block."""
    merged = []
    for block in blocks:
        if (
            merged
            and block["level"] == merged[-1]["level"]
            and block["level"] != "body"
            and block["page"] == merged[-1]["page"]
        ):
            merged[-1]["text"] += " " + block["text"]
        else:
            merged.append(dict(block))
    return merged


# Headings that indicate front matter rather than real chapters
_FRONT_MATTER_TITLES = re.compile(
    r"^(contents|table of contents|copyright|dedication|acknowledgements?"
    r"|preface|foreword|introduction|about (the )?author|publisher.?s note|index)$",
    re.IGNORECASE,
)


def _front_matter_page_cutoff(blocks: list[dict]) -> int:
    """Return the page number where real chapter content begins.
    Skips pages whose only headings are front-matter titles."""
    heading_pages: dict[int, list[str]] = {}
    for b in blocks:
        if b.get("level") in ("h1", "h2"):
            heading_pages.setdefault(b["page"], []).append(b["text"])

    for page in sorted(heading_pages):
        titles = heading_pages[page]
        if any(not _FRONT_MATTER_TITLES.match(t) for t in titles):
            return page  # first page with a non-front-matter heading
    return 0  # no front matter detected, start from page 0


def _build_structure(blocks: list[dict]) -> list[dict]:
    """Assemble blocks into chapter -> section tree."""
    blocks = _merge_heading_spans(blocks)
    cutoff = _front_matter_page_cutoff(blocks)
    blocks = [b for b in blocks if b["page"] >= cutoff]

    # If h1 appears only once (cover title bled through), promote h2 -> h1, h3 -> h2
    h1_count = sum(1 for b in blocks if b["level"] == "h1")
    if h1_count <= 1:
        level_map = {"h1": "body", "h2": "h1", "h3": "h2"}
        for b in blocks:
            b["level"] = level_map.get(b["level"], b["level"])

    chapters = []
    current_chapter = None
    current_section = None
    chapter_idx = 0

    for block in blocks:
        level = block.get("level", "body")
        text = block["text"]
        page = block["page"]

        if level == "h1":
            if current_section and current_chapter:
                current_chapter["sections"].append(current_section)
            if current_chapter:
                chapters.append(current_chapter)
            chapter_idx += 1
            current_chapter = {"title": text, "index": chapter_idx, "sections": []}
            current_section = {"title": "__intro__", "text": "", "page_start": page, "page_end": page}

        elif level in ("h2", "h3"):
            if current_section and current_chapter:
                if current_section["text"].strip():
                    current_chapter["sections"].append(current_section)
            current_section = {"title": text, "text": "", "page_start": page, "page_end": page}
            if current_chapter is None:
                chapter_idx += 1
                current_chapter = {"title": "Preamble", "index": chapter_idx, "sections": []}

        else:  # body
            if current_section is None:
                current_section = {"title": "__intro__", "text": "", "page_start": page, "page_end": page}
            if current_chapter is None:
                chapter_idx += 1
                current_chapter = {"title": "Preamble", "index": chapter_idx, "sections": []}
            current_section["text"] += " " + text
            current_section["page_end"] = page

    # flush remaining
    if current_section and current_chapter:
        if current_section["text"].strip():
            current_chapter["sections"].append(current_section)
    if current_chapter:
        chapters.append(current_chapter)

    return chapters


def _ocr_fallback(pdf_path: Path) -> dict:
    """Stub — OCR pipeline for scanned PDFs. Not implemented in current scope."""
    raise NotImplementedError("OCR support is out of current scope.")
