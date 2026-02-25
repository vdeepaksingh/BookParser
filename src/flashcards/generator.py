"""
Phase 8 — Flashcard / quiz generation.

Output schema per book:
{
  "book": str,
  "chapters": [
    {
      "chapter": str,
      "sections": [
        {
          "section": str,
          "cards": [ { "question": str, "answer": str } ]
        }
      ]
    }
  ]
}
"""
import json
import re
import requests
from pathlib import Path

from config import (
    PARSED_DIR, FLASHCARDS_DIR,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    FLASHCARDS_PER_SECTION, CHUNK_CONTEXT_CAP,
)

_PROMPT = """\
Based on the following text, generate exactly {n} question-and-answer pairs for study flashcards.
Respond ONLY with a JSON array, no explanation:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Text:
{text}
"""


def _parse_cards(raw: str) -> list[dict]:
    """Extract JSON array from LLM response, tolerating surrounding text."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        cards = json.loads(match.group())
        return [c for c in cards if "question" in c and "answer" in c]
    except json.JSONDecodeError:
        return []


def _generate_for_section(text: str, n: int = FLASHCARDS_PER_SECTION) -> list[dict]:
    """Call Ollama to generate n Q&A pairs for a section."""
    prompt = _PROMPT.format(n=n, text=text[:CHUNK_CONTEXT_CAP * 2])
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return _parse_cards(resp.json().get("response", ""))
    except Exception as e:
        print(f"    [WARN] LLM call failed: {e}")
        return []


def generate_flashcards(book_stem: str | None = None, force: bool = False) -> list[Path]:
    """Generate flashcards for all books (or a specific one by stem).
    Skips existing output unless force=True. Returns list of written paths."""
    FLASHCARDS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(PARSED_DIR.glob("*.json"))
    if book_stem:
        json_files = [f for f in json_files if f.stem.lower() == book_stem.lower()]
    if not json_files:
        print("No matching parsed books found.")
        return []

    written = []
    for json_path in json_files:
        out_path = FLASHCARDS_DIR / json_path.name
        if not force and out_path.exists():
            print(f"[SKIP] {json_path.name} — flashcards already exist")
            continue

        book_json = json.loads(json_path.read_text(encoding="utf-8"))
        book_title = book_json["book"]
        print(f"[GEN] {book_title}")

        result = {"book": book_title, "chapters": []}
        for chapter in book_json["chapters"]:
            ch_entry = {"chapter": chapter["title"], "sections": []}
            for section in chapter["sections"]:
                text = section["text"].strip()
                if not text:
                    continue
                print(f"  {chapter['title']} › {section['title']}")
                cards = _generate_for_section(text)
                if cards:
                    ch_entry["sections"].append({
                        "section": section["title"],
                        "cards": cards,
                    })
            if ch_entry["sections"]:
                result["chapters"].append(ch_entry)

        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {out_path.name}")
        written.append(out_path)

    return written


def load_flashcards(book_stem: str) -> dict | None:
    """Load flashcard JSON for a book by stem. Returns None if not found."""
    for path in FLASHCARDS_DIR.glob("*.json"):
        if path.stem.lower() == book_stem.lower():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def list_flashcard_books() -> list[str]:
    """Return list of book titles that have generated flashcards."""
    books = []
    for path in sorted(FLASHCARDS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            books.append(data["book"])
        except Exception:
            pass
    return books
