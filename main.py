"""
BookParser CLI — run individual pipeline phases.

Usage:
  python main.py ingest          # parse all PDFs -> JSON (skips existing)
  python main.py ingest --force  # reparse all PDFs
  python main.py embed           # chunk + embed -> Qdrant (skips existing)
  python main.py embed --force   # re-embed all books
  python main.py graph-struct    # structural graph (NetworkX)
  python main.py graph-entity    # stanza NER entity graph
  python main.py cluster           # build topic clusters (UMAP + HDBSCAN)
  python main.py cluster --force   # rebuild even if clusters.json exists
  python main.py flashcards            # generate flashcards for all books
  python main.py flashcards <stem>      # generate for one book (by filename stem)
  python main.py flashcards --force     # regenerate even if already exist
  python main.py serve           # start FastAPI server
  python main.py serve-ui        # start Streamlit UI
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "ingest":
        from src.ingestion.parser import parse_all
        from config import BOOKS_DIR, PARSED_DIR
        parse_all(BOOKS_DIR, PARSED_DIR, force="--force" in sys.argv)

    elif cmd == "embed":
        from src.embedding.embedder import embed_all
        from config import PARSED_DIR, QDRANT_PATH
        embed_all(PARSED_DIR, QDRANT_PATH, force="--force" in sys.argv)

    elif cmd == "graph-struct":
        from src.graph.knowledge_graph import build_all_structural
        from config import PARSED_DIR
        build_all_structural(PARSED_DIR)

    elif cmd == "graph-entity":
        from src.graph.knowledge_graph import build_entity_graph
        from config import PARSED_DIR
        build_entity_graph(PARSED_DIR)

    elif cmd == "cluster":
        from src.clustering.clusterer import build_clusters
        build_clusters(force="--force" in sys.argv)

    elif cmd == "flashcards":
        from src.flashcards.generator import generate_flashcards
        args = [a for a in sys.argv[2:] if a != "--force"]
        book_stem = args[0] if args else None
        generate_flashcards(book_stem=book_stem, force="--force" in sys.argv)

    elif cmd == "ask":
        from src.rag.engine import ask
        query = " ".join(sys.argv[2:])
        result = ask(query)
        for i, c in enumerate(result["citations"], 1):
            print(f"  [{i}] {c}")

    elif cmd == "serve":
        import uvicorn
        from config import API_PORT
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=API_PORT, reload=True)

    elif cmd == "serve-ui":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.runOnSave", "true"])

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
