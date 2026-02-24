"""
BookParser CLI — run individual pipeline phases.

Usage:
  python main.py ingest          # M1: parse all PDFs -> JSON
  python main.py embed           # M2: chunk + embed -> ChromaDB
  python main.py graph-struct    # M4: structural graph -> Neo4j
  python main.py graph-semantic  # M5: Rebel triplets -> Neo4j
  python main.py ask "query"     # M3: RAG query
  python main.py serve           # M6: start FastAPI server
"""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "ingest":
        from src.ingestion.parser import parse_all
        from config import BOOKS_DIR, PARSED_DIR
        parse_all(BOOKS_DIR, PARSED_DIR)

    elif cmd == "embed":
        from src.embedding.embedder import embed_all
        from config import PARSED_DIR, QDRANT_PATH
        embed_all(PARSED_DIR, QDRANT_PATH)

    elif cmd == "graph-struct":
        from src.graph.knowledge_graph import build_all_structural, get_driver, close_driver
        from config import PARSED_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        driver = get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        build_all_structural(PARSED_DIR, driver)
        close_driver(driver)

    elif cmd == "graph-semantic":
        from src.graph.knowledge_graph import build_semantic_graph, get_driver, close_driver
        from config import PARSED_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        driver = get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        build_semantic_graph(PARSED_DIR, driver)
        close_driver(driver)

    elif cmd == "ask":
        from src.rag.engine import ask
        query = " ".join(sys.argv[2:])
        result = ask(query)
        print(result["answer"])
        for c in result["citations"]:
            print(" ", c)

    elif cmd == "serve":
        import uvicorn
        uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
