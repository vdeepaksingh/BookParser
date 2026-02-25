"""
Phase 10 — Topic clustering over all chunk embeddings.

Pipeline:
  Qdrant embeddings -> UMAP (2D) -> HDBSCAN clusters -> LLM cluster labels
  Output: data/clusters.json
"""
import json
import requests
import numpy as np

from config import (
    QDRANT_PATH, QDRANT_COLLECTION, CLUSTERS_PATH,
    UMAP_N_COMPONENTS, HDBSCAN_MIN_CLUSTER_SIZE,
    OLLAMA_BASE_URL, OLLAMA_MODEL, CHUNK_CONTEXT_CAP,
)


def _fetch_all_embeddings() -> tuple[np.ndarray, list[dict]]:
    """Scroll all vectors + payloads from Qdrant. Returns (matrix, payloads)."""
    from qdrant_client import QdrantClient
    from src.embedding.embedder import _get_client

    client = _get_client(QDRANT_PATH)
    vectors, payloads = [], []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_vectors=True,
            with_payload=["book", "chapter_title", "section_title", "text"],
        )
        for r in results:
            vectors.append(r.vector)
            payloads.append(r.payload)
        if offset is None:
            break
    return np.array(vectors, dtype=np.float32), payloads


def _label_cluster(texts: list[str]) -> str:
    """Ask LLM to name a cluster given sample texts."""
    sample = "\n".join(f"- {t[:200]}" for t in texts[:5])
    prompt = (
        "Given these text excerpts from the same topic cluster, "
        "respond with ONLY a short 2-5 word topic label, nothing else:\n" + sample
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip().strip('"').strip("'")
    except Exception:
        return "Unknown"


def build_clusters(force: bool = False) -> dict:
    """Run full clustering pipeline. Returns cluster data dict."""
    if not force and CLUSTERS_PATH.exists():
        print("[SKIP] clusters.json already exists. Use --force to rebuild.")
        return json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))

    print("Fetching embeddings from Qdrant...")
    vectors, payloads = _fetch_all_embeddings()
    if len(vectors) == 0:
        print("No embeddings found. Run 'python main.py embed' first.")
        return {}

    print(f"Reducing {len(vectors)} vectors with UMAP...")
    import umap
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, random_state=42, verbose=False)
    coords = reducer.fit_transform(vectors)

    print("Clustering with HDBSCAN...")
    import hdbscan
    labels = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE).fit_predict(coords)

    # group points by cluster
    cluster_map: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append(i)

    print(f"Found {len([k for k in cluster_map if k != -1])} clusters ({cluster_map.get(-1, []).__len__()} noise points)")

    clusters = []
    for cluster_id in sorted(k for k in cluster_map if k != -1):
        indices = cluster_map[cluster_id]
        texts = [payloads[i].get("text", "") for i in indices]
        print(f"  Labelling cluster {cluster_id} ({len(indices)} points)...")
        label = _label_cluster(texts)
        clusters.append({
            "id": cluster_id,
            "label": label,
            "size": len(indices),
            "points": [
                {
                    "x": round(float(coords[i][0]), 4),
                    "y": round(float(coords[i][1]), 4),
                    "book": payloads[i].get("book", ""),
                    "chapter": payloads[i].get("chapter_title", ""),
                    "section": payloads[i].get("section_title", ""),
                    "text": payloads[i].get("text", "")[:CHUNK_CONTEXT_CAP],
                }
                for i in indices
            ],
        })

    # include noise points unlabelled
    noise_indices = cluster_map.get(-1, [])
    noise_points = [
        {
            "x": round(float(coords[i][0]), 4),
            "y": round(float(coords[i][1]), 4),
            "book": payloads[i].get("book", ""),
            "chapter": payloads[i].get("chapter_title", ""),
            "section": payloads[i].get("section_title", ""),
            "text": payloads[i].get("text", "")[:CHUNK_CONTEXT_CAP],
        }
        for i in noise_indices
    ]

    result = {"clusters": clusters, "noise": noise_points}
    CLUSTERS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Saved to {CLUSTERS_PATH}")
    return result
