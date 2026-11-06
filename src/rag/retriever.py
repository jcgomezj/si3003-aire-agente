"""Recuperador del RAG: busca fragmentos relevantes en ChromaDB."""
from __future__ import annotations

from src import config
from src.rag.indexer import build_index, get_collection


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Devuelve los fragmentos más relevantes para la consulta.

    Cada resultado incluye el texto y la fuente (para que el agente CITE).
    """
    top_k = top_k or config.RAG_TOP_K
    collection = get_collection()

    # Si el índice está vacío, lo construimos al vuelo (primer arranque).
    if collection.count() == 0:
        build_index()
        collection = get_collection()

    res = collection.query(query_texts=[query], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results = []
    for doc, meta, dist in zip(docs, metas, dists):
        results.append(
            {
                "texto": doc,
                "fuente": meta.get("fuente", "desconocida"),
                "score": round(1 - dist, 3),  # mayor = más relevante
            }
        )
    return results


if __name__ == "__main__":
    for r in search("¿qué hago si el aire está dañino para la salud?"):
        print(f"[{r['fuente']} | {r['score']}] {r['texto'][:120]}...")
