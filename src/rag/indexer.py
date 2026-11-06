"""Indexador del RAG: trocea el corpus y lo guarda en ChromaDB.

Usa embeddings locales (sentence-transformers) para no depender de una API
externa. Ejecuta este módulo una vez para construir el índice:

    python -m src.rag.indexer
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from src import config

COLLECTION_NAME = "corpus_aire"


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Troceo simple por caracteres con solapamiento."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )


def get_collection():
    """Devuelve (o crea) la colección Chroma persistente."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=get_embedding_function()
    )


def build_index() -> int:
    """(Re)construye el índice a partir de los .md del directorio corpus/."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Reinicio limpio para reproducibilidad
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=get_embedding_function()
    )

    docs, ids, metadatas = [], [], []
    for md_file in sorted(config.CORPUS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{md_file.stem}-{i}")
            metadatas.append({"fuente": md_file.name, "chunk": i})

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
    return len(docs)


if __name__ == "__main__":
    n = build_index()
    print(f"Índice construido con {n} fragmentos en {config.CHROMA_DIR}")
