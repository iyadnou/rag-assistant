"""Milestone 4: save chunks + their vectors in Chroma, a local vector database.

Build (or rebuild) the index with:  .venv/bin/python -m src.store
"""
import os

import chromadb
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")
COLLECTION = "chunks"


def get_collection(path: str = CHROMA_PATH):
    client = chromadb.PersistentClient(path=path)  # saved on disk, survives restarts
    return client.get_or_create_collection(
        name=COLLECTION,
        # Cosine: compares the DIRECTION of vectors, not their length -> "same meaning".
        configuration={"hnsw": {"space": "cosine"}},
        # We compute embeddings ourselves with Ollama, so Chroma must not use
        # (or download) its own default embedding model.
        embedding_function=None,
    )


def reset(path: str = CHROMA_PATH) -> None:
    """Delete everything. Needed when chunk settings change, or old chunks would linger."""
    client = chromadb.PersistentClient(path=path)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)


def add_chunks(chunks: list[dict], vectors: list[list[float]], path: str = CHROMA_PATH) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    get_collection(path).add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )


def build_index(path: str = CHROMA_PATH) -> int:
    """Full pipeline: PDFs -> pages -> chunks -> vectors -> Chroma. Returns chunk count.

    `path` lets experiments build a separate index without touching the main one.
    """
    from src.chunk import chunk_records
    from src.embed import embed_documents
    from src.ingest import load_all

    chunks = chunk_records(load_all())
    reset(path)
    if not chunks:  # no PDFs (or only scanned ones): leave an empty index instead of crashing
        get_collection(path)
        return 0
    vectors = embed_documents([c["text"] for c in chunks])
    add_chunks(chunks, vectors, path)
    return get_collection(path).count()


if __name__ == "__main__":
    import time

    start = time.time()
    count = build_index()
    print(f"Indexed {count} chunks in {time.time() - start:.1f}s")

    if count == 0:
        print("No PDFs found in data/raw/ - add some (or upload them in the app) and run this again.")
    else:
        sample = get_collection().get(limit=1, include=["documents", "metadatas", "embeddings"])
        print("Sample id:  ", sample["ids"][0])
        print("Metadata:   ", sample["metadatas"][0])
        print("Vector size:", len(sample["embeddings"][0]))
        print("Text:       ", sample["documents"][0][:120], "...")
