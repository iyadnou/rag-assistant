"""Find the chunks most relevant to a question.

Modes (set RETRIEVAL_MODE in .env, or pass mode=...):
  dense  - embedding similarity only
  hybrid - dense + BM25 keyword search, merged with Reciprocal Rank Fusion
  rerank - hybrid finds CANDIDATES, then a cross-encoder re-scores and re-orders them (default)

Try it:  .venv/bin/python -m src.retrieve "What optimizer did the authors use?"
"""
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from src.embed import embed_query
from src.store import CHROMA_PATH, get_collection

load_dotenv()
TOP_K = 5  # how many chunks to return, and to give the LLM
MODE = os.getenv("RETRIEVAL_MODE", "rerank")
CANDIDATES = 20  # hybrid: how many results each method contributes before fusing
RRF_K = 60  # standard constant from the RRF paper; dampens the advantage of rank 1 over rank 2
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def dense_search(question: str, n: int, path: str = CHROMA_PATH) -> list[dict]:
    """Semantic search: chunks whose embedding is closest to the question's. score = cosine similarity."""
    collection = get_collection(path)
    n = min(n, collection.count())  # asking Chroma for more results than it has gives warnings
    if n == 0:
        return []
    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma answers a LIST of queries, so every field is a list of lists; we sent one, so take [0].
    return [
        {"id": i, "text": t, "source": m["source"], "page": m["page"], "score": round(1 - d, 3)}
        for i, t, m, d in zip(result["ids"][0], result["documents"][0],
                              result["metadatas"][0], result["distances"][0])
    ]


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers: 'Adam optimizer [20]' -> ['adam', 'optimizer', '20']."""
    return re.findall(r"\w+", text.lower())


def bm25_search(question: str, n: int, path: str = CHROMA_PATH) -> list[dict]:
    """Keyword search: rewards chunks containing the question's exact words, especially rare ones.

    The BM25 index is rebuilt on every call. Fine for 41 chunks (milliseconds);
    with millions of chunks you'd build it once, or use a search engine like Elasticsearch.
    """
    data = get_collection(path).get(include=["documents", "metadatas"])
    if not data["ids"]:
        return []
    bm25 = BM25Okapi([tokenize(doc) for doc in data["documents"]])
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [
        {"id": data["ids"][i], "text": data["documents"][i], "source": data["metadatas"][i]["source"],
         "page": data["metadatas"][i]["page"], "score": round(float(scores[i]), 3)}
        for i in best
    ]


def rrf_fuse(result_lists: list[list[dict]]) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (RRF_K + rank).

    Uses only RANKS, never raw scores - cosine (0..1) and BM25 (0..~20) aren't comparable.
    A chunk ranked well by BOTH methods beats one ranked #1 by only one of them.
    """
    fused = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            entry = fused.setdefault(chunk["id"], {**chunk, "score": 0.0})
            entry["score"] += 1 / (RRF_K + rank)
    merged = sorted(fused.values(), key=lambda c: c["score"], reverse=True)
    for chunk in merged:
        chunk["score"] = round(chunk["score"], 4)
    return merged


@lru_cache(maxsize=1)
def load_reranker():
    """Load the cross-encoder once (~90 MB, a few seconds) and reuse it for every question."""
    from sentence_transformers import CrossEncoder  # imported here: PyTorch is slow to import

    return CrossEncoder(RERANK_MODEL)


def rerank(question: str, chunks: list[dict]) -> list[dict]:
    """Re-order chunks with a cross-encoder.

    An embedding model encodes question and chunk SEPARATELY, then compares two vectors (fast,
    works on millions of chunks). A cross-encoder reads question + chunk TOGETHER and outputs
    one relevance score (much more accurate, but too slow to run on every chunk) - so we only
    run it on the few candidates that hybrid search already found.
    """
    if not chunks:
        return []
    scores = load_reranker().predict([(question, c["text"]) for c in chunks])
    reranked = [{**c, "score": round(float(s), 3)} for c, s in zip(chunks, scores)]
    return sorted(reranked, key=lambda c: c["score"], reverse=True)


def retrieve(question: str, k: int = TOP_K, path: str = CHROMA_PATH, mode: str | None = None) -> list[dict]:
    """Return the k most relevant chunks, best first: [{"id", "text", "source", "page", "score"}]."""
    mode = mode or MODE
    if mode == "dense":
        return dense_search(question, k, path)
    if mode in ("hybrid", "rerank"):
        candidates = rrf_fuse([dense_search(question, CANDIDATES, path),
                               bm25_search(question, CANDIDATES, path)])
        if mode == "rerank":
            candidates = rerank(question, candidates)
        return candidates[:k]
    raise ValueError(f"unknown RETRIEVAL_MODE: {mode!r} (use 'dense', 'hybrid' or 'rerank')")


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How many layers does the encoder have?"
    print(f"Question: {question}   (mode: {MODE})\n")
    for rank, r in enumerate(retrieve(question), start=1):
        print(f"#{rank}  score {r['score']}  {r['source']} p.{r['page']}  ({r['id']})")
        print(f"    {r['text'][:150]}...\n")
