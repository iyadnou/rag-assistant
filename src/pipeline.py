"""The whole RAG pipeline in one call.

Try it:  .venv/bin/python -m src.pipeline "What is the main contribution of this paper?"
"""
from src.generate import NO_ANSWER, cited_numbers, generate, remove_invalid_citations
from src.retrieve import TOP_K, retrieve


def ask(question: str, k: int = TOP_K) -> dict:
    """question -> {"answer", "sources" (only the ones cited), "retrieved" (all k, for debugging)}."""
    chunks = retrieve(question, k=k)
    answer = remove_invalid_citations(generate(question, chunks), len(chunks))
    cited = cited_numbers(answer, len(chunks))
    return {
        "answer": answer,
        "sources": [{"n": n, **chunks[n - 1]} for n in cited],
        "retrieved": chunks,
        "refused": answer.startswith(NO_ANSWER),
    }


if __name__ == "__main__":
    import sys
    import time

    question = " ".join(sys.argv[1:]) or "What is the main contribution of this paper?"
    start = time.time()
    result = ask(question)
    print(f"Q: {question}\n\nA: {result['answer']}\n")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['source']}, page {s['page']}  (score {s['score']})")
    print(f"\n({time.time() - start:.1f}s)")
