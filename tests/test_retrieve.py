"""Retrieve tests: fake 2-D vectors + a fake embed_query, so no Ollama is needed."""
from src import retrieve as retrieve_module
from src import store

CHUNKS = [
    {"id": "cats", "text": "about cats", "source": "a.pdf", "page": 1},
    {"id": "dogs", "text": "about dogs", "source": "a.pdf", "page": 2},
    {"id": "mixed", "text": "cats and dogs", "source": "b.pdf", "page": 5},
]
VECTORS = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]


def setup_store(tmp_path, monkeypatch, query_vector):
    store.add_chunks(CHUNKS, VECTORS, path=str(tmp_path))
    # Replace the real embedder: every question becomes `query_vector`.
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: query_vector)
    # Tests must not depend on .env: these test dense search unless they pass mode= themselves.
    monkeypatch.setattr(retrieve_module, "MODE", "dense")


def test_results_are_ranked_best_first(tmp_path, monkeypatch):
    setup_store(tmp_path, monkeypatch, query_vector=[1.0, 0.0])  # a "cats" question
    results = retrieve_module.retrieve("cats?", k=3, path=str(tmp_path))
    assert [r["id"] for r in results] == ["cats", "mixed", "dogs"]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["score"] == 1.0  # identical direction -> similarity 1


def test_results_carry_metadata_for_citations(tmp_path, monkeypatch):
    setup_store(tmp_path, monkeypatch, query_vector=[0.0, 1.0])
    top = retrieve_module.retrieve("dogs?", k=1, path=str(tmp_path))[0]
    assert (top["id"], top["source"], top["page"], top["text"]) == ("dogs", "a.pdf", 2, "about dogs")


def test_k_larger_than_collection_is_safe(tmp_path, monkeypatch):
    setup_store(tmp_path, monkeypatch, query_vector=[1.0, 0.0])
    assert len(retrieve_module.retrieve("x", k=50, path=str(tmp_path))) == 3


def test_empty_store_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve_module, "embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr(retrieve_module, "MODE", "dense")
    assert retrieve_module.retrieve("x", path=str(tmp_path)) == []


# ---------- BM25, hybrid and rerank ----------

def test_bm25_rewards_exact_rare_words(tmp_path):
    # Realistic case: the keyword appears in only 1 of 4 chunks.
    # (In classic BM25 a word found in MORE than half the chunks gets a negative weight,
    # so a tiny corpus where every chunk shares words gives misleading results.)
    texts = ["the model uses attention", "the model uses the Adam optimizer",
             "the model is trained on GPUs", "the model reaches high BLEU"]
    chunks = [{"id": f"c{i}", "text": t, "source": "a.pdf", "page": i} for i, t in enumerate(texts)]
    store.add_chunks(chunks, [[1.0, float(i)] for i in range(4)], path=str(tmp_path))
    results = retrieve_module.bm25_search("Which optimizer?", n=4, path=str(tmp_path))
    assert results[0]["id"] == "c1"
    assert results[0]["score"] > results[1]["score"]


def test_rrf_prefers_chunks_ranked_well_by_both_lists():
    a, b, c = ({"id": x, "text": x, "source": "s", "page": 1, "score": 0} for x in "abc")
    fused = retrieve_module.rrf_fuse([[a, b, c], [b, c, a]])
    # b: 1/62 + 1/61 beats a: 1/61 + 1/63 -> consistently good beats one #1
    assert [r["id"] for r in fused] == ["b", "a", "c"]


def test_hybrid_mode_returns_k_unique_chunks(tmp_path, monkeypatch):
    setup_store(tmp_path, monkeypatch, query_vector=[1.0, 0.0])
    results = retrieve_module.retrieve("dogs", k=2, path=str(tmp_path), mode="hybrid")
    assert len(results) == 2 and len({r["id"] for r in results}) == 2


def test_unknown_mode_fails_loudly(tmp_path, monkeypatch):
    setup_store(tmp_path, monkeypatch, query_vector=[1.0, 0.0])
    import pytest
    with pytest.raises(ValueError):
        retrieve_module.retrieve("x", path=str(tmp_path), mode="magic")


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    class FakeReranker:  # scores = length of the chunk text, so the order is predictable
        def predict(self, pairs):
            return [len(text) for _, text in pairs]
    monkeypatch.setattr(retrieve_module, "load_reranker", lambda: FakeReranker())
    chunks = [{"id": i, "text": t, "source": "s", "page": 1, "score": 0} for i, t in [("a", "x"), ("b", "xxx"), ("c", "xx")]]
    assert [c["id"] for c in retrieve_module.rerank("q", chunks)] == ["b", "c", "a"]
