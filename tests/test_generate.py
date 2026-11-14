"""Generation tests: check prompt building, citation parsing and wiring, with the LLM mocked out."""
from src import pipeline
from src.generate import NO_ANSWER, build_messages, cited_numbers, generate

CHUNKS = [
    {"id": "x-p1-c0", "text": "Alpha text.", "source": "x.pdf", "page": 1, "score": 0.8},
    {"id": "x-p4-c0", "text": "Beta text.", "source": "x.pdf", "page": 4, "score": 0.7},
]


def test_sources_are_numbered_with_page_and_come_before_question():
    user = build_messages("Why?", CHUNKS)[1]["content"]
    assert "[1] (source: x.pdf, page 1)\nAlpha text." in user
    assert "[2] (source: x.pdf, page 4)\nBeta text." in user
    assert user.index("[2]") < user.index("Question: Why?")


def test_cited_numbers_ignores_duplicates_and_invented_sources():
    assert cited_numbers("A [2]. B [1][2]. C [9].", n_sources=2) == [1, 2]
    assert cited_numbers(NO_ANSWER, n_sources=5) == []


def test_no_chunks_means_no_llm_call():
    assert generate("anything", []) == NO_ANSWER


def test_ask_returns_only_cited_sources(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q, k: CHUNKS)
    monkeypatch.setattr(pipeline, "generate", lambda q, c: "It is beta [2].")
    result = pipeline.ask("q")
    assert [s["page"] for s in result["sources"]] == [4]
    assert result["sources"][0]["n"] == 2
    assert len(result["retrieved"]) == 2
    assert result["refused"] is False


def test_ask_flags_refusals(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q, k: CHUNKS)
    monkeypatch.setattr(pipeline, "generate", lambda q, c: NO_ANSWER)
    result = pipeline.ask("q")
    assert result["refused"] is True and result["sources"] == []


def test_invalid_citations_are_removed_from_the_answer():
    from src.generate import remove_invalid_citations
    assert remove_invalid_citations("The authors used Adam [20]. [1]", n_sources=5) == "The authors used Adam. [1]"
    assert remove_invalid_citations("A [1][2]. B [3].", n_sources=3) == "A [1][2]. B [3]."
