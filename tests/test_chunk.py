import pytest

from src.chunk import chunk_records, split_text


def test_overlap_is_shared_between_neighbours():
    text = " ".join(str(n) for n in range(10))  # "0 1 2 ... 9"
    chunks = split_text(text, size=4, overlap=2)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1] == "2 3 4 5"  # starts with the last 2 words of chunk 0


def test_no_words_are_lost():
    words = [str(n) for n in range(103)]
    chunks = split_text(" ".join(words), size=10, overlap=3)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)


def test_short_text_gives_one_chunk():
    assert split_text("just a few words", size=200, overlap=40) == ["just a few words"]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("a b c", size=5, overlap=5)


def test_chunks_keep_page_metadata():
    records = [{"text": "word " * 300, "source": "doc.pdf", "page": 7}]
    chunks = chunk_records(records)
    assert all(c["page"] == 7 and c["source"] == "doc.pdf" for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)  # ids are unique


def test_clean_text_removes_bibliography_markers():
    from src.ingest import clean_text
    raw = "We used the Adam optimizer [20] with\nwarmup [35, 2, 5]. See layer norm [1]."
    assert clean_text(raw) == "We used the Adam optimizer with warmup. See layer norm."
    assert clean_text("position-\nwise network") == "position-wise network"
