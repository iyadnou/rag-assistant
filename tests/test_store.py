"""Store tests use tiny fake vectors in a temporary folder: no Ollama needed, real data untouched."""
import pytest

from src import store

CHUNKS = [
    {"id": "a-p1-c0", "text": "cats", "source": "a.pdf", "page": 1},
    {"id": "a-p2-c0", "text": "dogs", "source": "a.pdf", "page": 2},
]
VECTORS = [[1.0, 0.0], [0.0, 1.0]]


def test_add_and_read_back(tmp_path):
    store.add_chunks(CHUNKS, VECTORS, path=str(tmp_path))
    col = store.get_collection(str(tmp_path))
    assert col.count() == 2
    got = col.get(ids=["a-p2-c0"])
    assert got["documents"] == ["dogs"]
    assert got["metadatas"][0] == {"source": "a.pdf", "page": 2}


def test_reset_empties_the_store(tmp_path):
    store.add_chunks(CHUNKS, VECTORS, path=str(tmp_path))
    store.reset(str(tmp_path))
    assert store.get_collection(str(tmp_path)).count() == 0


def test_mismatched_lengths_fail_loudly(tmp_path):
    with pytest.raises(ValueError):
        store.add_chunks(CHUNKS, VECTORS[:1], path=str(tmp_path))
