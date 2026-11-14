"""API tests: the pipeline and the store are faked, so no Ollama or real index is needed."""
from fastapi.testclient import TestClient

from api import main

client = TestClient(main.app)

CHUNK = {"id": "a.pdf-p3-c0", "text": "N = 6 layers", "source": "a.pdf", "page": 3, "score": 0.9}


class FakeCollection:
    def __init__(self, n):
        self.n = n

    def count(self):
        return self.n


def test_ask_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(main, "get_collection", lambda: FakeCollection(41))
    monkeypatch.setattr(main, "ask", lambda q, k: {
        "answer": "Six layers [1].", "refused": False,
        "sources": [{"n": 1, **CHUNK}], "retrieved": [CHUNK]})
    r = client.post("/ask", json={"question": "How many layers?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Six layers [1]."
    assert body["sources"][0]["page"] == 3 and body["sources"][0]["n"] == 1


def test_ask_validates_input(monkeypatch):
    monkeypatch.setattr(main, "get_collection", lambda: FakeCollection(41))
    assert client.post("/ask", json={"question": "x"}).status_code == 422  # too short
    assert client.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422


def test_ask_on_empty_index_says_what_to_do(monkeypatch):
    monkeypatch.setattr(main, "get_collection", lambda: FakeCollection(0))
    r = client.post("/ask", json={"question": "How many layers?"})
    assert r.status_code == 409 and "upload" in r.json()["detail"].lower()


def test_ollama_down_gives_503_not_a_crash(monkeypatch):
    monkeypatch.setattr(main, "get_collection", lambda: FakeCollection(41))

    def broken(q, k):
        raise ConnectionError("Failed to connect to Ollama")
    monkeypatch.setattr(main, "ask", broken)
    r = client.post("/ask", json={"question": "How many layers?"})
    assert r.status_code == 503 and "ollama" in r.json()["detail"].lower()


def test_health(monkeypatch):
    monkeypatch.setattr(main, "get_collection", lambda: FakeCollection(41))
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == 41


# ---------- upload ----------

class FakeIndexedCollection:
    """Pretends every PDF in the folder was indexed into 3 chunks, except 'scan.pdf' (no text)."""
    def __init__(self, folder):
        self.folder = folder

    def count(self):
        return 3

    def get(self, include):
        return {"metadatas": [{"source": p.name} for p in self.folder.glob("*.pdf")
                              if p.name != "scan.pdf" for _ in range(3)]}


def setup_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RAW_DIR", tmp_path)
    monkeypatch.setattr(main, "build_index", lambda: 3)
    monkeypatch.setattr(main, "get_collection", lambda: FakeIndexedCollection(tmp_path))


PDF_BYTES = b"%PDF-1.4 fake pdf content"


def test_upload_saves_pdf_and_reindexes(tmp_path, monkeypatch):
    setup_upload(tmp_path, monkeypatch)
    r = client.post("/upload", files=[("files", ("paper.pdf", PDF_BYTES, "application/pdf"))])
    assert r.status_code == 200
    assert r.json()["saved"] == ["paper.pdf"] and r.json()["no_text_found"] == []
    assert (tmp_path / "paper.pdf").read_bytes() == PDF_BYTES
    assert client.get("/documents").json()[0] == {"name": "paper.pdf", "size_kb": 0, "chunks": 3}


def test_upload_rejects_non_pdf(tmp_path, monkeypatch):
    setup_upload(tmp_path, monkeypatch)
    r = client.post("/upload", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400
    r = client.post("/upload", files=[("files", ("fake.pdf", b"not really a pdf", "application/pdf"))])
    assert r.status_code == 400
    assert list(tmp_path.iterdir()) == []  # nothing was saved


def test_upload_cannot_escape_the_folder(tmp_path, monkeypatch):
    setup_upload(tmp_path / "raw", monkeypatch)
    r = client.post("/upload", files=[("files", ("../../evil.pdf", PDF_BYTES, "application/pdf"))])
    assert r.status_code == 200
    assert (tmp_path / "raw" / "evil.pdf").exists() and not (tmp_path / "evil.pdf").exists()


def test_upload_warns_about_scanned_pdfs(tmp_path, monkeypatch):
    setup_upload(tmp_path, monkeypatch)
    r = client.post("/upload", files=[("files", ("scan.pdf", PDF_BYTES, "application/pdf"))])
    assert r.json()["no_text_found"] == ["scan.pdf"]


def test_delete_document(tmp_path, monkeypatch):
    setup_upload(tmp_path, monkeypatch)
    (tmp_path / "paper.pdf").write_bytes(PDF_BYTES)
    r = client.delete("/documents/paper.pdf")
    assert r.status_code == 200 and r.json()["deleted"] == "paper.pdf"
    assert not (tmp_path / "paper.pdf").exists()
    assert client.delete("/documents/paper.pdf").status_code == 404


def test_build_index_with_no_pdfs_gives_empty_index(tmp_path, monkeypatch):
    from src import ingest, store
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)  # empty folder
    monkeypatch.setattr(ingest.load_all, "__defaults__", (tmp_path,))
    assert store.build_index(path=str(tmp_path / "db")) == 0
