"""Milestone 9: HTTP API around the RAG pipeline.

Run:   .venv/bin/uvicorn api.main:app --reload
Docs:  http://localhost:8000/docs  (interactive: try /ask in the browser)
"""
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.chunk import CHUNK_OVERLAP, CHUNK_SIZE
from src.generate import LLM_MODEL
from src.ingest import RAW_DIR
from src.pipeline import ask
from src.retrieve import MODE, TOP_K
from src.store import build_index, get_collection

MAX_UPLOAD_MB = 50

app = FastAPI(title="Local RAG Assistant", version="1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What optimizer did the authors use?"])
    k: int = Field(default=TOP_K, ge=1, le=20)


class Chunk(BaseModel):
    id: str
    source: str
    page: int
    score: float
    text: str


class Source(Chunk):
    n: int  # the [n] number used in the answer


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]  # only the chunks the answer cites
    retrieved: list[Chunk]  # everything retrieval returned (for debugging / transparency)
    seconds: float


OLLAMA_DOWN = "Cannot reach Ollama. Start it with: brew services start ollama"


@app.get("/health")
def health() -> dict:
    """Is the index loaded, and with which settings?"""
    return {"status": "ok", "chunks": get_collection().count(), "retrieval_mode": MODE,
            "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "llm": LLM_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> dict:
    # A plain `def` (not `async def`): FastAPI runs it in a worker thread, so a slow LLM
    # call doesn't block the server from answering /health in the meantime.
    if get_collection().count() == 0:
        raise HTTPException(status_code=409, detail="No documents yet. Upload a PDF first (POST /upload).")
    start = time.time()
    try:
        result = ask(request.question, k=request.k)
    except ConnectionError:
        raise HTTPException(status_code=503, detail=OLLAMA_DOWN)
    return {**result, "seconds": round(time.time() - start, 1)}


@app.post("/ingest")
def ingest_endpoint() -> dict:
    """Rebuild the index from every PDF in data/raw/ (replaces the old index)."""
    start = time.time()
    try:
        chunks = build_index()
    except ConnectionError:
        raise HTTPException(status_code=503, detail=OLLAMA_DOWN)
    return {"chunks": chunks, "seconds": round(time.time() - start, 1)}


@app.get("/documents")
def list_documents() -> list[dict]:
    """The PDFs in data/raw/ and how many chunks each has in the index."""
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    counts = {}
    for m in metadatas:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return [{"name": p.name, "size_kb": round(p.stat().st_size / 1024), "chunks": counts.get(p.name, 0)}
            for p in sorted(RAW_DIR.glob("*.pdf"))]


@app.post("/upload")
def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    """Save uploaded PDFs to data/raw/, then rebuild the index so they're searchable right away.

    v1 simplicity: re-embeds ALL documents on every upload (seconds for a few papers).
    With many documents you'd embed only the new file and add its chunks.
    """
    saved = []
    for file in files:
        # Path(...).name strips any folders: a file named "../../.env" can't escape data/raw/.
        name = Path(file.filename or "").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {name!r}")
        content = file.file.read()
        if not content.startswith(b"%PDF"):  # check the content, not just the file name
            raise HTTPException(status_code=400, detail=f"{name!r} is not a valid PDF")
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{name!r} is larger than {MAX_UPLOAD_MB} MB")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / name).write_bytes(content)
        saved.append(name)

    result = ingest_endpoint()
    docs = {d["name"]: d["chunks"] for d in list_documents()}
    # A scanned PDF (images only, no text layer) produces zero chunks - say so instead of failing silently.
    no_text = [name for name in saved if docs.get(name, 0) == 0]
    return {"saved": saved, "no_text_found": no_text, **result}


@app.delete("/documents/{name}")
def delete_document(name: str) -> dict:
    """Remove a PDF from data/raw/ and rebuild the index without it."""
    path = RAW_DIR / Path(name).name  # same folder-escape protection as /upload
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail=f"No document named {name!r}")
    path.unlink()
    return {"deleted": path.name, **ingest_endpoint()}
