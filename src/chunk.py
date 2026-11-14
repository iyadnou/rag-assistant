"""Milestone 3: split page records into overlapping fixed-size chunks.

Run with:  .venv/bin/python -m src.chunk
"""
import os

from dotenv import load_dotenv

load_dotenv()
# Read from the environment so Milestone 8 experiments can change them without editing code:
#   CHUNK_SIZE=100 CHUNK_OVERLAP=20 .venv/bin/python -m src.store
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))  # words per chunk (~1,300 characters, ~260 tokens)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 40))  # words shared between neighbouring chunks


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of `size` words; each chunk repeats the last `overlap` words of the previous one."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, or the window never moves forward")

    words = text.split()
    step = size - overlap  # how far the window moves each time
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):  # this chunk reached the end; stop before making a tiny duplicate tail
            break
    return chunks


def chunk_records(records: list[dict]) -> list[dict]:
    """Chunk each page separately, so every chunk keeps its page's metadata."""
    chunks = []
    for record in records:
        for i, text in enumerate(split_text(record["text"])):
            chunks.append({
                "id": f"{record['source']}-p{record['page']}-c{i}",  # unique + readable, e.g. attention.pdf-p3-c1
                "text": text,
                "source": record["source"],
                "page": record["page"],
            })
    return chunks


if __name__ == "__main__":
    from src.ingest import load_all

    chunks = chunk_records(load_all())
    lengths = [len(c["text"].split()) for c in chunks]
    print(f"{len(chunks)} chunks | words per chunk: min {min(lengths)}, max {max(lengths)}, "
          f"avg {sum(lengths) / len(lengths):.0f}")
    print(chunks[5])
