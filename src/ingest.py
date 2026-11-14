"""Milestone 2: load PDFs into clean text + metadata, one record per page.

Run with:  .venv/bin/python -m src.ingest
"""
import re
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")

# Bibliography markers like [20], [35, 2, 5] or [3-5]. They clash with OUR [1], [2] source
# citations: the model copies them into answers ("Adam optimizer [20]"), so we remove them.
REFERENCE_MARKER = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")


def clean_text(text: str) -> str:
    """Turn raw PDF text into one clean string."""
    # Line-end hyphens: keep the hyphen, drop the line break.
    # "position-\nwise" -> "position-wise" (correct)
    # "trans-\nformer"  -> "trans-former"  (slightly wrong, but still readable)
    # Deleting the hyphen instead would break real hyphenated words.
    text = text.replace("-\n", "-")
    # Line breaks are just printed-line wraps, not real paragraphs -> spaces.
    text = text.replace("\n", " ")
    text = REFERENCE_MARKER.sub("", text)
    # split() with no argument splits on ANY whitespace (spaces, tabs, repeats),
    # so joining with one space collapses everything to single spaces.
    text = " ".join(text.split())
    # Removing "[11]" from "connection [11]." leaves "connection ." -> reattach the punctuation.
    return re.sub(r" ([.,;:])", r"\1", text)


def load_pdf(path: Path) -> list[dict]:
    """Return one record per page: {"text", "source", "page"}."""
    records = []
    with pymupdf.open(path) as doc:  # "with" closes the file when we're done
        for page_number, page in enumerate(doc, start=1):  # start=1: humans count pages from 1
            text = clean_text(page.get_text())
            if not text:  # blank pages (e.g. scanned images) would be useless chunks
                continue
            records.append({
                "text": text,
                "source": path.name,  # just "attention.pdf", not your private folder path
                "page": page_number,
            })
    return records


def load_all(folder: Path = RAW_DIR) -> list[dict]:
    """Load every PDF in the folder."""
    records = []
    for path in sorted(folder.glob("*.pdf")):  # sorted: same order every run
        records.extend(load_pdf(path))  # extend adds the pages; append would add one list inside the list
    return records


if __name__ == "__main__":
    records = load_all()
    print(f"Loaded {len(records)} pages")
    if records:
        print(records[2])  # look at one page: is the text clean? is the metadata right?
