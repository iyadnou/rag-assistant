"""Milestone 4: turn text into vectors with Ollama's nomic-embed-text."""
import os

import ollama
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
BATCH_SIZE = 32  # texts per request: fewer round-trips than one-by-one, small enough not to time out

# nomic-embed-text was TRAINED with these prefixes. It embeds documents and
# questions slightly differently so that a question lands close to the passage
# that answers it. Leaving them out quietly lowers retrieval quality.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks for storage. Returns one vector per text, in the same order."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [DOC_PREFIX + t for t in texts[start:start + BATCH_SIZE]]
        vectors.extend(ollama.embed(model=EMBED_MODEL, input=batch).embeddings)
    return vectors


def embed_query(question: str) -> list[float]:
    """Embed a user question for searching."""
    return ollama.embed(model=EMBED_MODEL, input=QUERY_PREFIX + question).embeddings[0]
