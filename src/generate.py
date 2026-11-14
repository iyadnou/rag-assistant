"""Build a prompt from retrieved chunks and ask the LLM for a cited answer."""
import os
import re

import ollama
from dotenv import load_dotenv

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")

# One exact sentence, so code (the pipeline, the UI) can detect a refusal.
NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources provided.

Rules:
- Every factual sentence must end with the number of the source it comes from, like [1] or [2][3].
- Use only information that appears in the sources. Do not use your own knowledge.
- If the sources do not contain the answer, reply with exactly: {NO_ANSWER}
- Be concise: answer in 1-3 sentences."""


def format_sources(chunks: list[dict]) -> str:
    """Number the chunks [1], [2], ... so the model has something to cite."""
    return "\n\n".join(
        f"[{n}] (source: {c['source']}, page {c['page']})\n{c['text']}"
        for n, c in enumerate(chunks, start=1)
    )


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    # Sources go BEFORE the question: the model reads the evidence first,
    # and the question is the last thing it sees before answering.
    user = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate(question: str, chunks: list[dict]) -> str:
    if not chunks:  # nothing retrieved: don't even ask the model, it would only guess
        return NO_ANSWER
    response = ollama.chat(
        model=LLM_MODEL,
        messages=build_messages(question, chunks),
        options={"temperature": 0},  # same question -> same answer: consistent, reproducible answers
    )
    return response.message.content.strip()


def remove_invalid_citations(answer: str, n_sources: int) -> str:
    """Delete [n] markers that don't point to a real source, e.g. "[9]" when there were only 5."""
    def keep_if_valid(match: re.Match) -> str:
        return match.group(0) if 1 <= int(match.group(1)) <= n_sources else ""
    return re.sub(r"\s*\[(\d+)\]", keep_if_valid, answer).strip()


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    """Which sources did the answer cite? Numbers that don't exist (e.g. [9] of 5) are dropped."""
    found = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in found if 1 <= n <= n_sources)
