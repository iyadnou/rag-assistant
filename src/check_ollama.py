"""Setup check: can Python talk to Ollama (chat model + embedding model)?

Run with:  .venv/bin/python src/check_ollama.py
"""
import os
import time

import ollama
from dotenv import load_dotenv

# Read settings from the .env file, so model names are not hard-coded.
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# --- Part 1: chat model (this is the "G" in RAG: Generation) ---
start = time.time()
response = ollama.chat(
    model=LLM_MODEL,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    think=False,  # qwen3 "thinks" before answering by default: slow (96s!) and not needed here
)
print("Chat reply:", response.message.content)
print(f"Chat took {time.time() - start:.1f}s")

# --- Part 2: embedding model (used for the "R" in RAG: Retrieval) ---
result = ollama.embed(
    model=EMBED_MODEL,  # must be the EMBEDDING model, not the chat model
    input="RAG means retrieval-augmented generation.",
)
vector = result.embeddings[0]  # embed() accepts a list of texts, so it returns a list of vectors
print("Embedding length:", len(vector))
print("First 5 numbers:", vector[:5])
