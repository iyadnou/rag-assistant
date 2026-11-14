"""Streamlit UI: upload your PDFs, then ask questions about them. Talks to the FastAPI backend over HTTP.

Run (two terminals):
    .venv/bin/uvicorn api.main:app
    .venv/bin/streamlit run app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", page_icon="📄", layout="centered")
st.title("📄 Local RAG Assistant")
st.caption("1. Upload your PDFs in the sidebar · 2. Ask a question · Answers cite their pages. Runs 100% locally.")


def show_error(response: requests.Response) -> None:
    try:
        st.error(response.json().get("detail", response.text))
    except ValueError:
        st.error(response.text)


# ---------- sidebar: backend status + documents ----------
with st.sidebar:
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        st.error(f"API not reachable at {API_URL}.\n\nStart it with:\n`.venv/bin/uvicorn api.main:app`")
        st.stop()

    st.header("Your documents")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Upload & index", type="primary"):
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploads]
        with st.spinner(f"Reading and embedding {len(files)} file(s)..."):
            r = requests.post(f"{API_URL}/upload", files=files, timeout=900)
        if r.ok:
            body = r.json()
            st.success(f"Added {', '.join(body['saved'])} ({body['seconds']}s)")
            for name in body["no_text_found"]:
                st.warning(f"No text found in {name}. Is it a scanned PDF? Scans need OCR first.")
        else:
            show_error(r)

    docs = requests.get(f"{API_URL}/documents", timeout=10).json()
    for d in docs:
        name_col, button_col = st.columns([5, 1])
        name_col.caption(f"📄 **{d['name']}**  \n{d['chunks']} chunks · {d['size_kb']} KB")
        if button_col.button("✕", key=f"del-{d['name']}", help=f"Remove {d['name']}"):
            with st.spinner(f"Removing {d['name']}..."):
                r = requests.delete(f"{API_URL}/documents/{d['name']}", timeout=900)
            if r.ok:
                st.rerun()
            else:
                show_error(r)

    st.header("Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)
    show_retrieved = st.toggle("Show all retrieved chunks", value=False)
    st.caption(f"Retrieval: **{health['retrieval_mode']}** · {health['chunk_size']}-word chunks · "
               f"LLM: {health['llm']}")

# ---------- main: question -> answer ----------
has_docs = any(d["chunks"] > 0 for d in docs)
if not has_docs:
    st.info("👈 Upload a PDF in the sidebar to get started.")

with st.form("ask"):
    question = st.text_input("Your question", placeholder="e.g. What is the main contribution of this paper?",
                             disabled=not has_docs)
    submitted = st.form_submit_button("Ask", type="primary", disabled=not has_docs)

if submitted and question.strip():
    with st.spinner("Searching your documents and writing an answer..."):
        r = requests.post(f"{API_URL}/ask", json={"question": question, "k": k}, timeout=300)
    if not r.ok:
        show_error(r)
        st.stop()
    result = r.json()

    if result["refused"]:
        st.warning(result["answer"])
        st.caption("The retrieved passages don't contain the answer, so the assistant declined "
                   "instead of guessing.")
    else:
        st.markdown(f"### Answer\n{result['answer']}")
    st.caption(f"⏱ {result['seconds']}s")

    if result["sources"]:
        st.markdown("### Sources")
        for s in result["sources"]:
            with st.expander(f"[{s['n']}] {s['source']}, page {s['page']}"):
                st.write(s["text"])

    if show_retrieved:
        st.markdown("### All retrieved chunks")
        for i, c in enumerate(result["retrieved"], start=1):
            with st.expander(f"#{i} {c['source']}, page {c['page']}  ·  score {c['score']}"):
                st.write(c["text"])
