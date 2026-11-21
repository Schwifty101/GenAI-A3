# Plan: RAG over 3 research papers (Assignment 3)

Corpus: corpus/*.pdf (Physical AI, LLM Agent Security, Rethinking SE in era of foundation models).
Stack: PyMuPDF, sentence-transformers all-MiniLM-L6-v2 (local), ChromaDB (cosine, persisted ./chroma_db),
local generator Qwen2.5-1.5B-Instruct via transformers, Streamlit UI. No API keys.

Shared contract (ingest.py exports): DB_DIR="chroma_db", COLLECTION="papers", EMBED_MODEL, CHUNK_SIZE=800, CHUNK_OVERLAP=150.
Chunk metadata: {source: filename, page: 1-based int}. Chunk ids "<source>-p<page>-c<n>".

Wave 1 (parallel, independent files):
 A. ingest.py  - load per page, chunk, embed, persist to Chroma; OCR fallback skipped if text found.
 B. ask.py (CLI + retrieve/answer functions) and app.py (Streamlit: input, Ask, Answer, collapsible Sources) + prompts.py
 C. eval_queries.json + eval.py - precision@k (k=3,5)
Wave 2 (sequential): run ingest, ask on 2 questions, eval; screenshots; note.pdf (1-2 pages); prompt_log.txt.
Ponytail: installed as project skill at .claude/skills/ponytail (smallest complete change, no extra abstractions).
