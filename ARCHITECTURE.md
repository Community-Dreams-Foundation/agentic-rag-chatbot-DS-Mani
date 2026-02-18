# Architecture Overview

## Goal
Provide a brief, readable overview of how your chatbot works:
- ingestion
- indexing
- retrieval + grounding with citations
- memory writing
- optional safe tool execution

Keep this short (1–2 pages).

---

## High-Level Flow

### 1) Ingestion (Upload → Parse → Chunk)
- Supported inputs: `.txt`, `.md`, `.pdf` via CLI directory/file paths.
- Parsing approach: read UTF-8 text directly; PDFs are extracted per page using PyMuPDF.
- Chunking strategy: split by paragraph, then further split long paragraphs into
  ~800-character chunks with ~100-character overlap.
- Metadata captured per chunk (recommended):
  - source filename
  - page (if PDF)
  - chunk_id (implicit locator)

### 2) Indexing / Storage
- Vector store choice: in-memory TF-IDF vectors computed in Python stdlib.
- Persistence: JSON index saved to `artifacts/index.json`.
- Optional lexical index (BM25): not implemented (TF-IDF only).

### 3) Retrieval + Grounded Answering
- Retrieval method: cosine similarity over TF-IDF vectors (top-k).
- How citations are built:
  - citation includes: source, locator (`chunk_<id>`), snippet.
- Failure behavior:
  - if no chunk clears a minimal score, respond with “I couldn’t find this in the uploaded documents.”

### 4) Memory System (Selective)
- What counts as “high-signal” memory:
  - user preferences, role/occupation, stable facts.
- What you explicitly do NOT store (PII/secrets/raw transcript):
  - raw conversation, secrets, or sensitive identifiers.
- How you decide when to write:
  - simple regex heuristics (e.g., “I prefer …”, “I am …”) or explicit summaries via CLI.
- Format written to:
  - `USER_MEMORY.md`
  - `COMPANY_MEMORY.md`

### 5) Optional: Safe Tooling (Open-Meteo)
- Not implemented in this version.

---

## Tradeoffs & Next Steps
- Why this design?
  - Simple, dependency-free pipeline to make the core RAG + memory behaviors easy to understand.
- What you would improve with more time:
  - Add PDF/HTML parsing, stronger embedding retrieval, and a small web UI.
