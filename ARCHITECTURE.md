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
- Chunking strategy: section-aware parsing using headings, then chunk each section into
  ~800-character pieces with ~100-character overlap.
- Metadata captured per chunk (recommended):
  - source filename
  - page (if PDF)
  - chunk_id (implicit locator)

### 2) Indexing / Storage
- Vector store choice: JSON index with BM25 stats and optional embeddings.
- Persistence: JSON index saved to `artifacts/index.json`.
- Lexical index: BM25 (per-chunk term frequencies + IDF).
- Semantic index: optional Sentence-Transformers embeddings.

### 3) Retrieval + Grounded Answering
- Retrieval method: hybrid BM25 + embedding similarity (top-k).
- Reranking: boosts chunks with higher query-term overlap.
- How citations are built:
  - citation includes: source, locator (page/section/chunk), snippet.
- Failure behavior:
  - if no chunk clears a minimal score, respond with “I cannot find this in the uploaded documents.”

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
- Tool interface shape:
  - CLI command `python3 -m app.cli weather --lat ... --lon ... --start ... --end ...`
- Safety boundaries:
  - date range capped to 31 days
  - timeout on HTTP request
  - restricted to Open-Meteo public endpoint

---

## Tradeoffs & Next Steps
- Why this design?
  - Simple, dependency-free pipeline to make the core RAG + memory behaviors easy to understand.
- What you would improve with more time:
  - Add PDF/HTML parsing, stronger embedding retrieval, and a small web UI.
