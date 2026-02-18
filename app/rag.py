from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")
ALLOWED_EXTS = {".txt", ".md", ".pdf"}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def collect_input_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in ALLOWED_EXTS:
                    files.append(fp)
        elif p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return files


def _extract_pdf_pages(path: Path) -> list[dict]:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyMuPDF is required for PDF support. Install with: pip install pymupdf") from exc

    doc = fitz.open(path)
    pages: list[dict] = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text")
            if text and text.strip():
                pages.append({"page": i + 1, "text": text})
    finally:
        doc.close()
    return pages


def _read_document(path: Path) -> list[dict]:
    if path.suffix.lower() == ".pdf":
        return _extract_pdf_pages(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"page": None, "text": text}]


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _chunk_long_paragraph(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    chunks: list[str] = []
    start = 0
    overlap_words = max(0, overlap // 6)
    while start < len(words):
        current: list[str] = []
        end = start
        while end < len(words):
            candidate = " ".join(current + [words[end]])
            if len(candidate) > max_chars:
                break
            current.append(words[end])
            end += 1
        if not current:
            current = [words[start]]
            end = start + 1
        chunks.append(" ".join(current))
        if end >= len(words):
            break
        effective_overlap = 0
        if overlap_words > 0:
            effective_overlap = min(overlap_words, max(0, len(current) - 1))
        start = end - effective_overlap
    return chunks


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    chunks: list[str] = []
    for para in _split_paragraphs(text):
        chunks.extend(_chunk_long_paragraph(para, max_chars=max_chars, overlap=overlap))
    return chunks


def build_index(files: Iterable[Path], max_chars: int = 800, overlap: int = 100) -> dict:
    chunks: list[dict] = []
    cwd = Path.cwd()
    for path in files:
        pages = _read_document(path)
        for page in pages:
            raw = page.get("text", "")
            if not raw.strip():
                continue
            for chunk_text_item in chunk_text(raw, max_chars=max_chars, overlap=overlap):
                try:
                    source = str(path.relative_to(cwd))
                except ValueError:
                    source = str(path)
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "source": source,
                        "text": chunk_text_item.strip(),
                        "page": page.get("page"),
                    }
                )

    df: Counter[str] = Counter()
    for chunk in chunks:
        terms = set(tokenize(chunk["text"]))
        for t in terms:
            df[t] += 1

    n_docs = max(1, len(chunks))
    idf = {t: math.log((n_docs + 1) / (df[t] + 1)) + 1 for t in df}

    for chunk in chunks:
        tf = Counter(tokenize(chunk["text"]))
        vec = {t: tf[t] * idf.get(t, 0.0) for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        chunk["vec"] = vec
        chunk["norm"] = norm

    return {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "params": {"max_chars": max_chars, "overlap": overlap},
        "idf": idf,
        "chunks": chunks,
    }


def save_index(index: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def search(index: dict, query: str, top_k: int = 3, min_score: float = 0.05) -> list[dict]:
    idf = index.get("idf", {})
    q_tf = Counter(tokenize(query))
    if not q_tf:
        return []
    q_vec = {t: q_tf[t] * idf.get(t, 0.0) for t in q_tf}
    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    if q_norm == 0:
        return []

    scored: list[tuple[float, dict]] = []
    for chunk in index.get("chunks", []):
        c_norm = chunk.get("norm", 0.0)
        if c_norm == 0:
            continue
        dot = 0.0
        c_vec = chunk.get("vec", {})
        for t, qv in q_vec.items():
            dot += qv * c_vec.get(t, 0.0)
        score = dot / (q_norm * c_norm)
        if score >= min_score:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def build_answer(chunks: list[dict]) -> tuple[str, list[dict]]:
    if not chunks:
        return (
            "I couldn't find this in the uploaded documents.",
            [],
        )

    answer_chunks = [c for c in chunks if len(c.get("text", "")) >= 40]
    if not answer_chunks:
        answer_chunks = chunks
    answer = " ".join(c["text"] for c in answer_chunks[:2]).strip()
    citations: list[dict] = []
    for c in chunks[:3]:
        locator = f"chunk_{c['chunk_id']}"
        if c.get("page"):
            locator = f"page_{c['page']}_chunk_{c['chunk_id']}"
        citations.append(
            {
                "source": c["source"],
                "locator": locator,
                "snippet": c["text"][:200].strip(),
            }
        )
    return answer, citations
