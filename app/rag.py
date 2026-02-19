from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

TOKEN_RE = re.compile(r"[a-z0-9]+")
ALLOWED_EXTS = {".txt", ".md", ".pdf"}
EMBED_MODEL_DEFAULT = "all-MiniLM-L6-v2"


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


def _is_heading(line: str) -> bool:
    if not line:
        return False
    if line.startswith("#"):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+\S+", line):
        return True
    if 4 <= len(line) <= 80 and line.isupper():
        return True
    if 4 <= len(line) <= 80 and line.endswith(":"):
        return True
    return False


def _normalize_heading(line: str) -> str:
    text = line.strip()
    if text.startswith("#"):
        text = text.lstrip("#").strip()
    return text or "Section"


def split_sections(text: str, default_section: str = "Document") -> list[dict]:
    lines = text.splitlines()
    sections: list[dict] = []
    current_title = default_section
    buffer: list[str] = []

    for raw in lines:
        line = raw.strip()
        if _is_heading(line):
            if buffer:
                sections.append({"section": current_title, "text": "\n".join(buffer).strip()})
                buffer = []
            current_title = _normalize_heading(line)
        else:
            buffer.append(raw)

    if buffer:
        sections.append({"section": current_title, "text": "\n".join(buffer).strip()})

    if not sections and text.strip():
        sections.append({"section": default_section, "text": text.strip()})

    return sections


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


def _load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "sentence-transformers is required for embeddings. Install with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def _embed_texts(model, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def build_index(
    files: Iterable[Path],
    max_chars: int = 800,
    overlap: int = 100,
    use_embeddings: bool = False,
    embed_model: str = EMBED_MODEL_DEFAULT,
) -> dict:
    chunks: list[dict] = []
    cwd = Path.cwd()

    for path in files:
        pages = _read_document(path)
        for page in pages:
            raw = page.get("text", "")
            if not raw.strip():
                continue
            default_section = f"Page {page['page']}" if page.get("page") else "Document"
            sections = split_sections(raw, default_section=default_section)
            for section in sections:
                section_text = section.get("text", "").strip()
                if not section_text:
                    continue
                for chunk_text_item in chunk_text(section_text, max_chars=max_chars, overlap=overlap):
                    text = chunk_text_item.strip()
                    tokens = tokenize(text)
                    if not tokens:
                        continue
                    try:
                        source = str(path.relative_to(cwd))
                    except ValueError:
                        source = str(path)
                    chunks.append(
                        {
                            "chunk_id": len(chunks),
                            "source": source,
                            "text": text,
                            "page": page.get("page"),
                            "section": section.get("section"),
                            "tf": dict(Counter(tokens)),
                            "length": len(tokens),
                        }
                    )

    df: Counter[str] = Counter()
    for chunk in chunks:
        for t in chunk["tf"].keys():
            df[t] += 1

    n_docs = max(1, len(chunks))
    idf = {t: math.log((n_docs - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}
    avgdl = sum(c["length"] for c in chunks) / n_docs if n_docs else 1.0

    embed_model_used: Optional[str] = None
    if use_embeddings and chunks:
        model = _load_embedder(embed_model)
        vectors = _embed_texts(model, [c["text"] for c in chunks])
        for chunk, vec in zip(chunks, vectors):
            chunk["embedding"] = vec
        embed_model_used = embed_model

    return {
        "version": 2,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "params": {"max_chars": max_chars, "overlap": overlap},
        "bm25": {"k1": 1.5, "b": 0.75, "avgdl": avgdl, "idf": idf},
        "has_embeddings": embed_model_used is not None,
        "embed_model": embed_model_used,
        "chunks": chunks,
    }


def save_index(index: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bm25_score(chunk: dict, q_tf: Counter[str], bm25: dict) -> float:
    score = 0.0
    tf = chunk.get("tf", {})
    dl = chunk.get("length", 0)
    avgdl = bm25.get("avgdl", 1.0) or 1.0
    k1 = bm25.get("k1", 1.5)
    b = bm25.get("b", 0.75)
    idf = bm25.get("idf", {})

    for term, qf in q_tf.items():
        if term not in tf:
            continue
        term_freq = tf[term]
        denom = term_freq + k1 * (1 - b + b * (dl / avgdl))
        score += idf.get(term, 0.0) * (term_freq * (k1 + 1) / (denom or 1.0)) * qf
    return score


def _min_max_norm(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if math.isclose(min_s, max_s):
        return [0.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def search(
    index: dict,
    query: str,
    top_k: int = 3,
    min_score: float = 0.1,
    use_embeddings: bool = False,
    embed_model: str = EMBED_MODEL_DEFAULT,
    bm25_weight: float = 0.6,
    embed_weight: float = 0.4,
    rerank_weight: float = 0.2,
) -> list[dict]:
    q_tf = Counter(tokenize(query))
    if not q_tf:
        return []

    chunks = index.get("chunks", [])
    if not chunks:
        return []

    bm25 = index.get("bm25", {})
    bm25_scores = [_bm25_score(chunk, q_tf, bm25) for chunk in chunks]
    bm25_norm = _min_max_norm(bm25_scores)

    embed_scores = [0.0 for _ in chunks]
    if use_embeddings and index.get("has_embeddings"):
        model = _load_embedder(index.get("embed_model") or embed_model)
        query_vec = _embed_texts(model, [query])[0]
        embed_scores = [_cosine(query_vec, c.get("embedding", [])) for c in chunks]
    embed_norm = _min_max_norm(embed_scores)

    use_embed_channel = use_embeddings and index.get("has_embeddings")
    total_weight = (bm25_weight if bm25_weight > 0 else 0) + (embed_weight if embed_weight > 0 else 0)
    if not use_embed_channel:
        total_weight = bm25_weight if bm25_weight > 0 else 1.0

    results: list[tuple[float, dict]] = []
    q_terms = set(q_tf.keys())
    for idx, chunk in enumerate(chunks):
        bm25_score = bm25_norm[idx]
        embed_score = embed_norm[idx]
        if use_embed_channel and total_weight > 0:
            combined = (bm25_weight * bm25_score + embed_weight * embed_score) / total_weight
        else:
            combined = bm25_score

        overlap = 0.0
        if q_terms:
            overlap = len(q_terms.intersection(chunk.get("tf", {}).keys())) / len(q_terms)
        combined += rerank_weight * overlap

        if combined >= min_score:
            results.append((combined, chunk))

    results.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in results[:top_k]]


def _slugify(text: Optional[str]) -> str:
    if not text:
        return ""
    value = text.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:40]


def build_answer(chunks: list[dict], max_citations: int = 3) -> tuple[str, list[dict]]:
    if not chunks:
        return (
            "I cannot find this in the uploaded documents.",
            [],
        )

    answer_chunks = [c for c in chunks if len(c.get("text", "")) >= 40]
    if not answer_chunks:
        answer_chunks = chunks
    answer = " ".join(c["text"] for c in answer_chunks[:2]).strip()
    citations: list[dict] = []
    for c in chunks[: max(1, max_citations)]:
        locator_parts = []
        if c.get("page"):
            locator_parts.append(f"page_{c['page']}")
        section_slug = _slugify(c.get("section"))
        if section_slug:
            locator_parts.append(f"section_{section_slug}")
        locator_parts.append(f"chunk_{c['chunk_id']}")
        citations.append(
            {
                "source": c["source"],
                "locator": "_".join(locator_parts),
                "snippet": c["text"][:200].strip(),
            }
        )
    return answer, citations
