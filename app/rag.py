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
MATH_SYMBOLS = set("=+-/*^_∑∫≤≥≈≠→←∞√·×÷")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def detect_intent(query: Optional[str]) -> str:
    if not query:
        return "default"
    q = query.lower()
    if "summarize" in q or "summary" in q or "main contribution" in q:
        return "summary"
    if "assumption" in q or "limitation" in q:
        return "assumptions"
    if "numeric" in q or "number" in q or "experimental" in q or "detail" in q:
        return "numeric"
    return "default"

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _get_tokenizer():
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _token_count(text: str) -> int:
    encoder = _get_tokenizer()
    if encoder:
        return len(encoder.encode(text))
    # Fallback approximation if tiktoken isn't installed
    return max(1, len(text.split()))


def _clean_pdf_pages(pages: list[dict]) -> list[dict]:
    if len(pages) < 2:
        return pages

    header_lines = 2
    footer_lines = 2
    counts: Counter[str] = Counter()
    page_lines: list[list[str]] = []

    for page in pages:
        lines = [ln.strip() for ln in page.get("text", "").splitlines() if ln.strip()]
        page_lines.append(lines)
        candidates = []
        if lines:
            candidates.extend(lines[:header_lines])
            candidates.extend(lines[-footer_lines:])
        for line in candidates:
            counts[line] += 1

    threshold = max(2, int(len(pages) * 0.6))
    drop_lines = {line for line, c in counts.items() if c >= threshold and len(line) <= 120}

    cleaned: list[dict] = []
    for page, lines in zip(pages, page_lines):
        cleaned_lines = [ln for ln in lines if ln not in drop_lines]
        cleaned.append({"page": page.get("page"), "text": "\n".join(cleaned_lines)})
    return cleaned


def _is_math_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    math_chars = sum(1 for ch in stripped if ch in MATH_SYMBOLS)
    non_alnum = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
    ratio = non_alnum / max(1, len(stripped))
    if math_chars >= 3 or ratio > 0.3:
        return True
    if re.search(r"\\[a-zA-Z]+", stripped):
        return True
    if re.search(r"\b\d+\s*[=<>]\s*\d+", stripped):
        return True
    return False


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_into_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    for para in _split_paragraphs(text):
        blocks.append({"text": para, "is_math": _is_math_block(para)})
    return blocks


def _split_units_by_tokens(units: list[str], max_tokens: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = _token_count(unit)
        if unit_tokens > max_tokens:
            words = unit.split()
            sub: list[str] = []
            sub_tokens = 0
            for word in words:
                wt = _token_count(word)
                if sub and sub_tokens + wt > max_tokens:
                    chunks.append(" ".join(sub))
                    sub = []
                    sub_tokens = 0
                sub.append(word)
                sub_tokens += wt
            if sub:
                chunks.append(" ".join(sub))
            continue

        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _split_large_block(text: str, max_tokens: int, is_math: bool) -> list[str]:
    if _token_count(text) <= max_tokens:
        return [text]
    if is_math:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return _split_units_by_tokens(lines, max_tokens)
    sentences = _split_sentences(text)
    return _split_units_by_tokens(sentences, max_tokens)


def _chunk_blocks(blocks: list[dict], max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current_blocks: list[dict] = []
    current_tokens = 0

    def finalize_chunk() -> None:
        nonlocal current_blocks, current_tokens
        if current_blocks:
            chunks.append("\n\n".join(b["text"] for b in current_blocks).strip())
        if overlap_tokens > 0:
            overlap_blocks: list[dict] = []
            overlap_count = 0
            for b in reversed(current_blocks):
                bt = _token_count(b["text"])
                if overlap_count + bt > overlap_tokens:
                    break
                overlap_blocks.insert(0, b)
                overlap_count += bt
            current_blocks = overlap_blocks
            current_tokens = overlap_count
        else:
            current_blocks = []
            current_tokens = 0

    for block in blocks:
        bt = _token_count(block["text"])
        if current_blocks and current_tokens + bt > max_tokens:
            finalize_chunk()
        current_blocks.append(block)
        current_tokens += bt

    if current_blocks:
        chunks.append("\n\n".join(b["text"] for b in current_blocks).strip())

    return chunks


def chunk_text(text: str, max_tokens: int = 600, overlap_tokens: int = 100) -> list[str]:
    raw_blocks = _split_into_blocks(text)
    blocks: list[dict] = []
    for block in raw_blocks:
        parts = _split_large_block(block["text"], max_tokens, block["is_math"])
        for part in parts:
            blocks.append({"text": part, "is_math": block["is_math"]})
    return _chunk_blocks(blocks, max_tokens, overlap_tokens)

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
        pages = _extract_pdf_pages(path)
        return _clean_pdf_pages(pages)
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
    max_tokens: int = 600,
    overlap_tokens: int = 100,
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
                for chunk_text_item in chunk_text(
                    section_text,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                ):
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
        "version": 3,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "params": {
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "chunking": "section-aware + token window + overlap",
        },
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
    require_overlap: bool = True,
    min_overlap_ratio: float = 0.34,
) -> list[dict]:
    q_terms = [t for t in tokenize(query) if t not in STOPWORDS]
    q_tf = Counter(q_terms)
    if not q_tf:
        return []

    chunks = index.get("chunks", [])
    if not chunks:
        return []

    bm25 = index.get("bm25", {})
    vocab_terms = set(bm25.get("idf", {}).keys())
    q_term_set = set(q_tf.keys())
    known_terms = q_term_set.intersection(vocab_terms)
    if require_overlap and not known_terms:
        return []
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
    required_overlap = 0
    if require_overlap:
        if len(known_terms) >= 3:
            required_overlap = 2
        elif len(known_terms) >= 1:
            required_overlap = 1
    for idx, chunk in enumerate(chunks):
        bm25_score = bm25_norm[idx]
        embed_score = embed_norm[idx]
        if use_embed_channel and total_weight > 0:
            combined = (bm25_weight * bm25_score + embed_weight * embed_score) / total_weight
        else:
            combined = bm25_score

        overlap = 0.0
        overlap_count = 0
        if known_terms:
            overlap_count = len(known_terms.intersection(chunk.get("tf", {}).keys()))
            overlap = overlap_count / len(known_terms)
        if require_overlap and overlap_count < required_overlap:
            continue
        if require_overlap and overlap < min_overlap_ratio:
            continue
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

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _format_citation(chunk: dict) -> dict:
    locator_parts = []
    if chunk.get("page"):
        locator_parts.append(f"page_{chunk['page']}")
    section_slug = _slugify(chunk.get("section"))
    if section_slug:
        locator_parts.append(f"section_{section_slug}")
    locator_parts.append(f"chunk_{chunk['chunk_id']}")
    return {
        "source": chunk["source"],
        "locator": "_".join(locator_parts),
        "snippet": chunk["text"][:200].strip(),
    }


def _collect_sentences(chunks: list[dict]) -> list[tuple[str, dict]]:
    collected: list[tuple[str, dict]] = []
    for chunk in chunks:
        for sentence in _split_sentences(chunk.get("text", "")):
            if len(sentence) >= 30:
                collected.append((sentence, chunk))
    return collected


def _build_bullets(selections: list[tuple[str, dict]], max_items: int = 3) -> tuple[str, list[dict]]:
    bullets: list[str] = []
    used_chunks: dict[int, dict] = {}
    for sentence, chunk in selections:
        if len(bullets) >= max_items:
            break
        cleaned = re.sub(r"^(?:[-•]|o)\s+", "", sentence, flags=re.IGNORECASE)
        bullets.append(f"- {cleaned}")
        used_chunks[chunk["chunk_id"]] = chunk
    if not bullets:
        return "I cannot find this in the uploaded documents.", []
    citations = [_format_citation(c) for c in list(used_chunks.values())[:max_items]]
    return "\n".join(bullets), citations


def _numeric_detail(selections: list[tuple[str, dict]]) -> tuple[str, list[dict]]:
    for sentence, chunk in selections:
        if re.search(r"\\d", sentence):
            citations = [_format_citation(chunk)]
            return sentence, citations
    return "I cannot find this in the uploaded documents.", []


def build_answer(
    chunks: list[dict],
    query: Optional[str] = None,
    max_citations: int = 3,
) -> tuple[str, list[dict]]:
    if not chunks:
        return (
            "I cannot find this in the uploaded documents.",
            [],
        )

    selections = _collect_sentences(chunks)

    if query:
        q = query.lower()
        if "summarize" in q and "bullet" in q:
            return _build_bullets(selections, max_items=3)
        if "assumption" in q or "limitation" in q:
            filtered = [
                (s, c)
                for s, c in selections
                if ("assumption" in s.lower() or "limitation" in s.lower())
            ]
            return _build_bullets(filtered, max_items=3)
        if "numeric" in q or "number" in q or "experimental" in q or "detail" in q:
            return _numeric_detail(selections)

    answer_chunks = [c for c in chunks if len(c.get("text", "")) >= 40]
    if not answer_chunks:
        answer_chunks = chunks
    answer = " ".join(c["text"] for c in answer_chunks[:2]).strip()
    citations = [_format_citation(c) for c in chunks[: max(1, max_citations)]]
    return answer, citations
