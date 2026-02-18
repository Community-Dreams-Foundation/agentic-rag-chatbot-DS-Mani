from __future__ import annotations

import re
from pathlib import Path

USER_PATTERNS = [
    (re.compile(r"\bI am ([^.]+?)(?:[.!?]|$)", re.IGNORECASE), "User is {}."),
    (re.compile(r"\bI'm ([^.]+?)(?:[.!?]|$)", re.IGNORECASE), "User is {}."),
    (re.compile(r"\bI work as ([^.]+?)(?:[.!?]|$)", re.IGNORECASE), "User works as {}."),
    (re.compile(r"\bI prefer ([^.]+?)(?:[.!?]|$)", re.IGNORECASE), "User prefers {}."),
    (re.compile(r"\bI like ([^.]+?)(?:[.!?]|$)", re.IGNORECASE), "User likes {}."),
]

COMPANY_PATTERNS = [
    (
        re.compile(r"\bOur (?:team|company) ([^.]+?)(?:[.!?]|$)", re.IGNORECASE),
        "Our team/company {}.",
    ),
    (
        re.compile(r"\bOur (?:workflow|process) ([^.]+?)(?:[.!?]|$)", re.IGNORECASE),
        "Our workflow {}.",
    ),
]


def extract_memories(text: str) -> list[dict]:
    entries: list[dict] = []
    for pattern, template in USER_PATTERNS:
        for match in pattern.findall(text):
            summary = template.format(match.strip())
            if 3 <= len(summary) <= 140:
                entries.append({"target": "USER", "summary": summary, "confidence": 0.6})

    for pattern, template in COMPANY_PATTERNS:
        for match in pattern.findall(text):
            summary = template.format(match.strip())
            if 3 <= len(summary) <= 140:
                entries.append({"target": "COMPANY", "summary": summary, "confidence": 0.6})

    return entries


def _ensure_file(path: Path, header: str) -> None:
    if not path.exists():
        path.write_text(header + "\n\n", encoding="utf-8")


def _load_existing_summaries(path: Path) -> set[str]:
    if not path.exists():
        return set()
    summaries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- "):
            summaries.add(line[2:].strip())
    return summaries


def write_memory_entries(
    entries: list[dict],
    user_path: Path = Path("USER_MEMORY.md"),
    company_path: Path = Path("COMPANY_MEMORY.md"),
) -> list[dict]:
    _ensure_file(user_path, "# USER MEMORY")
    _ensure_file(company_path, "# COMPANY MEMORY")

    user_existing = _load_existing_summaries(user_path)
    comp_existing = _load_existing_summaries(company_path)

    written: list[dict] = []
    for entry in entries:
        target = entry.get("target")
        summary = entry.get("summary", "").strip()
        if not summary:
            continue
        if target == "USER":
            if summary in user_existing:
                continue
            user_path.write_text(
                user_path.read_text(encoding="utf-8").rstrip() + f"\n- {summary}\n",
                encoding="utf-8",
            )
            user_existing.add(summary)
            written.append(entry)
        elif target == "COMPANY":
            if summary in comp_existing:
                continue
            company_path.write_text(
                company_path.read_text(encoding="utf-8").rstrip() + f"\n- {summary}\n",
                encoding="utf-8",
            )
            comp_existing.add(summary)
            written.append(entry)
    return written

