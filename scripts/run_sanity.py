from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from app import memory, rag


def main() -> None:
    artifacts = repo_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    sample_dir = repo_root / "sample_docs"
    demo_doc = sample_dir / "demo_doc.txt"
    if demo_doc.exists():
        files = [demo_doc]
    else:
        files = rag.collect_input_files([sample_dir])
    if not files:
        raise SystemExit("No sample documents found in sample_docs/")

    index = rag.build_index(files)
    index_path = artifacts / "index.json"
    rag.save_index(index, index_path)

    question = "What retrieval scoring strategy does the demo document describe?"
    hits = rag.search(index, question, top_k=3, min_score=0.05)
    answer, citations = rag.build_answer(hits)
    if not citations:
        raise SystemExit("Sanity failed: no citations produced. Check sample docs or retrieval.")

    memory_writes = [
        {"target": "USER", "summary": "User prefers weekly summaries on Mondays."},
        {"target": "COMPANY", "summary": "Recurring workflow bottleneck is manual document triage."},
    ]
    memory.write_memory_entries(memory_writes)

    output = {
        "implemented_features": ["A", "B"],
        "qa": [
            {
                "question": question,
                "answer": answer,
                "citations": citations,
            }
        ],
        "demo": {
            "ingested_files": [str(p) for p in files],
            "memory_writes": memory_writes,
        },
    }

    out_path = artifacts / "sanity_output.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
