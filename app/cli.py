from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import memory, rag


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAG + memory CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Ingest files into an index")
    ingest.add_argument("--input", nargs="+", required=True, help="File(s) or dir(s)")
    ingest.add_argument("--index", required=True, help="Output index path (json)")

    ask = sub.add_parser("ask", help="Ask a question against an index")
    ask.add_argument("--index", required=True, help="Path to index json")
    ask.add_argument("--question", required=True, help="Question to ask")
    ask.add_argument("--top-k", type=int, default=3, help="Top-K chunks")
    ask.add_argument("--json", action="store_true", help="Output JSON")

    mem = sub.add_parser("memory", help="Write selective memory")
    mem_group = mem.add_mutually_exclusive_group(required=True)
    mem_group.add_argument("--text", help="Raw user text to extract memory from")
    mem_group.add_argument("--summary", help="Explicit memory summary to store")
    mem.add_argument("--target", choices=["USER", "COMPANY"], help="Target for --summary")

    demo = sub.add_parser("demo", help="Quick demo flow")
    demo.add_argument("--input", nargs="+", required=True, help="File(s) or dir(s)")
    demo.add_argument("--question", required=True, help="Question to ask")
    demo.add_argument("--index", required=True, help="Output index path (json)")

    return parser.parse_args()


def cmd_ingest(args: argparse.Namespace) -> None:
    inputs = [Path(p) for p in args.input]
    files = rag.collect_input_files(inputs)
    if not files:
        raise SystemExit("No supported files found. Use .txt or .md")
    index = rag.build_index(files)
    rag.save_index(index, Path(args.index))
    print(f"Ingested {len(files)} files into {args.index}")


def cmd_ask(args: argparse.Namespace) -> None:
    index = rag.load_index(Path(args.index))
    hits = rag.search(index, args.question, top_k=args.top_k)
    answer, citations = rag.build_answer(hits)
    if args.json:
        print(json.dumps({"question": args.question, "answer": answer, "citations": citations}, indent=2))
        return

    print("Answer:")
    print(answer)
    if citations:
        print("\nCitations:")
        for c in citations:
            print(f"- {c['source']} ({c['locator']}) :: {c['snippet']}")


def cmd_memory(args: argparse.Namespace) -> None:
    entries: list[dict] = []
    if args.text:
        entries = memory.extract_memories(args.text)
    else:
        if not args.target:
            raise SystemExit("--target is required when using --summary")
        entries = [{"target": args.target, "summary": args.summary, "confidence": 1.0}]

    if not entries:
        print("No high-signal memory found.")
        return

    written = memory.write_memory_entries(entries)
    if not written:
        print("No new memory entries written.")
        return

    print("Wrote memory:")
    for entry in written:
        print(f"- {entry['target']}: {entry['summary']}")


def cmd_demo(args: argparse.Namespace) -> None:
    inputs = [Path(p) for p in args.input]
    files = rag.collect_input_files(inputs)
    if not files:
        raise SystemExit("No supported files found. Use .txt or .md")
    index = rag.build_index(files)
    rag.save_index(index, Path(args.index))
    hits = rag.search(index, args.question, top_k=3)
    answer, citations = rag.build_answer(hits)
    print("Answer:")
    print(answer)
    if citations:
        print("\nCitations:")
        for c in citations:
            print(f"- {c['source']} ({c['locator']}) :: {c['snippet']}")


def main() -> None:
    args = _parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "memory":
        cmd_memory(args)
    elif args.cmd == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
