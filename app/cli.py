from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import memory, rag, weather


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAG + memory CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Ingest files into an index")
    ingest.add_argument("--input", nargs="+", required=True, help="File(s) or dir(s)")
    ingest.add_argument("--index", required=True, help="Output index path (json)")
    ingest.add_argument("--use-embeddings", action="store_true", help="Compute semantic embeddings")
    ingest.add_argument(
        "--embed-model",
        default=rag.EMBED_MODEL_DEFAULT,
        help=f"Embedding model name (default: {rag.EMBED_MODEL_DEFAULT})",
    )

    ask = sub.add_parser("ask", help="Ask a question against an index")
    ask.add_argument("--index", required=True, help="Path to index json")
    ask.add_argument("--question", required=True, help="Question to ask")
    ask.add_argument("--top-k", type=int, default=3, help="Top-K chunks")
    ask.add_argument("--json", action="store_true", help="Output JSON")
    ask.add_argument("--use-embeddings", action="store_true", help="Use semantic embeddings if available")
    ask.add_argument(
        "--embed-model",
        default=rag.EMBED_MODEL_DEFAULT,
        help=f"Embedding model name (default: {rag.EMBED_MODEL_DEFAULT})",
    )

    mem = sub.add_parser("memory", help="Write selective memory")
    mem_group = mem.add_mutually_exclusive_group(required=True)
    mem_group.add_argument("--text", help="Raw user text to extract memory from")
    mem_group.add_argument("--summary", help="Explicit memory summary to store")
    mem.add_argument("--target", choices=["USER", "COMPANY"], help="Target for --summary")

    demo = sub.add_parser("demo", help="Quick demo flow")
    demo.add_argument("--input", nargs="+", required=True, help="File(s) or dir(s)")
    demo.add_argument("--question", required=True, help="Question to ask")
    demo.add_argument("--index", required=True, help="Output index path (json)")
    demo.add_argument("--use-embeddings", action="store_true", help="Compute semantic embeddings")
    demo.add_argument(
        "--embed-model",
        default=rag.EMBED_MODEL_DEFAULT,
        help=f"Embedding model name (default: {rag.EMBED_MODEL_DEFAULT})",
    )

    weather_cmd = sub.add_parser("weather", help="Fetch and analyze Open-Meteo time series")
    weather_cmd.add_argument("--lat", type=float, required=True, help="Latitude")
    weather_cmd.add_argument("--lon", type=float, required=True, help="Longitude")
    weather_cmd.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    weather_cmd.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    weather_cmd.add_argument("--json", action="store_true", help="Output JSON")

    return parser.parse_args()


def cmd_ingest(args: argparse.Namespace) -> None:
    inputs = [Path(p) for p in args.input]
    files = rag.collect_input_files(inputs)
    if not files:
        raise SystemExit("No supported files found. Use .txt, .md, or .pdf")
    index = rag.build_index(files, use_embeddings=args.use_embeddings, embed_model=args.embed_model)
    rag.save_index(index, Path(args.index))
    print(f"Ingested {len(files)} files into {args.index}")


def cmd_ask(args: argparse.Namespace) -> None:
    index = rag.load_index(Path(args.index))
    if args.use_embeddings and not index.get("has_embeddings"):
        print("Embeddings not found in index; falling back to BM25 only.")
    hits = rag.search(
        index,
        args.question,
        top_k=args.top_k,
        use_embeddings=args.use_embeddings,
        embed_model=args.embed_model,
    )
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
        raise SystemExit("No supported files found. Use .txt, .md, or .pdf")
    index = rag.build_index(files, use_embeddings=args.use_embeddings, embed_model=args.embed_model)
    rag.save_index(index, Path(args.index))
    hits = rag.search(index, args.question, top_k=3, use_embeddings=args.use_embeddings, embed_model=args.embed_model)
    answer, citations = rag.build_answer(hits)
    print("Answer:")
    print(answer)
    if citations:
        print("\nCitations:")
        for c in citations:
            print(f"- {c['source']} ({c['locator']}) :: {c['snippet']}")


def cmd_weather(args: argparse.Namespace) -> None:
    result = weather.run_weather_analysis(args.lat, args.lon, args.start, args.end)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    stats = result.get("stats", {})
    print("Weather Summary (Open-Meteo):")
    print(f"Location: {args.lat}, {args.lon}")
    print(f"Range: {args.start} to {args.end}")
    print(f"Mean temp: {stats.get('mean')}")
    print(f"Std dev: {stats.get('std')}")
    print(f"Min: {stats.get('min')}  Max: {stats.get('max')}")
    print(f"Missing: {stats.get('missing')}  Anomalies: {stats.get('anomalies')}")


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
    elif args.cmd == "weather":
        cmd_weather(args)


if __name__ == "__main__":
    main()
