import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.retrieval.pipeline import RetrievalSession
from finalrag.generation.citations import format_source
from finalrag.generation.context_builder import build_grounded_context, generation_assets
from finalrag.generation.gemini_generator import generate_grounded_answer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run domain-routed hybrid retrieval.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dense-limit", type=int, default=50)
    parser.add_argument("--sparse-limit", type=int, default=50)
    parser.add_argument("--fused-limit", type=int, default=50)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller retrieval/rerank limits for lower latency.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print complete parent contexts and matched child evidence.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print retrieval results without calling Gemini.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep SPLADE loaded and answer multiple queries in one process.",
    )
    return parser.parse_args()


def run_query(session: RetrievalSession, query: str, args: argparse.Namespace) -> None:
    started = time.perf_counter()
    dense_limit = 20 if args.fast else args.dense_limit
    sparse_limit = 20 if args.fast else args.sparse_limit
    fused_limit = 20 if args.fast else args.fused_limit
    rerank_limit = min(args.top_k, 5) if args.fast else args.top_k
    result = session.retrieve(
        query,
        dense_limit=dense_limit,
        sparse_limit=sparse_limit,
        fused_limit=fused_limit,
        rerank_limit=rerank_limit,
    )
    if args.retrieval_only and not args.full:
        result["results"] = [
            {
                "parent_id": item["parent_id"],
                "domain": item["domain"],
                "section_title": item["section_title"],
                "rerank_score": item.get("rerank_score"),
                "best_rrf_score": item["best_rrf_score"],
                "matched_chunk_ids": [
                    child["chunk_id"] for child in item["matched_children"]
                ],
            }
            for item in result["results"]
        ]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.retrieval_only:
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        print(f"\nCompleted in {time.perf_counter() - started:.2f}s")
        return

    prompt, citations = build_grounded_context(query, result["results"])
    assets = generation_assets(result["results"])
    answer = generate_grounded_answer(prompt, image_paths=assets["image_paths"])
    print(answer)
    print("\nSources:")
    for citation in citations:
        print(format_source(citation))
    print(f"\nCompleted in {time.perf_counter() - started:.2f}s")


def main() -> None:
    args = arguments()
    if not args.interactive and not args.query:
        raise SystemExit("Provide a query or use --interactive")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Loading reusable retrieval session...")
    load_started = time.perf_counter()
    session = RetrievalSession()
    print(f"Session ready in {time.perf_counter() - load_started:.2f}s")

    if args.query:
        run_query(session, args.query, args)
    if not args.interactive:
        return

    print("Enter a query. Use 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query:
            run_query(session, query, args)


if __name__ == "__main__":
    main()
