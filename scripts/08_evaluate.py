import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.evaluation.ragas_evaluator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_SLEEP,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_GEMINI_KEYS,
    DEFAULT_SAFE_RPM,
    DEFAULT_TIMEOUT,
    PER_KEY_MAX_WORKERS,
    generate_evaluation_samples,
    load_gemini_api_keys,
    load_questions,
    run_ragas_parallel,
    save_results,
    validate_gemini_api_keys,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end RAGAS evaluation.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "questions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "results.json",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=60 / DEFAULT_SAFE_RPM,
        help="Seconds between Gemini generation and RAGAS metric calls.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Questions processed by each key before its cooldown.",
    )
    parser.add_argument(
        "--batch-sleep",
        type=float,
        default=DEFAULT_BATCH_SLEEP,
        help="Per-key cooldown in seconds after each batch.",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--skip-key-preflight",
        action="store_true",
        help="Do not make a tiny preflight request to validate Gemini keys.",
    )
    parser.add_argument(
        "--retry-fallback-delay",
        type=float,
        default=65.0,
        help="Wait used when a rate-limit response does not specify a delay.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate answers and traces without running RAGAS judge metrics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions already fully scored in the output file.",
    )
    return parser.parse_args()


def completed_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        print(f"Ignoring invalid resume file and starting fresh: {path}")
        return []
    if not isinstance(payload, dict):
        return []
    required_metrics = {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }
    return [
        trace
        for trace in payload.get("samples", [])
        if required_metrics <= trace.get("ragas", {}).keys()
    ]


def main() -> None:
    args = arguments()
    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    selected_ids = {
        item.get("id", f"question-{index}")
        for index, item in enumerate(questions, start=1)
    }
    previous_traces = completed_traces(args.output) if args.resume else []
    previous_traces = [
        trace for trace in previous_traces if trace.get("id") in selected_ids
    ]
    completed_ids = {trace["id"] for trace in previous_traces}
    pending_questions = [
        item
        for index, item in enumerate(questions, start=1)
        if item.get("id", f"question-{index}") not in completed_ids
    ]
    if not pending_questions:
        save_results(
            args.output,
            previous_traces,
            [trace["ragas"] for trace in previous_traces],
        )
        print(f"All selected questions are already scored in {args.output}")
        return

    api_keys = load_gemini_api_keys()
    if not args.skip_key_preflight:
        api_keys = validate_gemini_api_keys(api_keys, timeout=args.timeout)
    print(
        f"Evaluation config: keys={len(api_keys)}, "
        f"max_configured_keys={DEFAULT_MAX_GEMINI_KEYS}, "
        f"per_key_max_workers={PER_KEY_MAX_WORKERS}, "
        f"safe_rpm={DEFAULT_SAFE_RPM}, batch_size={args.batch_size}, "
        f"batch_sleep={args.batch_sleep:.0f}s, max_retries={args.max_retries}, "
        f"timeout={args.timeout:.0f}s"
    )
    samples, new_traces = generate_evaluation_samples(
        pending_questions,
        rerank_limit=args.top_k,
        request_delay=args.request_delay,
        max_retries=args.max_retries,
        retry_fallback_delay=args.retry_fallback_delay,
        timeout=args.timeout,
        gemini_api_keys=api_keys,
        batch_size=args.batch_size,
        batch_sleep=args.batch_sleep,
    )
    traces = previous_traces + new_traces
    if args.generate_only:
        payload = {"samples": traces}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Generated evaluation traces: {args.output}")
        return

    partial_rows = [trace.get("ragas") for trace in traces]

    def checkpoint(index: int, row: dict) -> None:
        partial_rows[len(previous_traces) + index] = row
        save_results(args.output, traces, partial_rows)
        completed = sum(score is not None for score in partial_rows)
        print(f"Checkpointed {completed}/{len(traces)} samples: {args.output}")

    result = run_ragas_parallel(
        samples,
        api_keys=api_keys,
        request_delay=args.request_delay,
        batch_size=args.batch_size,
        batch_sleep=args.batch_sleep,
        max_retries=args.max_retries,
        retry_fallback_delay=args.retry_fallback_delay,
        timeout=args.timeout,
        on_sample_scored=checkpoint,
    )
    payload = save_results(args.output, traces, partial_rows)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved evaluation results: {args.output}")


if __name__ == "__main__":
    main()
