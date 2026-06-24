import asyncio
import json
import math
import numbers
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

from dotenv import load_dotenv
from google import genai
from ragas import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from voyageai import Client as VoyageClient

from finalrag.generation.citations import citation_for_parent
from finalrag.generation.context_builder import (
    build_grounded_context,
    context_blocks,
    generation_assets,
)
from finalrag.generation.gemini_generator import generate_grounded_answer
from finalrag.embeddings.voyage_embeddings import create_client as create_voyage_client
from finalrag.retrieval.pipeline import RetrievalSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
T = TypeVar("T")

RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "quota exceeded", "rate limit")
AUTH_ERROR_MARKERS = (
    "401",
    "UNAUTHENTICATED",
    "ACCESS_TOKEN_TYPE_UNSUPPORTED",
    "API_KEY_INVALID",
    "invalid authentication",
    "invalid api key",
)
TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection is closed",
    "connection reset",
    "temporarily unavailable",
    "503",
)
RETRY_DELAY_PATTERNS = (
    re.compile(r"retry(?:Delay| delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I),
    re.compile(r"retry in (\d+(?:\.\d+)?)", re.I),
)
DEFAULT_SAFE_RPM = 13
DEFAULT_BATCH_SIZE = 1
DEFAULT_BATCH_SLEEP = 60.0
DEFAULT_TIMEOUT = 400.0
DEFAULT_MAX_RETRIES = 10
PER_KEY_MAX_WORKERS = 1
DEFAULT_MAX_GEMINI_KEYS = 6
OUT_OF_SCOPE_DOMAIN = "out_of_scope"


class VoyageRagasEmbeddings(BaseRagasEmbedding):
    def __init__(self, client: VoyageClient, model: str, dimension: int = 1024):
        super().__init__()
        self.client = client
        self.model = model
        self.dimension = dimension

    def embed_text(self, text: str, **kwargs) -> list[float]:
        result = self.client.multimodal_embed(
            inputs=[[text]],
            model=self.model,
            input_type="query",
            truncation=True,
            output_dtype="float",
            output_dimension=self.dimension,
        )
        return result.embeddings[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text, **kwargs)


class AsyncFromSyncJudge(InstructorBaseRagasLLM):
    def __init__(self, judge: InstructorBaseRagasLLM):
        self.judge = judge

    def generate(self, prompt: str, response_model):
        return self.judge.generate(prompt, response_model)

    async def agenerate(self, prompt: str, response_model):
        return await asyncio.to_thread(self.judge.generate, prompt, response_model)


def load_questions(path: Path) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    for index, item in enumerate(questions, start=1):
        missing = {"question", "reference_answer"} - item.keys()
        if missing:
            raise ValueError(f"Question {index} missing fields: {sorted(missing)}")
    return questions


def load_gemini_api_keys(max_keys: int = DEFAULT_MAX_GEMINI_KEYS) -> list[str]:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    keys = [
        os.getenv(f"GEMINI_API_KEY_{index}")
        for index in range(1, max_keys + 1)
    ]
    configured = [key for key in keys if key]
    if not configured and os.getenv("GEMINI_API_KEY"):
        configured = [os.environ["GEMINI_API_KEY"]]
    if not configured:
        raise RuntimeError(
            "Configure GEMINI_API_KEY_1 through GEMINI_API_KEY_6, "
            "or set GEMINI_API_KEY"
        )
    return configured


def clean_metric_value(value):
    if isinstance(value, numbers.Number):
        numeric_value = float(value)
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None
    return value


def metric_averages(traces: list[dict]) -> dict:
    metric_names = sorted(
        {
            metric
            for trace in traces
            for metric in trace.get("ragas", {})
            if isinstance(clean_metric_value(trace["ragas"][metric]), numbers.Number)
        }
    )
    averages = {}
    for metric in metric_names:
        values = [
            float(clean_metric_value(trace["ragas"][metric]))
            for trace in traces
            if isinstance(
                clean_metric_value(trace.get("ragas", {}).get(metric)),
                numbers.Number,
            )
        ]
        if values:
            averages[metric] = sum(values) / len(values)
    return averages


def grouped_metric_averages(traces: list[dict], group_key: str) -> dict:
    groups = {}
    for trace in traces:
        groups.setdefault(trace.get(group_key) or "unknown", []).append(trace)
    return {
        group: {
            "count": len(items),
            "averages": metric_averages(items),
        }
        for group, items in sorted(groups.items())
    }


def is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker.lower() in message for marker in RATE_LIMIT_MARKERS)


def is_auth_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker.lower() in message for marker in AUTH_ERROR_MARKERS)


def is_retryable_error(error: Exception) -> bool:
    if is_auth_error(error):
        return False
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return True
    message = str(error).lower()
    return is_rate_limit_error(error) or any(
        marker in message for marker in TRANSIENT_ERROR_MARKERS
    )


def retry_delay_seconds(error: Exception, fallback: float = 65.0) -> float:
    message = str(error)
    for pattern in RETRY_DELAY_PATTERNS:
        match = pattern.search(message)
        if match:
            return max(float(match.group(1)) + 2.0, 1.0)
    return fallback


def call_with_rate_limit_retry(
    label: str,
    operation: Callable[[], T],
    max_retries: int,
    fallback_delay: float,
) -> T:
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            if not is_retryable_error(error) or attempt >= max_retries:
                raise
            delay = retry_delay_seconds(error, fallback=fallback_delay)
            print(f"{label} transient failure; retrying in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after retries")


def validate_gemini_api_keys(
    api_keys: list[str],
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[str]:
    selected_model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    valid_keys = []
    for index, api_key in enumerate(api_keys, start=1):
        try:
            client = genai.Client(
                api_key=api_key,
                http_options={"timeout": int(timeout * 1_000)},
            )
            client.models.generate_content(
                model=selected_model,
                contents="Return exactly: ok",
            )
            valid_keys.append(api_key)
        except Exception as error:
            if is_auth_error(error):
                print(f"Skipping Gemini key slot {index}: authentication failed")
                continue
            print(f"Could not preflight Gemini key slot {index}; keeping it: {error}")
            valid_keys.append(api_key)
    if not valid_keys:
        raise RuntimeError("No valid Gemini API keys are available")
    return valid_keys


async def acall_with_rate_limit_retry(
    label: str,
    operation: Callable[[], Awaitable[T]],
    max_retries: int,
    fallback_delay: float,
    timeout: float,
) -> T:
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(operation(), timeout=timeout)
        except Exception as error:
            if not is_retryable_error(error) or attempt >= max_retries:
                raise
            delay = retry_delay_seconds(error, fallback=fallback_delay)
            print(f"{label} transient failure; retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
    raise RuntimeError(f"{label} failed after retries")


def generate_evaluation_samples(
    questions: list[dict],
    rerank_limit: int = 8,
    request_delay: float = 60 / DEFAULT_SAFE_RPM,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_fallback_delay: float = 65.0,
    timeout: float = DEFAULT_TIMEOUT,
    gemini_api_keys: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_sleep: float = DEFAULT_BATCH_SLEEP,
) -> tuple[list[SingleTurnSample], list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    session = RetrievalSession()
    api_keys = gemini_api_keys or load_gemini_api_keys()
    prepared = []
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[{index}/{len(questions)}] Retrieving evidence: {question}")
        retrieval = session.retrieve(question, rerank_limit=rerank_limit)
        prompt, citations = build_grounded_context(question, retrieval["results"])
        assets = generation_assets(retrieval["results"])
        contexts = context_blocks(retrieval["results"])
        prepared.append(
            {
                "index": index - 1,
                "item": item,
                "question": question,
                "retrieval": retrieval,
                "prompt": prompt,
                "citations": citations,
                "assets": assets,
                "contexts": contexts,
            }
        )

    assignments = [[] for _ in api_keys]
    for record in prepared:
        assignments[record["index"] % len(api_keys)].append(record)
    answers: list[str | None] = [None] * len(prepared)

    def generation_worker(key_slot: int, api_key: str, assigned: list[dict]) -> None:
        for batch_start in range(0, len(assigned), batch_size):
            batch = assigned[batch_start : batch_start + batch_size]
            for record in batch:
                question_number = record["index"] + 1
                print(
                    f"[key_{key_slot}] Generating answer "
                    f"{question_number}/{len(questions)}"
                )
                answers[record["index"]] = call_with_rate_limit_retry(
                    f"[key_{key_slot}] Gemini generation",
                    lambda record=record: generate_grounded_answer(
                        record["prompt"],
                        api_key=api_key,
                        timeout_seconds=timeout,
                        image_paths=record["assets"]["image_paths"],
                    ),
                    max_retries=max_retries,
                    fallback_delay=retry_fallback_delay,
                )
                if request_delay > 0 and len(batch) > 1:
                    time.sleep(request_delay)
            has_more = batch_start + batch_size < len(assigned)
            if has_more and batch_sleep > 0:
                print(f"[key_{key_slot}] Generation sleeping {batch_sleep:.0f}s")
                time.sleep(batch_sleep)

    with ThreadPoolExecutor(
        max_workers=len(api_keys) * PER_KEY_MAX_WORKERS
    ) as executor:
        futures = [
            executor.submit(generation_worker, slot, api_key, assigned)
            for slot, (api_key, assigned) in enumerate(
                zip(api_keys, assignments),
                start=1,
            )
            if assigned
        ]
        for future in as_completed(futures):
            future.result()

    samples = []
    traces = []
    for record, answer in zip(prepared, answers):
        item = record["item"]
        retrieval = record["retrieval"]
        contexts = record["contexts"]
        assets = record["assets"]
        if answer is None:
            raise RuntimeError(f"Missing generated answer for {record['question']}")
        samples.append(
            SingleTurnSample(
                user_input=record["question"],
                response=answer,
                retrieved_contexts=contexts,
                reference=item["reference_answer"],
                reference_contexts=item.get("reference_contexts"),
            )
        )
        traces.append(
            {
                "id": item.get("id", f"question-{record['index'] + 1}"),
                "domain": item.get("domain"),
                "difficulty": item.get("difficulty"),
                "question": record["question"],
                "reference_answer": item["reference_answer"],
                "answer": answer,
                "routing": retrieval["routing"],
                "source_routing": retrieval.get("source_routing"),
                "retrieval_counts": retrieval["counts"],
                "citations": [
                    citation_for_parent(parent, citation_index)
                    for citation_index, parent in enumerate(
                        retrieval["results"], start=1
                    )
                ],
                "retrieved_contexts": contexts,
                "generation_assets": {
                    "image_paths": assets["image_paths"],
                    "table_html_count": len(assets["table_html"]),
                },
                "gemini_key_slot": (record["index"] % len(api_keys)) + 1,
            }
        )
    return samples, traces


async def _score_sample(
    sample: SingleTurnSample,
    metrics: dict,
    request_delay: float,
    max_retries: int,
    retry_fallback_delay: float,
    timeout: float,
) -> dict:
    operations = {
        "faithfulness": lambda: metrics["faithfulness"].ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        ),
        "answer_relevancy": lambda: metrics["answer_relevancy"].ascore(
            user_input=sample.user_input,
            response=sample.response,
        ),
        "context_precision": lambda: metrics["context_precision"].ascore(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        ),
        "context_recall": lambda: metrics["context_recall"].ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            reference=sample.reference,
        ),
    }
    scores = {}
    errors = {}
    for index, (name, operation) in enumerate(operations.items(), start=1):
        try:
            result = await acall_with_rate_limit_retry(
                name,
                operation,
                max_retries=max_retries,
                fallback_delay=retry_fallback_delay,
                timeout=timeout,
            )
            scores[name] = result.value
        except Exception as error:
            errors[name] = str(error)
            print(f"{name} failed: {error}")
        if request_delay > 0 and index < len(operations):
            await asyncio.sleep(request_delay)
    if errors:
        scores["errors"] = errors
    return scores


def run_ragas(
    samples: list[SingleTurnSample],
    request_delay: float = 60 / DEFAULT_SAFE_RPM,
    sample_delay: float = DEFAULT_BATCH_SLEEP,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_fallback_delay: float = 65.0,
    timeout: float = DEFAULT_TIMEOUT,
    api_key: str | None = None,
    on_sample_scored: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    gemini_model = os.getenv("RAGAS_JUDGE_MODEL", os.getenv("GEMINI_MODEL"))
    selected_key = api_key or load_gemini_api_keys(max_keys=1)[0]
    gemini_client = genai.Client(
        api_key=selected_key,
        http_options={"timeout": int(timeout * 1_000)},
    )
    judge = AsyncFromSyncJudge(
        llm_factory(
            gemini_model,
            provider="google",
            client=gemini_client,
            temperature=0,
        )
    )
    voyage_model = os.getenv("VOYAGE_EMBED_MODEL", "voyage-multimodal-3.5")
    voyage_embeddings = VoyageRagasEmbeddings(
        create_voyage_client(),
        model=voyage_model,
        dimension=1024,
    )
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge,
            embeddings=voyage_embeddings,
            strictness=1,
        ),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }

    async def score_all() -> list[dict]:
        rows = []
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] Running RAGAS judge metrics")
            row = await _score_sample(
                sample,
                metrics,
                request_delay=request_delay,
                max_retries=max_retries,
                retry_fallback_delay=retry_fallback_delay,
                timeout=timeout,
            )
            rows.append(row)
            if on_sample_scored is not None:
                on_sample_scored(index, row)
            if sample_delay > 0 and index < len(samples):
                await asyncio.sleep(sample_delay)
        return rows

    return asyncio.run(score_all())


def run_ragas_parallel(
    samples: list[SingleTurnSample],
    api_keys: list[str],
    request_delay: float = 60 / DEFAULT_SAFE_RPM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_sleep: float = DEFAULT_BATCH_SLEEP,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_fallback_delay: float = 65.0,
    timeout: float = DEFAULT_TIMEOUT,
    on_sample_scored: Callable[[int, dict], None] | None = None,
) -> list[dict | None]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    assignments = [[] for _ in api_keys]
    for sample_index, sample in enumerate(samples):
        assignments[sample_index % len(api_keys)].append((sample_index, sample))

    rows: list[dict | None] = [None] * len(samples)
    callback_lock = threading.Lock()

    def run_key_worker(key_slot: int, api_key: str, assigned: list[tuple]) -> None:
        for batch_start in range(0, len(assigned), batch_size):
            batch = assigned[batch_start : batch_start + batch_size]
            batch_samples = [sample for _, sample in batch]
            print(
                f"[key_{key_slot}] Scoring batch "
                f"{batch_start // batch_size + 1} ({len(batch_samples)} sample)"
            )

            def scored(local_index: int, row: dict) -> None:
                global_index = batch[local_index - 1][0]
                rows[global_index] = row
                if on_sample_scored is not None:
                    with callback_lock:
                        on_sample_scored(global_index, row)

            try:
                run_ragas(
                    batch_samples,
                    request_delay=request_delay,
                    sample_delay=0,
                    max_retries=max_retries,
                    retry_fallback_delay=retry_fallback_delay,
                    timeout=timeout,
                    api_key=api_key,
                    on_sample_scored=scored,
                )
            except Exception as error:
                for global_index, _sample in batch:
                    row = {
                        "errors": {
                            "worker": str(error),
                            "gemini_key_slot": key_slot,
                        }
                    }
                    rows[global_index] = row
                    if on_sample_scored is not None:
                        with callback_lock:
                            on_sample_scored(global_index, row)
            has_more = batch_start + batch_size < len(assigned)
            if has_more and batch_sleep > 0:
                print(f"[key_{key_slot}] Sleeping {batch_sleep:.0f}s")
                time.sleep(batch_sleep)

    with ThreadPoolExecutor(
        max_workers=len(api_keys) * PER_KEY_MAX_WORKERS
    ) as executor:
        futures = [
            executor.submit(run_key_worker, slot, api_key, assigned)
            for slot, (api_key, assigned) in enumerate(
                zip(api_keys, assignments),
                start=1,
            )
            if assigned
        ]
        for future in as_completed(futures):
            future.result()
    return rows


def save_results(
    path: Path,
    traces: list[dict],
    score_rows: list[dict | None],
) -> dict:
    for trace, scores in zip(traces, score_rows):
        if scores is None:
            continue
        trace["ragas"] = {
            key: clean_metric_value(value)
            for key, value in scores.items()
            if key not in {"user_input", "retrieved_contexts", "response", "reference"}
        }
    averages = metric_averages(traces)
    in_scope_traces = [
        trace for trace in traces if trace.get("domain") != OUT_OF_SCOPE_DOMAIN
    ]
    out_of_scope_traces = [
        trace for trace in traces if trace.get("domain") == OUT_OF_SCOPE_DOMAIN
    ]
    payload = {
        "summary": averages,
        "summary_in_scope": metric_averages(in_scope_traces),
        "summary_out_of_scope": metric_averages(out_of_scope_traces),
        "summary_by_domain": grouped_metric_averages(traces, "domain"),
        "progress": {
            "completed": sum(row is not None for row in score_rows),
            "total": len(traces),
        },
        "samples": traces,
        "metrics_report": {
            "per_question": [
                {
                    "id": trace.get("id"),
                    "domain": trace.get("domain"),
                    "difficulty": trace.get("difficulty"),
                    "question": trace.get("question"),
                    "metrics": trace.get("ragas", {}),
                }
                for trace in traces
            ],
            "averages": averages,
            "in_scope_averages": metric_averages(in_scope_traces),
            "out_of_scope_averages": metric_averages(out_of_scope_traces),
            "domain_averages": grouped_metric_averages(traces, "domain"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
