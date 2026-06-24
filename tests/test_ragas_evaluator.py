import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from finalrag.evaluation.ragas_evaluator import (
    is_rate_limit_error,
    is_retryable_error,
    generate_evaluation_samples,
    is_auth_error,
    clean_metric_value,
    load_questions,
    retry_delay_seconds,
    run_ragas_parallel,
    save_results,
)


class RagasEvaluatorTests(unittest.TestCase):
    def test_load_questions_requires_reference_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            path.write_text(json.dumps([{"question": "What?"}]), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_questions(path)

    def test_rate_limit_detection_and_retry_delay(self):
        error = RuntimeError("429 RESOURCE_EXHAUSTED retryDelay: '52s'")

        self.assertTrue(is_rate_limit_error(error))
        self.assertEqual(retry_delay_seconds(error), 54.0)
        self.assertTrue(is_retryable_error(TimeoutError("timed out")))

    def test_auth_error_is_not_retryable(self):
        error = RuntimeError("401 UNAUTHENTICATED ACCESS_TOKEN_TYPE_UNSUPPORTED")

        self.assertTrue(is_auth_error(error))
        self.assertFalse(is_retryable_error(error))

    def test_save_results_supports_partial_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            traces = [{"id": "one"}, {"id": "two"}]

            payload = save_results(path, traces, [{"faithfulness": 0.8}])

            self.assertEqual(payload["progress"], {"completed": 1, "total": 2})
            self.assertEqual(payload["summary"]["faithfulness"], 0.8)
            self.assertNotIn("ragas", payload["samples"][1])
            self.assertEqual(
                payload["metrics_report"]["per_question"][0]["metrics"],
                {"faithfulness": 0.8},
            )
            self.assertEqual(
                payload["metrics_report"]["averages"],
                {"faithfulness": 0.8},
            )

    def test_save_results_converts_nan_to_null_and_excludes_from_average(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            traces = [{"id": "one"}, {"id": "two"}]

            payload = save_results(
                path,
                traces,
                [
                    {"faithfulness": math.nan, "context_recall": 1.0},
                    {"faithfulness": 0.5, "context_recall": math.nan},
                ],
            )

            self.assertIsNone(payload["samples"][0]["ragas"]["faithfulness"])
            self.assertEqual(payload["summary"]["faithfulness"], 0.5)
            self.assertEqual(payload["summary"]["context_recall"], 1.0)
            self.assertIsNone(clean_metric_value(float("nan")))

    def test_parallel_ragas_preserves_original_sample_order(self):
        def fake_run_ragas(samples, on_sample_scored, **kwargs):
            on_sample_scored(1, {"sample": samples[0]})
            return [{"sample": samples[0]}]

        with patch(
            "finalrag.evaluation.ragas_evaluator.run_ragas",
            side_effect=fake_run_ragas,
        ):
            rows = run_ragas_parallel(
                ["zero", "one", "two", "three", "four"],
                api_keys=["key-a", "key-b"],
                batch_size=1,
                batch_sleep=0,
            )

        self.assertEqual(
            rows,
            [
                {"sample": "zero"},
                {"sample": "one"},
                {"sample": "two"},
                {"sample": "three"},
                {"sample": "four"},
            ],
        )

    def test_parallel_ragas_checkpoints_worker_failure(self):
        def fake_run_ragas(samples, on_sample_scored, **kwargs):
            raise RuntimeError("bad key")

        callbacks = []
        with patch(
            "finalrag.evaluation.ragas_evaluator.run_ragas",
            side_effect=fake_run_ragas,
        ):
            rows = run_ragas_parallel(
                ["zero"],
                api_keys=["key-a"],
                batch_size=1,
                batch_sleep=0,
                on_sample_scored=lambda index, row: callbacks.append((index, row)),
            )

        self.assertEqual(rows[0]["errors"]["worker"], "bad key")
        self.assertEqual(rows[0]["errors"]["gemini_key_slot"], 1)
        self.assertEqual(callbacks[0][0], 0)

    def test_generation_uses_key_workers_in_parallel(self):
        questions = [
            {
                "id": f"question-{index}",
                "question": f"Question {index}",
                "reference_answer": f"Reference {index}",
                "domain": "finance",
            }
            for index in range(4)
        ]
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def fake_generate(prompt, api_key, **kwargs):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return f"Answer from {api_key}"

        fake_retrieval = {
            "routing": {},
            "counts": {},
            "results": [],
        }
        with (
            patch("finalrag.evaluation.ragas_evaluator.RetrievalSession") as session,
            patch(
                "finalrag.evaluation.ragas_evaluator.generate_grounded_answer",
                side_effect=fake_generate,
            ),
        ):
            session.return_value.retrieve.return_value = fake_retrieval
            _, traces = generate_evaluation_samples(
                questions,
                gemini_api_keys=["key-1", "key-2", "key-3", "key-4"],
                batch_sleep=0,
            )

        self.assertGreater(maximum_active, 1)
        self.assertEqual(
            [trace["gemini_key_slot"] for trace in traces],
            [1, 2, 3, 4],
        )


if __name__ == "__main__":
    unittest.main()
