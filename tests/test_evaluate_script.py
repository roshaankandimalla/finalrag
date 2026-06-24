import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "08_evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_script", SCRIPT_PATH)
EVALUATE_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE_SCRIPT)


class EvaluateScriptTests(unittest.TestCase):
    def test_completed_traces_accepts_empty_output_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text("", encoding="utf-8")

            self.assertEqual(EVALUATE_SCRIPT.completed_traces(path), [])

    def test_completed_traces_accepts_invalid_output_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text("{not-json", encoding="utf-8")

            self.assertEqual(EVALUATE_SCRIPT.completed_traces(path), [])

    def test_completed_traces_only_returns_fully_scored_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            complete = {
                "id": "complete",
                "ragas": {
                    "faithfulness": 1.0,
                    "answer_relevancy": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                },
            }
            partial = {"id": "partial", "ragas": {"faithfulness": 1.0}}
            path.write_text(
                json.dumps({"samples": [complete, partial]}),
                encoding="utf-8",
            )

            self.assertEqual(EVALUATE_SCRIPT.completed_traces(path), [complete])


if __name__ == "__main__":
    unittest.main()
