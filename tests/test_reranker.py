import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.retrieval.reranker import rerank_parents


class FakeRerankItem:
    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class FakeRerankResult:
    def __init__(self, results: list[FakeRerankItem]) -> None:
        self.results = results


class FakeVoyageClient:
    def __init__(self) -> None:
        self.calls = []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        top_k = kwargs["top_k"]
        return FakeRerankResult(
            [
                FakeRerankItem(index=index, relevance_score=1.0 - index / 10)
                for index in range(top_k)
            ]
        )


class VoyageRerankerTests(unittest.TestCase):
    def test_voyage_reranker_uses_top_k(self) -> None:
        parents = [
            {"parent_id": f"parent-{index}", "rerank_text": f"document {index}"}
            for index in range(12)
        ]
        client = FakeVoyageClient()

        with patch.dict(
            os.environ,
            {"VOYAGE_RERANK_MODEL": "rerank-2.5"},
            clear=False,
        ):
            reranked = rerank_parents(client, "query", parents, top_k=8)

        self.assertEqual(len(reranked), 8)
        self.assertEqual(client.calls[0]["top_k"], 8)
        self.assertEqual(client.calls[0]["model"], "rerank-2.5")
        self.assertEqual(reranked[0]["parent_id"], "parent-0")
        self.assertEqual(reranked[0]["rerank_provider"], "voyage")


if __name__ == "__main__":
    unittest.main()
