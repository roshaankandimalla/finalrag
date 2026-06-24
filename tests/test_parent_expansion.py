import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.retrieval.parent_expansion import expand_and_deduplicate_parents


class FakeCursor:
    def __init__(self, parents: list[dict]) -> None:
        self.parents = parents

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, *_args, **_kwargs) -> None:
        return None

    def fetchall(self) -> list[dict]:
        return self.parents


class FakeConnection:
    def __init__(self, parents: list[dict]) -> None:
        self.parents = parents

    def cursor(self, *_args, **_kwargs) -> FakeCursor:
        return FakeCursor(self.parents)


class ParentExpansionTests(unittest.TestCase):
    def test_matched_children_default_is_three(self) -> None:
        parent_id = "11111111-1111-1111-1111-111111111111"
        connection = FakeConnection(
            [
                {
                    "parent_id": parent_id,
                    "document_id": "doc-1",
                    "domain": "finance",
                    "source_type": "pdf",
                    "section_title": "Section",
                    "section_path": ["Section"],
                    "page_numbers": [1],
                    "parent_text": "Parent text",
                    "metadata": {},
                }
            ]
        )
        fused_children = [
            {
                "parent_id": parent_id,
                "rrf_score": 0.2,
                "retrieval_text": "third",
            },
            {
                "parent_id": parent_id,
                "rrf_score": 0.9,
                "retrieval_text": "first",
            },
            {
                "parent_id": parent_id,
                "rrf_score": 0.7,
                "retrieval_text": "second",
            },
        ]

        parents = expand_and_deduplicate_parents(connection, fused_children)

        self.assertEqual(len(parents), 1)
        matched = parents[0]["matched_children"]
        self.assertEqual(len(matched), 3)
        self.assertEqual([item["retrieval_text"] for item in matched], ["first", "second", "third"])


if __name__ == "__main__":
    unittest.main()
