import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.generation.citations import citation_for_parent, format_source
from finalrag.generation.context_builder import build_grounded_context, generation_assets


class CitationTests(unittest.TestCase):
    def test_context_and_source_preserve_provenance(self) -> None:
        parent = {
            "parent_id": "parent-1",
            "document_id": "document-1",
            "domain": "finance",
            "source_type": "pdf",
            "section_title": "Financial Performance",
            "parent_text": "Parent evidence",
            "rerank_score": 0.9,
            "matched_children": [
                {
                    "chunk_id": "chunk-1",
                    "file_name": "report.pdf",
                    "page_numbers": [10],
                    "source_row_numbers": [],
                    "retrieval_text": "Revenue increased.",
                }
            ],
        }

        prompt, citations = build_grounded_context("What changed?", [parent])

        self.assertIn("[1]", prompt)
        self.assertIn("Revenue increased.", prompt)
        self.assertEqual(citations[0]["page_numbers"], [10])
        self.assertIn("report.pdf", format_source(citations[0]))

    def test_generation_assets_deduplicate_images_and_tables(self) -> None:
        parent = {
            "matched_children": [
                {
                    "image_paths": ["chart.png"],
                    "table_html": "<table><tr><td>Revenue</td></tr></table>",
                },
                {
                    "image_paths": ["chart.png", "map.png"],
                    "table_html": "<table><tr><td>Revenue</td></tr></table>",
                },
            ]
        }

        assets = generation_assets([parent])

        self.assertEqual(assets["image_paths"], ["chart.png", "map.png"])
        self.assertEqual(len(assets["table_html"]), 1)


if __name__ == "__main__":
    unittest.main()
