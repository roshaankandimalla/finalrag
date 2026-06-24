import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.chunking.hierarchical_chunker import base_child


class DailymedTableRetentionTests(unittest.TestCase):
    def test_clinical_figure_text_keeps_table_channel(self) -> None:
        parent = {
            "parent_id": "parent",
            "document_id": "document",
            "domain": "medical",
            "source_type": "html",
            "source_name": "dailymed_ozempic_prescribing_label",
            "section_title": "Clinical Studies",
            "section_path": ["Clinical Studies"],
            "part_number": 1,
        }
        element = {
            "element_id": "element",
            "sequence": 1,
            "page_number": None,
            "source_url": "https://example.test",
            "source_row_numbers": [],
            "element_type": "text",
        }
        text = (
            "Figure 4. 24-hour plasma glucose profile. "
            "OZEMPIC 1 mg end-of-treatment n=36 and OZEMPIC baseline n=37."
        )

        chunk = base_child(parent, 1, [element], text)

        self.assertIn("table", chunk["modalities"])
        self.assertIn("24-hour plasma glucose", chunk["table_markdown"])
        self.assertIn("<table>", chunk["table_html"])
        self.assertEqual(
            chunk["metadata"]["table_rescue_strategy"],
            "dailymed_clinical_table_or_figure_text",
        )


if __name__ == "__main__":
    unittest.main()
