import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.normalization.normalize import (
    clean_text,
    element,
    page_from_filename,
    stable_id,
)


class NormalizationTests(unittest.TestCase):
    def test_clean_text_collapses_whitespace(self) -> None:
        self.assertEqual(clean_text("  A\n\n table\t value  "), "A table value")
        self.assertEqual(clean_text(None), "")

    def test_page_from_filename_supports_llamaparse_patterns(self) -> None:
        self.assertEqual(page_from_filename("report_p145_table_1.png"), 145)
        self.assertEqual(page_from_filename("page_12_image.png"), 12)
        self.assertIsNone(page_from_filename("image_without_page.png"))

    def test_stable_id_is_deterministic(self) -> None:
        self.assertEqual(stable_id("doc", "page", "1"), stable_id("doc", "page", "1"))
        self.assertNotEqual(stable_id("doc", "page", "1"), stable_id("doc", "page", "2"))

    def test_element_schema_preserves_multimodal_fields(self) -> None:
        item = element(
            element_id="element-1",
            document_id="document-1",
            domain="medical",
            source_type="html",
            source_name="dailymed",
            sequence=7,
            element_type="table",
            section_title="Dosage",
            section_path=["Prescribing Information", "Dosage"],
            page_number=2,
            text="Dose table",
            table_markdown="| Dose |",
            table_html="<table></table>",
            image_path="data/images/medical/dailymed/image.png",
            source_row_numbers=[10, 11],
            metadata={"audience": "professional"},
        )

        self.assertEqual(item["element_type"], "table")
        self.assertEqual(item["table_markdown"], "| Dose |")
        self.assertEqual(item["table_html"], "<table></table>")
        self.assertEqual(item["image_path"], "data/images/medical/dailymed/image.png")
        self.assertEqual(item["source_row_numbers"], [10, 11])
        self.assertEqual(item["metadata"]["audience"], "professional")


if __name__ == "__main__":
    unittest.main()
