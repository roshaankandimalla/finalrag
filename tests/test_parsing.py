import sys
import unittest
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.parsing.csv_parser import (
    build_hospital_outputs,
    category_for_measure,
    row_result,
)
from finalrag.parsing.html_parser import extract_structured_page, table_to_markdown
from finalrag.parsing.pdf_parser import build_parse_options


class ParsingTests(unittest.TestCase):
    def test_pdf_parse_options_keep_agentic_tables_and_images(self) -> None:
        options = build_parse_options()

        self.assertEqual(options["tier"], "agentic")
        self.assertEqual(options["version"], "latest")
        self.assertTrue(
            options["processing_options"]["aggressive_table_extraction"]
        )
        self.assertEqual(
            options["processing_options"]["specialized_chart_parsing"],
            "agentic",
        )
        self.assertEqual(
            options["output_options"]["images_to_save"],
            ["embedded", "layout"],
        )
        self.assertTrue(
            options["output_options"]["markdown"]["tables"][
                "output_tables_as_markdown"
            ]
        )

    def test_html_table_to_markdown_escapes_pipe_characters(self) -> None:
        soup = BeautifulSoup(
            """
            <table>
              <tr><th>Drug</th><th>Value</th></tr>
              <tr><td>A|B</td><td>10</td></tr>
            </table>
            """,
            "lxml",
        )

        markdown = table_to_markdown(soup.find("table"))

        self.assertIn("| Drug | Value |", markdown)
        self.assertIn("| A\\|B | 10 |", markdown)

    def test_html_structured_page_groups_content_under_headings(self) -> None:
        page = {
            "html": """
                <html><body>
                    <h1>Prescribing Information</h1>
                    <p>Important administration instructions.</p>
                    <h2>Dosage</h2>
                    <table><tr><th>Dose</th></tr><tr><td>0.25 mg</td></tr></table>
                    <img src="/label.png" alt="Dose pen">
                </body></html>
            """,
            "metadata": {
                "sourceURL": "https://example.com/drug",
                "title": "Drug Label",
            },
        }

        structured = extract_structured_page(page, page_number=1)
        sections = structured["sections"]

        self.assertEqual(structured["source_url"], "https://example.com/drug")
        self.assertTrue(any(section["heading"] == "Dosage" for section in sections))
        dosage = next(section for section in sections if section["heading"] == "Dosage")
        self.assertEqual(dosage["section_path"], ["Prescribing Information", "Dosage"])
        self.assertTrue(any(item["type"] == "table" for item in dosage["elements"]))
        self.assertTrue(any(item["type"] == "image" for item in dosage["elements"]))

    def test_csv_measure_category_and_result_selection(self) -> None:
        self.assertEqual(category_for_measure("H_COMP_1_A_P"), "nurse_communication")
        label, value = row_result(
            {
                "Patient Survey Star Rating": "Not Available",
                "HCAHPS Answer Percent": "84",
                "HCAHPS Linear Mean Value": "92",
            }
        )

        self.assertEqual(label, "answer percent")
        self.assertEqual(value, "84%")

    def test_csv_hospital_outputs_create_profile_and_category_docs(self) -> None:
        rows = [
            {
                "Facility ID": "100001",
                "Facility Name": "Example Hospital",
                "Address": "1 Main St",
                "City/Town": "Austin",
                "State": "TX",
                "ZIP Code": "78701",
                "County/Parish": "Travis",
                "Telephone Number": "555-0100",
                "HCAHPS Measure ID": "H_COMP_1_A_P",
                "HCAHPS Question": "Nurses explained things clearly",
                "HCAHPS Answer Description": "Always",
                "Patient Survey Star Rating": "4",
                "HCAHPS Answer Percent": "",
                "HCAHPS Linear Mean Value": "",
                "Number of Completed Surveys": "300",
                "Survey Response Rate Percent": "22",
                "Start Date": "01/01/2025",
                "End Date": "12/31/2025",
                "_source_row_number": 2,
            }
        ]

        profile, category_docs = build_hospital_outputs(
            rows,
            "document-1",
            "medical",
            "HCAHPS-Hospital.csv",
        )

        self.assertEqual(profile["facility_id"], "100001")
        self.assertIn("Available HCAHPS categories:", profile["retrieval_text"])
        self.assertEqual(len(category_docs), 1)
        self.assertEqual(category_docs[0]["category"], "nurse_communication")
        self.assertIn("| Measure ID | Measure |", category_docs[0]["table_markdown"])


if __name__ == "__main__":
    unittest.main()
