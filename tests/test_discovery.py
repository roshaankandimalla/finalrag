import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.discovery.file_discovery import (
    create_document_id,
    create_url_name,
    discover_documents,
)


class FileDiscoveryTests(unittest.TestCase):
    def test_document_id_is_stable_and_case_insensitive_for_location(self) -> None:
        first = create_document_id("finance", "pdf", "DATA/Input/Finance/RIL.pdf")
        second = create_document_id("finance", "pdf", "data/input/finance/ril.pdf")

        self.assertEqual(first, second)

    def test_url_name_uses_meaningful_path_or_domain_fallback(self) -> None:
        self.assertEqual(
            create_url_name("https://example.com/reports/msft-2025.htm"),
            "msft-2025",
        )
        self.assertEqual(
            create_url_name("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"),
            "dailymed_nlm_nih_gov",
        )

    def test_discover_documents_routes_local_files_and_firecrawl_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "data" / "input"
            finance = input_dir / "finance"
            medical = input_dir / "medical"
            finance.mkdir(parents=True)
            medical.mkdir(parents=True)

            (finance / "report.pdf").write_bytes(b"%PDF-1.4")
            (finance / "ignore.txt").write_text("not supported", encoding="utf-8")
            (medical / "hospitals.csv").write_text("Facility ID\n1\n", encoding="utf-8")
            (medical / "ozempic.json").write_text(
                json.dumps(
                    {
                        "name": "ozempic_label",
                        "firecrawl_config": {
                            "url": "https://dailymed.nlm.nih.gov/example"
                        },
                    }
                ),
                encoding="utf-8",
            )

            documents = discover_documents(input_dir, root)
            by_name = {document.file_name: document for document in documents}

        self.assertEqual(set(by_name), {"report.pdf", "hospitals.csv", "ozempic_label"})
        self.assertEqual(by_name["report.pdf"].parser_used, "llamaparse")
        self.assertEqual(by_name["hospitals.csv"].parser_used, "pandas")
        self.assertEqual(by_name["ozempic_label"].source_type, "html")
        self.assertEqual(by_name["ozempic_label"].parser_used, "firecrawl")
        self.assertEqual(
            by_name["report.pdf"].file_path,
            "data/input/finance/report.pdf",
        )


if __name__ == "__main__":
    unittest.main()
