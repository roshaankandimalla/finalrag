import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.database.repository import get_parser_version, upsert_document
from finalrag.discovery.file_discovery import DiscoveredDocument


class FakeCursor:
    def __init__(self) -> None:
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None) -> None:
        self.executed.append((query, params))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()

    def cursor(self):
        return self.cursor_instance


class RepositoryTests(unittest.TestCase):
    def test_unknown_parser_version_is_none(self) -> None:
        self.assertIsNone(get_parser_version("custom-parser"))

    @patch("finalrag.database.repository.version", return_value="2.9.0")
    def test_known_parser_version_uses_package_mapping(self, mocked_version) -> None:
        self.assertEqual(get_parser_version("llamaparse"), "2.9.0")
        mocked_version.assert_called_once_with("llama-cloud")

    @patch("finalrag.database.repository.get_parser_version", return_value="1.2.3")
    def test_upsert_document_writes_document_registry_row(self, _mocked_version) -> None:
        connection = FakeConnection()
        document = DiscoveredDocument(
            document_id=uuid.uuid4(),
            domain="finance",
            file_name="report.pdf",
            source_type="pdf",
            file_path="data/input/finance/report.pdf",
            parser_used="llamaparse",
            metadata={"file_size_bytes": 123},
        )

        upsert_document(connection, document)

        query, params = connection.cursor_instance.executed[0]
        self.assertIn("INSERT INTO documents", query)
        self.assertIn("ON CONFLICT (document_id) DO UPDATE", query)
        self.assertEqual(params["document_id"], document.document_id)
        self.assertEqual(params["domain"], "finance")
        self.assertEqual(params["source_type"], "pdf")
        self.assertEqual(params["parser_used"], "llamaparse")
        self.assertEqual(params["parser_version"], "1.2.3")


if __name__ == "__main__":
    unittest.main()
