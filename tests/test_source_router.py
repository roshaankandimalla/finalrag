import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.retrieval.source_router import route_sources


class SourceRouterTests(unittest.TestCase):
    def test_routes_ozempic_questions_to_dailymed(self) -> None:
        routing = route_sources(
            "What did the Ozempic 24-hour plasma glucose study show?",
            ["medical"],
        )

        self.assertEqual(
            routing["file_names"], ["dailymed_ozempic_prescribing_label"]
        )
        self.assertEqual(routing["confidence"], "high_single_source")

    def test_routes_hcahps_questions_to_hospital_csv(self) -> None:
        routing = route_sources(
            "What is the HCAHPS nurse communication rating for this hospital?",
            ["medical"],
        )

        self.assertEqual(routing["file_names"], ["HCAHPS-Hospital"])

    def test_no_match_keeps_retrieval_unfiltered(self) -> None:
        routing = route_sources("Tell me something general", ["finance", "legal"])

        self.assertEqual(routing["file_names"], [])
        self.assertEqual(routing["confidence"], "no_source_filter")


if __name__ == "__main__":
    unittest.main()
