import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.retrieval.domain_router import ALL_DOMAINS, select_routed_domains


class DomainRouterTests(unittest.TestCase):
    def test_high_confidence_routes_one_domain(self) -> None:
        domains, confidence = select_routed_domains(
            {"medical": 0.70, "finance": 0.50, "legal": 0.48},
            minimum=0.25,
            margin=0.03,
        )
        self.assertEqual(domains, ["medical"])
        self.assertEqual(confidence, "high_single_domain")

    def test_ambiguous_query_routes_two_domains(self) -> None:
        domains, confidence = select_routed_domains(
            {"medical": 0.70, "legal": 0.69, "finance": 0.40},
            minimum=0.25,
            margin=0.03,
        )
        self.assertEqual(domains, ["medical", "legal"])
        self.assertEqual(confidence, "ambiguous_two_domains")

    def test_low_confidence_routes_all_domains(self) -> None:
        domains, confidence = select_routed_domains(
            {"medical": 0.20, "legal": 0.19, "finance": 0.18},
            minimum=0.25,
            margin=0.03,
        )
        self.assertEqual(domains, ALL_DOMAINS)
        self.assertEqual(confidence, "low_all_domains")


if __name__ == "__main__":
    unittest.main()
