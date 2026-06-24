import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.retrieval.rrf import reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_chunk_found_by_both_retrievers_ranks_first(self) -> None:
        dense = [{"chunk_id": "shared"}, {"chunk_id": "dense-only"}]
        sparse = [{"chunk_id": "sparse-only"}, {"chunk_id": "shared"}]

        fused = reciprocal_rank_fusion([dense, sparse])

        self.assertEqual(fused[0]["chunk_id"], "shared")
        self.assertEqual(fused[0]["retrieval_sources"], ["dense", "sparse"])


if __name__ == "__main__":
    unittest.main()
