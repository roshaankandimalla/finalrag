import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.embeddings.voyage_embeddings import (  # noqa: E402
    embed_chunk_batch,
    embedding_input_hash,
    prepare_multimodal_input,
    select_safe_batch,
)
import finalrag.embeddings.voyage_embeddings as voyage_embeddings  # noqa: E402


class FakeResult:
    def __init__(self, count: int, dimension: int):
        self.embeddings = [[0.25] * dimension for _ in range(count)]


class FakeClient:
    def __init__(self):
        self.request = None

    def multimodal_embed(self, **kwargs):
        self.request = kwargs
        return FakeResult(len(kwargs["inputs"]), kwargs["output_dimension"])


class VoyageEmbeddingTests(unittest.TestCase):
    def test_text_and_image_are_sent_as_one_multimodal_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "chart.png"
            Image.new("RGB", (20, 20), "white").save(image_path)
            chunk = {
                "chunk_id": "chunk-1",
                "retrieval_text": "Revenue increased during the reporting period.",
                "image_paths": ["chart.png"],
            }

            content, opened = prepare_multimodal_input(chunk, root)
            try:
                self.assertEqual(content[0], chunk["retrieval_text"])
                self.assertIsInstance(content[1], Image.Image)
            finally:
                for image in opened:
                    image.close()

    def test_embedding_hash_changes_when_image_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "chart.png"
            chunk = {
                "chunk_id": "chunk-1",
                "retrieval_text": "Revenue chart.",
                "image_paths": ["chart.png"],
            }
            Image.new("RGB", (20, 20), "white").save(image_path)
            first_hash = embedding_input_hash(chunk, root)
            Image.new("RGB", (20, 20), "black").save(image_path)
            second_hash = embedding_input_hash(chunk, root)
            self.assertNotEqual(first_hash, second_hash)

    def test_embedding_hash_changes_when_image_order_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 20), "white").save(root / "first.png")
            Image.new("RGB", (20, 20), "black").save(root / "second.png")
            chunk = {
                "chunk_id": "chunk-1",
                "retrieval_text": "Two charts.",
                "image_paths": ["first.png", "second.png"],
            }
            first_hash = embedding_input_hash(chunk, root)
            chunk["image_paths"].reverse()
            second_hash = embedding_input_hash(chunk, root)
            self.assertNotEqual(first_hash, second_hash)

    def test_embedding_batch_uses_document_mode_and_requested_dimension(self) -> None:
        client = FakeClient()
        chunks = [
            {
                "chunk_id": "chunk-1",
                "retrieval_text": "Grounded retrieval text.",
                "image_paths": [],
            }
        ]

        records = embed_chunk_batch(
            client,
            chunks,
            model="voyage-multimodal-3.5",
            dimension=2048,
        )

        self.assertEqual(len(records[0]["embedding"]), 2048)
        self.assertEqual(client.request["input_type"], "document")
        self.assertFalse(client.request["truncation"])
        self.assertEqual(client.request["output_dimension"], 2048)

    def test_large_image_is_resized_only_for_embedding_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "large.png"
            Image.new("RGB", (40, 40), "white").save(image_path)
            chunk = {
                "chunk_id": "chunk-1",
                "retrieval_text": "Large chart.",
                "image_paths": ["large.png"],
                "token_count": 10,
            }

            with patch.object(voyage_embeddings, "MAX_IMAGE_PIXELS", 400):
                content, opened = prepare_multimodal_input(chunk, root)
            try:
                self.assertLessEqual(content[1].width * content[1].height, 400)
                with Image.open(image_path) as original:
                    self.assertEqual(original.size, (40, 40))
            finally:
                for image in opened:
                    image.close()

    def test_safe_batch_respects_total_token_budget(self) -> None:
        chunks = [
            {
                "chunk_id": f"chunk-{index}",
                "retrieval_text": "Evidence",
                "image_paths": [],
                "token_count": 20_000,
            }
            for index in range(3)
        ]

        selected = select_safe_batch(chunks, max_inputs=10, max_tokens=35_000)

        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
