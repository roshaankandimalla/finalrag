import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import finalrag.generation.gemini_generator as gemini_generator


class GeminiGeneratorTests(unittest.TestCase):
    def test_generation_contents_include_base64_backed_inline_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "chart.png"
            Image.new("RGB", (8, 8), "white").save(image_path)

            with patch.object(gemini_generator, "PROJECT_ROOT", root):
                contents = gemini_generator.build_generation_contents(
                    "Use the evidence.",
                    image_paths=["chart.png"],
                )

            parts = contents[0].parts
            self.assertEqual(parts[0].text, "Use the evidence.")
            self.assertEqual(parts[1].inline_data.mime_type, "image/png")
            self.assertGreater(len(parts[1].inline_data.data), 0)

    def test_generation_contents_limit_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("one.png", "two.png"):
                Image.new("RGB", (8, 8), "white").save(root / name)

            with patch.object(gemini_generator, "PROJECT_ROOT", root):
                contents = gemini_generator.build_generation_contents(
                    "Use the evidence.",
                    image_paths=["one.png", "two.png"],
                    max_images=1,
                )

            self.assertEqual(len(contents[0].parts), 2)


if __name__ == "__main__":
    unittest.main()
