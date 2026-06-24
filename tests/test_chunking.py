import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.chunking.hierarchical_chunker import (  # noqa: E402
    CHILD_MAX_TOKENS,
    build_csv_chunks,
    build_document_chunks,
    clip_text,
    token_count,
)


def element(
    element_id: str,
    sequence: int,
    element_type: str,
    text: str | None = None,
    table_markdown: str | None = None,
    page_number: int = 1,
    **overrides,
) -> dict:
    value = {
        "element_id": element_id,
        "document_id": "document-1",
        "domain": "legal",
        "source_type": "pdf",
        "source_name": "test_document",
        "element_type": element_type,
        "sequence": sequence,
        "text": text,
        "table_markdown": table_markdown,
        "table_html": None,
        "image_path": None,
        "page_number": page_number,
        "source_url": None,
        "source_row_numbers": [],
        "section_title": "Test section",
        "section_path": ["Test section"],
        "metadata": {},
    }
    value.update(overrides)
    return value


class HierarchicalChunkerTests(unittest.TestCase):
    def test_token_count_handles_subword_heavy_text(self) -> None:
        self.assertGreater(token_count("antidisestablishmentarianism"), 1)

    def test_clip_text_preserves_beginning_middle_and_end(self) -> None:
        long_text = " ".join(
            [
                "BEGIN_MARKER",
                *["alpha"] * 600,
                "MIDDLE_MARKER",
                *["beta"] * 600,
                "END_MARKER",
            ]
        )

        clipped = clip_text(long_text, max_tokens=150)

        self.assertIn("BEGIN_MARKER", clipped)
        self.assertIn("MIDDLE_MARKER", clipped)
        self.assertIn("END_MARKER", clipped)
        self.assertLessEqual(token_count(clipped), 150)

    def test_long_text_is_split_within_child_limit(self) -> None:
        long_text = " ".join(
            f"Sentence {index} contains useful evidence."
            for index in range(1_000)
        )
        parents, children = build_document_chunks(
            [element("long-text", 1, "text", text=long_text)]
        )

        self.assertGreater(len(parents), 1)
        self.assertGreater(len(children), 1)
        self.assertLessEqual(
            max(child["token_count"] for child in children),
            CHILD_MAX_TOKENS,
        )

    def test_csv_retrieval_avoids_duplicate_markdown_representation(self) -> None:
        retrieval_text = "Hospital evidence " * 100
        table_markdown = "| Measure | Value |\n| --- | --- |\n" + "\n".join(
            f"| Measure {index} | Result {index} |" for index in range(100)
        )
        _, children = build_csv_chunks(
            [
                element(
                    "csv-category",
                    1,
                    "table",
                    text=retrieval_text,
                    table_markdown=table_markdown,
                    source_type="csv",
                    metadata={
                        "profile_id": "profile-1",
                        "facility_id": "facility-1",
                        "hospital_name": "Example Hospital",
                        "state": "CA",
                    },
                )
            ]
        )

        self.assertEqual(children[0]["table_markdown"], table_markdown)
        self.assertNotIn("| Measure | Value |", children[0]["retrieval_text"])
        self.assertLessEqual(children[0]["token_count"], CHILD_MAX_TOKENS)

    def test_hospital_profile_chunk_id_is_distinct_from_parent_id(self) -> None:
        from finalrag.chunking.hierarchical_chunker import stable_id

        parent_id = "profile-1"
        profile_chunk_id = stable_id(parent_id, "hospital_profile_chunk")

        self.assertNotEqual(profile_chunk_id, parent_id)

    def test_oversized_table_is_split_and_traceable(self) -> None:
        rows = "\n".join(
            f"| Measure {index} | {'value ' * 25} |"
            for index in range(200)
        )
        markdown = f"| Measure | Details |\n| --- | --- |\n{rows}"
        _, children = build_document_chunks(
            [
                element(
                    "large-table",
                    1,
                    "table",
                    table_markdown=markdown,
                )
            ]
        )

        self.assertGreater(len(children), 1)
        self.assertTrue(all("table" in child["modalities"] for child in children))
        self.assertTrue(
            all(
                child["metadata"]["source_table_element_id"] == "large-table"
                for child in children
            )
        )
        self.assertLessEqual(
            max(child["token_count"] for child in children),
            CHILD_MAX_TOKENS,
        )

    def test_duplicate_children_merge_page_provenance(self) -> None:
        repeated = " ".join(["Repeated evidence sentence."] * 65)
        _, children = build_document_chunks(
            [
                element("repeat-1", 1, "text", repeated, page_number=1),
                element("repeat-2", 2, "text", repeated, page_number=2),
            ]
        )

        duplicate_children = [
            child
            for child in children
            if child["metadata"]["duplicate_occurrences"] == 2
        ]
        self.assertTrue(duplicate_children)
        self.assertEqual(duplicate_children[0]["page_numbers"], [1, 2])

    def test_broken_markdown_recovers_from_html_rows(self) -> None:
        long_markdown = "broken table " * 500
        table_html = (
            "<table><thead><tr><th>Measure</th><th>Value</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td>Measure {index}</td><td>{'value ' * 20}</td></tr>"
                for index in range(100)
            )
            + "</tbody></table>"
        )
        _, children = build_document_chunks(
            [
                element(
                    "html-table",
                    1,
                    "table",
                    table_markdown=long_markdown,
                    table_html=table_html,
                )
            ]
        )

        self.assertGreater(len(children), 1)
        self.assertTrue(
            all(
                child["metadata"]["table_split_strategy"] == "html_rows"
                for child in children
            )
        )
        self.assertTrue(
            all("Measure" in child["table_markdown"] for child in children)
        )
        self.assertTrue(
            all("Table segment" not in child["table_markdown"] for child in children)
        )

    def test_image_assignment_prefers_same_page_geometry_over_sequence(self) -> None:
        _, children = build_document_chunks(
            [
                element(
                    "page-one-table",
                    1,
                    "table",
                    table_markdown="| Item | Value |\n| --- | --- |\n| Old | 1 |",
                    page_number=1,
                    bbox={"x": 10, "y": 10, "w": 100, "h": 100},
                ),
                element(
                    "image",
                    2,
                    "image",
                    page_number=2,
                    image_path="data/images/test/chart.png",
                    bbox={"x": 500, "y": 500, "w": 100, "h": 100},
                    metadata={"nearby_text_after": "Current result chart"},
                ),
                element(
                    "page-two-table",
                    10,
                    "table",
                    table_markdown="| Item | Value |\n| --- | --- |\n| Current | 2 |",
                    page_number=2,
                    bbox={"x": 510, "y": 510, "w": 100, "h": 100},
                ),
            ]
        )

        image_child = next(child for child in children if child["image_paths"])
        self.assertEqual(image_child["page_numbers"], [2])
        self.assertIn("Current", image_child["table_markdown"])
        details = image_child["metadata"]["image_context"][0]["assignment_details"]
        self.assertTrue(details["same_page"])

    def test_image_without_same_page_candidate_uses_contextual_fallback(self) -> None:
        _, children = build_document_chunks(
            [
                element(
                    "page-one-table",
                    1,
                    "table",
                    table_markdown="| Item | Value |\n| --- | --- |\n| Old | 1 |",
                    page_number=1,
                    bbox={"x": 10, "y": 10, "w": 100, "h": 100},
                ),
                element(
                    "page-two-image",
                    2,
                    "image",
                    page_number=2,
                    image_path="data/images/test/page-two-chart.png",
                    bbox={"x": 500, "y": 500, "w": 100, "h": 100},
                    metadata={"nearby_text_after": "Page two chart evidence"},
                ),
            ]
        )

        image_child = next(child for child in children if child["image_paths"])
        self.assertEqual(image_child["page_numbers"], [2])
        self.assertEqual(image_child["modalities"], ["text", "image"])
        context = image_child["metadata"]["image_context"][0]
        self.assertEqual(context["assignment_method"], "contextual_fallback")

    def test_image_without_same_page_or_context_is_not_forced_cross_page(self) -> None:
        _, children = build_document_chunks(
            [
                element(
                    "page-one-table",
                    1,
                    "table",
                    table_markdown="| Item | Value |\n| --- | --- |\n| Old | 1 |",
                    page_number=1,
                ),
                element(
                    "page-two-image",
                    2,
                    "image",
                    page_number=2,
                    image_path="data/images/test/unrelated.png",
                    metadata={},
                ),
            ]
        )

        self.assertTrue(all(not child["image_paths"] for child in children))

    def test_table_crop_recovers_across_parent_sections_on_same_page(self) -> None:
        _, children = build_document_chunks(
            [
                element(
                    "parsed-table",
                    1,
                    "table",
                    table_markdown="| Item | Value |\n| --- | --- |\n| Current | 2 |",
                    page_number=5,
                    bbox={"x": 10, "y": 10, "w": 500, "h": 300},
                    section_title="Parsed table section",
                    section_path=["Parsed table section"],
                ),
                element(
                    "table-crop",
                    2,
                    "image",
                    page_number=5,
                    image_path="data/images/test/table-crop.png",
                    bbox={"x": 12, "y": 12, "w": 500, "h": 300},
                    section_title="Visual-only section",
                    section_path=["Visual-only section"],
                    metadata={"selection_reason": "layout table or form crop"},
                ),
            ]
        )

        image_child = next(child for child in children if child["image_paths"])
        self.assertEqual(image_child["modalities"], ["text", "image"])
        self.assertIsNotNone(
            image_child["metadata"]["matched_table_chunk_id"]
        )
        self.assertEqual(
            image_child["metadata"]["image_context"][0]["assignment_method"],
            "document_same_page_table_reference",
        )

    def test_visual_table_without_parsed_table_gets_grounded_fallback(self) -> None:
        _, children = build_document_chunks(
            [
                element(
                    "table-crop",
                    1,
                    "image",
                    page_number=8,
                    image_path="data/images/test/table-crop.png",
                    metadata={"selection_reason": "layout table or form crop"},
                )
            ]
        )

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["modalities"], ["text", "image"])
        self.assertIn("Visual table crop", children[0]["text_content"])
        self.assertEqual(
            children[0]["metadata"]["image_context"][0]["assignment_method"],
            "visual_table_fallback",
        )

    def test_identical_text_with_distinct_images_is_not_deduplicated(self) -> None:
        _, children = build_document_chunks(
            [
                element(
                    "table-one",
                    1,
                    "table",
                    table_markdown="| Item | Value |\n| --- | --- |\n| Current | 2 |",
                    page_number=1,
                ),
                element(
                    "image-one",
                    2,
                    "image",
                    page_number=1,
                    image_path="data/images/test/one.png",
                    metadata={"nearby_text_after": "Current value"},
                ),
                element(
                    "table-two",
                    3,
                    "table",
                    table_markdown="| Item | Value |\n| --- | --- |\n| Current | 2 |",
                    page_number=2,
                ),
                element(
                    "image-two",
                    4,
                    "image",
                    page_number=2,
                    image_path="data/images/test/two.png",
                    metadata={"nearby_text_after": "Current value"},
                ),
            ]
        )

        image_paths = {
            path
            for child in children
            for path in child["image_paths"]
        }
        self.assertEqual(
            image_paths,
            {"data/images/test/one.png", "data/images/test/two.png"},
        )

    def test_tiny_layout_chart_icon_is_not_attached(self) -> None:
        _, children = build_document_chunks(
            [
                element(
                    "parsed-table",
                    1,
                    "table",
                    table_markdown="| Item | Value |\n| --- | --- |\n| Current | 2 |",
                    page_number=5,
                ),
                element(
                    "warning-icon",
                    2,
                    "image",
                    page_number=5,
                    image_path="data/images/test/warning.png",
                    bbox={"x": 10, "y": 10, "w": 35, "h": 35},
                    metadata={"selection_reason": "layout chart crop"},
                ),
            ]
        )

        self.assertTrue(all(not child["image_paths"] for child in children))


if __name__ == "__main__":
    unittest.main()
