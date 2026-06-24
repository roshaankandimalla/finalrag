import hashlib
import html
import json
import math
import re
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARENT_MAX_TOKENS = 3_000
CHILD_TARGET_TOKENS = 500
CHILD_MIN_TOKENS = 300
CHILD_MAX_TOKENS = 700
MAX_IMAGES_PER_CHUNK = 3
TEXT_SPLIT_MAX_TOKENS = 520
CHILD_CONTENT_MAX_TOKENS = 560
TABLE_SPLIT_MAX_TOKENS = 520
TOKEN_SAFETY_FACTOR = 1.10


def load_token_encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Restricted/offline environments may not have the vocabulary cached yet.
        return None


TOKEN_ENCODING = load_token_encoding()
TOKENIZER_NAME = "tiktoken:cl100k_base" if TOKEN_ENCODING else "offline_subword_fallback"


def stable_id(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, ":".join(parts)))


def encoded_tokens(text: str | None) -> list[int] | list[str]:
    value = text or ""
    if TOKEN_ENCODING:
        return TOKEN_ENCODING.encode(value, disallowed_special=())
    return re.findall(r"\s*\w{1,8}|\s*[^\w\s]|\s+", value, flags=re.UNICODE)


def decode_tokens(tokens: list[int] | list[str]) -> str:
    if TOKEN_ENCODING:
        return TOKEN_ENCODING.decode(tokens)
    return "".join(tokens)


def token_count(text: str | None) -> int:
    raw_count = len(encoded_tokens(text))
    return math.ceil(raw_count * TOKEN_SAFETY_FACTOR)


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def split_oversized_text(
    text: str | None,
    max_tokens: int = TEXT_SPLIT_MAX_TOKENS,
) -> list[str]:
    cleaned = clean_text(text)
    cleaned = re.sub(r"([^\w\s])\1{20,}", r"\1", cleaned)
    if not cleaned or token_count(cleaned) <= max_tokens:
        return [cleaned] if cleaned else []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    pieces = []
    current = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            pieces.append(" ".join(current))
        current = []
        current_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_count(sentence)
        if sentence_tokens > max_tokens:
            flush()
            word_buffer = []
            word_tokens = 0
            for word in sentence.split():
                count = token_count(word)
                if word_buffer and word_tokens + count > max_tokens:
                    pieces.append(" ".join(word_buffer))
                    word_buffer = []
                    word_tokens = 0
                word_buffer.append(word)
                word_tokens += count
            if word_buffer:
                pieces.append(" ".join(word_buffer))
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            flush()
        current.append(sentence)
        current_tokens += sentence_tokens

    flush()
    return pieces


def clip_text(text: str | None, max_tokens: int = 560) -> str:
    cleaned = clean_text(text)
    if token_count(cleaned) <= max_tokens:
        return cleaned

    tokens = encoded_tokens(cleaned)
    middle_marker = " ... [middle excerpt] ... "
    ending_marker = " ... [ending excerpt] ... "
    marker_tokens = len(encoded_tokens(middle_marker + ending_marker))
    raw_budget = max(3, math.floor(max_tokens / TOKEN_SAFETY_FACTOR) - marker_tokens)
    window_sizes = [raw_budget // 3, raw_budget // 3, raw_budget // 3]
    for index in range(raw_budget % 3):
        window_sizes[index] += 1

    def representative_excerpt() -> str:
        head_size, middle_size, tail_size = window_sizes
        middle_start = max(head_size, (len(tokens) - middle_size) // 2)
        tail_start = max(middle_start + middle_size, len(tokens) - tail_size)
        return clean_text(
            decode_tokens(tokens[:head_size])
            + middle_marker
            + decode_tokens(tokens[middle_start : middle_start + middle_size])
            + ending_marker
            + decode_tokens(tokens[tail_start:])
        )

    excerpt = representative_excerpt()
    while token_count(excerpt) > max_tokens and max(window_sizes) > 1:
        largest_window = max(range(3), key=window_sizes.__getitem__)
        window_sizes[largest_window] -= 1
        excerpt = representative_excerpt()
    return excerpt


def image_context_text(item: dict) -> str:
    metadata = item.get("metadata") or {}
    nearby_text = "\n".join(
        value
        for value in [
            metadata.get("nearby_text_before"),
            metadata.get("nearby_text_after"),
        ]
        if value
    )
    return clip_text(
        nearby_text or metadata.get("nearby_table_markdown"),
        max_tokens=560,
    )


def primary_bbox(item: dict) -> dict | None:
    boxes = item.get("bbox")
    if isinstance(boxes, dict):
        boxes = [boxes]
    if not isinstance(boxes, list):
        return None
    for box in boxes:
        if not isinstance(box, dict):
            continue
        if all(box.get(key) is not None for key in ["x", "y", "w", "h"]):
            return {key: float(box[key]) for key in ["x", "y", "w", "h"]}
    return None


def bbox_center(box: dict) -> tuple[float, float]:
    return box["x"] + box["w"] / 2, box["y"] + box["h"] / 2


def lexical_terms(text: str | None) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"\b[\w-]{3,}\b", text or "", flags=re.UNICODE)
    }


def image_child_score(image: dict, child: dict) -> tuple[float, dict]:
    image_page = image.get("page_number")
    child_pages = child.get("page_numbers") or []
    same_page = image_page is not None and image_page in child_pages
    if same_page:
        page_penalty = 0.0
    elif image_page is None or not child_pages:
        page_penalty = 80.0
    else:
        page_penalty = 160.0 + min(
            abs(image_page - page_number) for page_number in child_pages
        ) * 20

    positions = child["metadata"].get("element_positions") or []
    image_box = primary_bbox(image)
    matching_boxes = [
        position["bbox"]
        for position in positions
        if position.get("bbox")
        and (
            image_page is None
            or position.get("page_number") is None
            or position.get("page_number") == image_page
        )
    ]
    if image_box and matching_boxes:
        image_center = bbox_center(image_box)
        bbox_distance = min(
            math.dist(image_center, bbox_center(box))
            for box in matching_boxes
        )
        bbox_penalty = min(bbox_distance / 40, 50)
    else:
        bbox_distance = None
        bbox_penalty = 20.0

    sequences = child["metadata"].get("element_sequences") or [
        position["sequence"]
        for position in child["metadata"].get("element_positions", [])
        if position.get("sequence") is not None
    ]
    if not sequences:
        sequences = [image["sequence"]]
    sequence_distance = min(abs(image["sequence"] - sequence) for sequence in sequences)
    sequence_penalty = min(sequence_distance * 3, 60)

    image_terms = lexical_terms(image_context_text(image))
    child_terms = lexical_terms(
        "\n".join(
            value
            for value in [child.get("text_content"), child.get("table_markdown")]
            if value
        )
    )
    overlap_ratio = (
        len(image_terms & child_terms) / max(1, min(len(image_terms), len(child_terms)))
        if image_terms and child_terms
        else 0.0
    )
    lexical_bonus = min(overlap_ratio * 50, 50)
    table_bonus = (
        12.0
        if (image.get("metadata") or {}).get("nearby_table_markdown")
        and "table" in child["modalities"]
        else 0.0
    )
    score = page_penalty + bbox_penalty + sequence_penalty - lexical_bonus - table_bonus
    return score, {
        "same_page": same_page,
        "page_penalty": round(page_penalty, 4),
        "bbox_distance": round(bbox_distance, 4) if bbox_distance is not None else None,
        "bbox_penalty": round(bbox_penalty, 4),
        "sequence_distance": round(sequence_distance, 4),
        "sequence_penalty": round(sequence_penalty, 4),
        "lexical_overlap": round(overlap_ratio, 4),
        "lexical_bonus": round(lexical_bonus, 4),
        "table_bonus": round(table_bonus, 4),
        "total_score": round(score, 4),
    }


def attach_image_to_child(
    target: dict,
    image: dict,
    assignment_method: str,
    assignment_score: float | None = None,
    assignment_details: dict | None = None,
) -> None:
    if image["image_path"] not in target["image_paths"]:
        target["image_paths"].append(image["image_path"])
    if image["element_id"] not in target["element_ids"]:
        target["element_ids"].append(image["element_id"])
    target["page_numbers"] = sorted(
        set(target["page_numbers"])
        | ({image["page_number"]} if image.get("page_number") is not None else set())
    )
    if "image" not in target["modalities"]:
        target["modalities"].append("image")
    target["metadata"]["image_assignment_method"] = assignment_method
    target["metadata"].setdefault("image_context", []).append(
        {
            "image_path": image["image_path"],
            "nearby_text_before": clip_text(
                (image.get("metadata") or {}).get("nearby_text_before"),
                max_tokens=250,
            ),
            "nearby_text_after": clip_text(
                (image.get("metadata") or {}).get("nearby_text_after"),
                max_tokens=250,
            ),
            "selection_reason": (image.get("metadata") or {}).get(
                "selection_reason"
            ),
            "assignment_method": assignment_method,
            "assignment_score": round(assignment_score, 4)
            if assignment_score is not None
            else None,
            "assignment_details": assignment_details,
        }
    )
    target["retrieval_text"] = retrieval_text(target)
    target["token_count"] = token_count(target["retrieval_text"])


def is_tiny_layout_chart(image: dict) -> bool:
    if (image.get("metadata") or {}).get("selection_reason") != "layout chart crop":
        return False
    box = primary_bbox(image)
    return bool(box and (box["w"] < 80 or box["h"] < 80))


def escape_markdown_cell(value: str) -> str:
    return clean_text(value).replace("|", r"\|")


def rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    width = max(len(headers), *(len(row) for row in rows), 1)
    normalized_headers = (
        headers + [f"Column {index}" for index in range(len(headers) + 1, width + 1)]
    )[:width]
    lines = [
        "| " + " | ".join(escape_markdown_cell(value) for value in normalized_headers) + " |",
        "| " + " | ".join("---" for _ in normalized_headers) + " |",
    ]
    for row in rows:
        normalized_row = (row + [""] * width)[:width]
        lines.append(
            "| " + " | ".join(escape_markdown_cell(value) for value in normalized_row) + " |"
        )
    return "\n".join(lines)


def rows_to_html(headers: list[str], rows: list[list[str]]) -> str:
    width = max(len(headers), *(len(row) for row in rows), 1)
    normalized_headers = (
        headers + [f"Column {index}" for index in range(len(headers) + 1, width + 1)]
    )[:width]
    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in normalized_headers)
    body_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(value)}</td>"
            for value in (row + [""] * width)[:width]
        )
        + "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr>"
        + header_html
        + "</tr></thead><tbody>"
        + body_html
        + "</tbody></table>"
    )


def parse_markdown_table(markdown: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if (
        len(lines) < 3
        or "|" not in lines[0]
        or not re.fullmatch(r"[\s|:-]+", lines[1])
    ):
        return None

    def cells(line: str) -> list[str]:
        return [
            clean_text(value.replace(r"\|", "|"))
            for value in re.split(r"(?<!\\)\|", line.strip("|"))
        ]

    return cells(lines[0]), [cells(line) for line in lines[2:]]


def parse_html_table(table_html: str | None) -> tuple[list[str], list[list[str]]] | None:
    if not table_html:
        return None
    soup = BeautifulSoup(table_html, "html.parser")
    rows = []
    header = []
    for row_index, row in enumerate(soup.find_all("tr")):
        cells = row.find_all(["th", "td"])
        values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
        if not values:
            continue
        if row.find("th") and not header:
            header = values
        else:
            rows.append(values)
    if not rows:
        return None
    width = max(len(row) for row in rows)
    if not header:
        header = [f"Column {index}" for index in range(1, width + 1)]
    return header, rows


def structured_row_text(headers: list[str], row: list[str], row_number: int) -> str:
    width = max(len(headers), len(row))
    names = (headers + [f"Column {index}" for index in range(len(headers) + 1, width + 1)])[:width]
    values = (row + [""] * width)[:width]
    fields = [
        f"{name} = {value}"
        for name, value in zip(names, values)
        if clean_text(value)
    ]
    return f"Row {row_number}: " + "; ".join(fields)


def structured_rows_to_parts(
    headers: list[str],
    rows: list[list[str]],
) -> list[dict]:
    records = [
        structured_row_text(headers, row, row_number)
        for row_number, row in enumerate(rows, start=1)
    ]
    parts = []
    current = []
    start_row = 1
    for row_number, record in enumerate(records, start=1):
        pieces = split_oversized_text(record, TABLE_SPLIT_MAX_TOKENS - 60)
        for piece in pieces:
            candidate = current + [piece]
            if current and token_count("\n".join(candidate)) > TABLE_SPLIT_MAX_TOKENS:
                rows_for_part = [[value] for value in current]
                parts.append(
                    {
                        "markdown": rows_to_markdown(["Structured table row"], rows_for_part),
                        "html": rows_to_html(["Structured table row"], rows_for_part),
                        "row_start": start_row,
                        "row_end": row_number - 1,
                        "strategy": "structured_rows",
                        "headers": headers,
                    }
                )
                current = []
                start_row = row_number
            current.append(piece)
    if current:
        rows_for_part = [[value] for value in current]
        parts.append(
            {
                "markdown": rows_to_markdown(["Structured table row"], rows_for_part),
                "html": rows_to_html(["Structured table row"], rows_for_part),
                "row_start": start_row,
                "row_end": len(records),
                "strategy": "structured_rows",
                "headers": headers,
            }
        )
    return parts


def split_parsed_table(
    headers: list[str],
    rows: list[list[str]],
    strategy: str,
) -> list[dict]:
    parts = []
    current = []
    start_row = 1
    for row_number, row in enumerate(rows, start=1):
        candidate = current + [row]
        candidate_markdown = rows_to_markdown(headers, candidate)
        if current and token_count(candidate_markdown) > TABLE_SPLIT_MAX_TOKENS:
            markdown = rows_to_markdown(headers, current)
            parts.append(
                {
                    "markdown": markdown,
                    "html": rows_to_html(headers, current),
                    "row_start": start_row,
                    "row_end": row_number - 1,
                    "strategy": strategy,
                    "headers": headers,
                }
            )
            current = []
            start_row = row_number
        single_markdown = rows_to_markdown(headers, [row])
        if token_count(single_markdown) > TABLE_SPLIT_MAX_TOKENS:
            if current:
                markdown = rows_to_markdown(headers, current)
                parts.append(
                    {
                        "markdown": markdown,
                        "html": rows_to_html(headers, current),
                        "row_start": start_row,
                        "row_end": row_number - 1,
                        "strategy": strategy,
                        "headers": headers,
                    }
                )
                current = []
            parts.extend(structured_rows_to_parts(headers, [row]))
            start_row = row_number + 1
        else:
            current.append(row)
    if current:
        markdown = rows_to_markdown(headers, current)
        parts.append(
            {
                "markdown": markdown,
                "html": rows_to_html(headers, current),
                "row_start": start_row,
                "row_end": len(rows),
                "strategy": strategy,
                "headers": headers,
            }
        )
    return parts


def markdown_table_to_html(markdown: str) -> str:
    parsed = parse_markdown_table(markdown)
    if not parsed:
        return ""
    return rows_to_html(*parsed)


def one_column_table(text: str) -> tuple[str, str]:
    value = text.replace("|", r"\|")
    markdown = f"| Table segment |\n| --- |\n| {value} |"
    return markdown, markdown_table_to_html(markdown)


def split_oversized_table(item: dict) -> list[dict]:
    markdown = (item.get("table_markdown") or item.get("text") or "").strip()
    if not markdown or token_count(markdown) <= TABLE_SPLIT_MAX_TOKENS:
        return [item]

    parsed = parse_markdown_table(markdown)
    parts = split_parsed_table(*parsed, "markdown_rows") if parsed else []
    if not parts or any(
        token_count(part["markdown"]) > TABLE_SPLIT_MAX_TOKENS + 20
        for part in parts
    ):
        parsed = parse_html_table(item.get("table_html"))
        parts = split_parsed_table(*parsed, "html_rows") if parsed else []
    if not parts or any(
        token_count(part["markdown"]) > TABLE_SPLIT_MAX_TOKENS + 20
        for part in parts
    ):
        table_text = re.sub(r"<br\s*/?>", " ", markdown, flags=re.IGNORECASE)
        table_text = re.sub(r"\s*\|\s*", " ", table_text)
        table_text = re.sub(r"[-_=]{10,}", " ", table_text)
        parts = [
            {
                "markdown": part_markdown,
                "html": part_html,
                "row_start": None,
                "row_end": None,
                "strategy": "unstructured_last_resort",
                "headers": [],
            }
            for piece in split_oversized_text(
                table_text,
                TABLE_SPLIT_MAX_TOKENS - 40,
            )
            for part_markdown, part_html in [one_column_table(piece)]
        ]

    split_items = []
    for part_index, part in enumerate(parts, start=1):
        split_item = dict(item)
        split_item["element_id"] = stable_id(
            item["element_id"],
            "table_split",
            str(part_index),
        )
        split_item["sequence"] = item["sequence"] + (
            part_index / (len(parts) + 1)
        )
        split_item["text"] = None
        split_item["table_markdown"] = part["markdown"]
        split_item["table_html"] = part["html"]
        split_item["metadata"] = dict(item.get("metadata") or {})
        split_item["metadata"].update(
            {
                "original_element_id": item["element_id"],
                "table_split_part": part_index,
                "table_split_count": len(parts),
                "original_table_preserved_in_normalized_elements": True,
                "table_split_strategy": part["strategy"],
                "table_headers": part["headers"],
                "table_row_start": part["row_start"],
                "table_row_end": part["row_end"],
            }
        )
        split_items.append(split_item)
    return split_items


def expand_oversized_elements(elements: list[dict]) -> list[dict]:
    expanded = []
    for item in elements:
        if item["element_type"] == "table":
            expanded.extend(split_oversized_table(item))
            continue
        if item["element_type"] in {"heading", "image"}:
            expanded.append(item)
            continue

        pieces = split_oversized_text(item.get("text"))
        if len(pieces) <= 1:
            expanded.append(item)
            continue

        for part_index, piece in enumerate(pieces, start=1):
            split_item = dict(item)
            split_item["element_id"] = stable_id(
                item["element_id"],
                "text_split",
                str(part_index),
            )
            split_item["sequence"] = item["sequence"] + (
                part_index / (len(pieces) + 1)
            )
            split_item["text"] = piece
            split_item["metadata"] = dict(item.get("metadata") or {})
            split_item["metadata"].update(
                {
                    "original_element_id": item["element_id"],
                    "text_split_part": part_index,
                    "text_split_count": len(pieces),
                }
            )
            expanded.append(split_item)
    return expanded


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False, default=str))
            output.write("\n")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def element_content(item: dict) -> str:
    if item["element_type"] == "image":
        return image_context_text(item)
    if item["element_type"] == "table":
        return item.get("table_markdown") or item.get("text") or ""
    return item.get("text") or ""


def retrieval_text(chunk: dict) -> str:
    lines = [
        f"Domain: {chunk['domain']}",
        f"Source: {chunk['source_name']}",
    ]
    if chunk.get("section_title"):
        lines.append(f"Section: {chunk['section_title']}")
    if chunk.get("text_content"):
        lines.extend(["", chunk["text_content"]])
    if chunk.get("table_markdown") and not chunk.get("metadata", {}).get(
        "exclude_table_from_retrieval"
    ):
        lines.extend(["", chunk["table_markdown"]])
    return "\n".join(lines)


def dailymed_clinical_table_signal(source_name: str, text: str | None) -> str | None:
    if source_name != "dailymed_ozempic_prescribing_label":
        return None
    cleaned = clean_text(text)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    clinical_terms = [
        "ozempic",
        "semaglutide",
        "placebo",
        "hba",
        "plasma glucose",
        "sitagliptin",
        "baseline",
        "end-of-treatment",
        "randomization",
        "week 30",
        "week 56",
    ]
    if not any(term in lowered for term in clinical_terms):
        return None
    if re.search(r"\b(table|figure)\s+\d+\b", cleaned, flags=re.IGNORECASE):
        return "dailymed_clinical_table_or_figure_text"
    if re.search(r"\bn\s*=\s*\d+", lowered) and lowered.count("ozempic") >= 2:
        return "dailymed_treatment_group_count_text"
    return None


def rescue_dailymed_table_channel(
    source_name: str,
    text: str | None,
) -> tuple[str, str, str] | None:
    signal = dailymed_clinical_table_signal(source_name, text)
    if not signal:
        return None
    excerpt = clip_text(text, max_tokens=140)
    markdown, table_html = one_column_table(excerpt)
    return markdown, table_html, signal


def chunk_fingerprint(chunk: dict) -> str:
    identity = "\n".join(
        [
            chunk["retrieval_text"].lower(),
            "Images:",
            *sorted(chunk.get("image_paths") or []),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def group_parent_parts(elements: list[dict]) -> list[list[dict]]:
    parts = []
    current = []
    current_tokens = 0

    for item in elements:
        if item["element_type"] == "heading":
            continue
        item_tokens = token_count(element_content(item))
        if current and current_tokens + item_tokens > PARENT_MAX_TOKENS:
            parts.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += item_tokens

    if current:
        parts.append(current)
    return parts


def base_child(
    parent: dict,
    child_index: int,
    elements: list[dict],
    text_content: str | None,
    table_markdown: str | None = None,
    table_html: str | None = None,
) -> dict:
    page_numbers = sorted(
        {item["page_number"] for item in elements if item.get("page_number") is not None}
    )
    source_urls = sorted(
        {item["source_url"] for item in elements if item.get("source_url")}
    )
    source_rows = sorted(
        {
            row
            for item in elements
            for row in (item.get("source_row_numbers") or [])
        }
    )
    cleaned_text = clean_text(text_content)
    table_rescue_strategy = None
    if not table_markdown and not table_html:
        rescued = rescue_dailymed_table_channel(parent["source_name"], cleaned_text)
        if rescued:
            table_markdown, table_html, table_rescue_strategy = rescued

    modalities = []
    if cleaned_text:
        modalities.append("text")
    if table_markdown or table_html:
        modalities.append("table")

    table_metadata = next(
        (
            item.get("metadata") or {}
            for item in elements
            if item["element_type"] == "table"
        ),
        {},
    )
    chunk = {
        "chunk_id": stable_id(parent["parent_id"], str(child_index)),
        "retrieval_type": "child_chunk",
        "parent_id": parent["parent_id"],
        "document_id": parent["document_id"],
        "domain": parent["domain"],
        "source_type": parent["source_type"],
        "source_name": parent["source_name"],
        "section_title": parent["section_title"],
        "section_path": parent["section_path"],
        "page_numbers": page_numbers,
        "source_urls": source_urls,
        "source_row_numbers": source_rows,
        "modalities": modalities,
        "text_content": cleaned_text,
        "table_markdown": table_markdown,
        "table_html": table_html,
        "image_paths": [],
        "element_ids": [item["element_id"] for item in elements],
        "metadata": {
            "part_number": parent["part_number"],
            "table_kept_intact": bool(table_markdown or table_html)
            and not table_metadata.get("table_split_count")
            and table_rescue_strategy is None,
            "source_table_element_id": table_metadata.get("original_element_id"),
            "table_split_part": table_metadata.get("table_split_part"),
            "table_split_count": table_metadata.get("table_split_count"),
            "table_split_strategy": table_metadata.get("table_split_strategy"),
            "table_headers": table_metadata.get("table_headers"),
            "table_row_start": table_metadata.get("table_row_start"),
            "table_row_end": table_metadata.get("table_row_end"),
            "table_rescue_strategy": table_rescue_strategy,
            "element_positions": [
                {
                    "element_id": item["element_id"],
                    "sequence": item["sequence"],
                    "page_number": item.get("page_number"),
                    "bbox": primary_bbox(item),
                }
                for item in elements
            ],
        },
    }
    chunk["retrieval_text"] = retrieval_text(chunk)
    chunk["token_count"] = token_count(chunk["retrieval_text"])
    return chunk


def attach_images(children: list[dict], images: list[dict], parent: dict) -> list[dict]:
    for image in images:
        if is_tiny_layout_chart(image):
            continue
        image_sequence = image["sequence"]
        candidates = [
            child
            for child in children
            if len(child["image_paths"]) < MAX_IMAGES_PER_CHUNK
        ]
        if candidates:
            scored_candidates = [
                (image_child_score(image, child), child)
                for child in candidates
            ]
            (best_score, score_details), target = min(
                scored_candidates,
                key=lambda candidate: candidate[0][0],
            )
            if not score_details["same_page"] and image_context_text(image):
                target = base_child(
                    parent,
                    len(children) + 1,
                    [image],
                    image_context_text(image),
                )
                target["metadata"]["element_sequences"] = [image_sequence]
                target["metadata"]["image_assignment_method"] = "contextual_fallback"
                children.append(target)
            elif not score_details["same_page"]:
                continue
            else:
                target["metadata"]["image_assignment_method"] = "weighted_geometry"
        else:
            context = image_context_text(image)
            if not context:
                continue
            target = base_child(
                parent,
                len(children) + 1,
                [image],
                context,
            )
            target["metadata"]["element_sequences"] = [image_sequence]
            target["metadata"]["image_assignment_method"] = "contextual_fallback"
            children.append(target)
            best_score = None
            score_details = None

        attach_image_to_child(
            target,
            image,
            target["metadata"].get("image_assignment_method", "weighted_geometry"),
            best_score,
            score_details,
        )
    return children


def build_children(parent: dict, elements: list[dict]) -> list[dict]:
    children = []
    text_buffer = []
    text_tokens = 0
    images = []

    def flush_text() -> None:
        nonlocal text_buffer, text_tokens
        if not text_buffer:
            return
        content = "\n\n".join(item.get("text") or "" for item in text_buffer)
        child = base_child(parent, len(children) + 1, text_buffer, content)
        child["metadata"]["element_sequences"] = [
            item["sequence"] for item in text_buffer
        ]
        children.append(child)
        text_buffer = []
        text_tokens = 0

    for item in elements:
        kind = item["element_type"]
        if kind == "heading":
            continue
        if kind == "image":
            images.append(item)
            continue
        if kind == "table":
            flush_text()
            child = base_child(
                parent,
                len(children) + 1,
                [item],
                None,
                item.get("table_markdown"),
                item.get("table_html"),
            )
            child["metadata"]["element_sequences"] = [item["sequence"]]
            children.append(child)
            continue

        item_tokens = token_count(item.get("text"))
        if text_buffer and text_tokens + item_tokens > CHILD_CONTENT_MAX_TOKENS:
            flush_text()
        text_buffer.append(item)
        text_tokens += item_tokens
        if text_tokens >= CHILD_TARGET_TOKENS:
            flush_text()

    flush_text()
    children = attach_images(children, images, parent)
    for child in children:
        child["metadata"].pop("element_sequences", None)
    return children


def recover_unattached_table_images(
    images: list[dict],
    image_parent_ids: dict[str, str],
    all_parents: dict[str, dict],
    parents: list[dict],
    children: list[dict],
) -> tuple[list[dict], list[dict]]:
    attached_paths = {
        path
        for child in children
        for path in child["image_paths"]
    }
    parent_ids = {parent["parent_id"] for parent in parents}

    for image in images:
        if image["image_path"] in attached_paths:
            continue
        if (image.get("metadata") or {}).get("selection_reason") != (
            "layout table or form crop"
        ):
            continue

        page_number = image.get("page_number")
        candidates = [
            child
            for child in children
            if "table" in child["modalities"]
            and page_number is not None
            and page_number in child["page_numbers"]
            and len(child["image_paths"]) < MAX_IMAGES_PER_CHUNK
        ]
        parent_id = image_parent_ids.get(image["element_id"])
        parent = all_parents.get(parent_id)
        if not parent:
            continue
        page_label = (
            f"page {page_number}"
            if page_number is not None
            else "an unspecified page"
        )
        context = (
            f"Visual table crop {Path(image['image_path']).name} from section "
            f"{parent['section_title']} on {page_label}. "
            "Use the attached image as the table evidence."
        )
        child_index = sum(
            child["parent_id"] == parent["parent_id"] for child in children
        ) + 1
        target = base_child(parent, child_index, [image], context)

        if candidates:
            scored_candidates = [
                (image_child_score(image, child), child)
                for child in candidates
            ]
            (score, details), matched_table = min(
                scored_candidates,
                key=lambda candidate: candidate[0][0],
            )
            target["metadata"]["matched_table_chunk_id"] = matched_table["chunk_id"]
            attach_image_to_child(
                target,
                image,
                "document_same_page_table_reference",
                score,
                details,
            )
        else:
            attach_image_to_child(
                target,
                image,
                "visual_table_fallback",
            )
        children.append(target)
        attached_paths.add(image["image_path"])
        if parent["parent_id"] not in parent_ids:
            parents.append(parent)
            parent_ids.add(parent["parent_id"])

    child_counts = Counter(child["parent_id"] for child in children)
    for parent in parents:
        parent["child_count"] = child_counts[parent["parent_id"]]
    return parents, children


def deduplicate_children(
    parents: list[dict],
    children: list[dict],
) -> tuple[list[dict], list[dict]]:
    unique = {}
    for child in children:
        fingerprint = chunk_fingerprint(child)
        existing = unique.get(fingerprint)
        if existing is None:
            child["metadata"]["duplicate_occurrences"] = 1
            child["metadata"]["alternate_parent_ids"] = []
            unique[fingerprint] = child
            continue

        existing["metadata"]["duplicate_occurrences"] += 1
        if child["parent_id"] != existing["parent_id"]:
            existing["metadata"]["alternate_parent_ids"] = list(
                dict.fromkeys(
                    existing["metadata"]["alternate_parent_ids"]
                    + [child["parent_id"]]
                )
            )
        for field in [
            "page_numbers",
            "source_urls",
            "source_row_numbers",
            "element_ids",
        ]:
            existing[field] = sorted(set(existing[field]) | set(child[field]))
        merged_images = list(dict.fromkeys(existing["image_paths"] + child["image_paths"]))
        existing["image_paths"] = merged_images[:MAX_IMAGES_PER_CHUNK]
        existing["metadata"]["omitted_duplicate_image_count"] = max(
            0,
            len(merged_images) - MAX_IMAGES_PER_CHUNK,
        )
        existing["metadata"]["image_context"] = (
            existing["metadata"].get("image_context", [])
            + child["metadata"].get("image_context", [])
        )[:MAX_IMAGES_PER_CHUNK]
        existing["modalities"] = list(
            dict.fromkeys(existing["modalities"] + child["modalities"])
        )

    children = list(unique.values())
    child_counts = Counter(child["parent_id"] for child in children)
    parents = [parent for parent in parents if child_counts[parent["parent_id"]]]
    for parent in parents:
        parent["child_count"] = child_counts[parent["parent_id"]]
    return parents, children


def build_document_chunks(elements: list[dict]) -> tuple[list[dict], list[dict]]:
    elements = expand_oversized_elements(elements)
    first = elements[0]
    grouped = defaultdict(list)
    for item in elements:
        key = (
            item.get("source_url") or "",
            tuple(item.get("section_path") or [item.get("section_title") or "Unsectioned"]),
        )
        grouped[key].append(item)

    parents, children = [], []
    all_parents = {}
    image_parent_ids = {}
    for (_, section_path), section_elements in grouped.items():
        section_elements.sort(key=lambda item: item["sequence"])
        parts = group_parent_parts(section_elements)
        for part_number, part in enumerate(parts, start=1):
            section_title = (
                part[0].get("section_title")
                or (section_path[-1] if section_path else "Unsectioned")
            )
            parent_id = stable_id(
                first["document_id"],
                "|".join(section_path),
                str(part_number),
                part[0].get("source_url") or "",
            )
            parent_text = "\n\n".join(
                element_content(item)
                for item in part
                if element_content(item)
            )
            parent = {
                "parent_id": parent_id,
                "document_id": first["document_id"],
                "domain": first["domain"],
                "source_type": first["source_type"],
                "source_name": first["source_name"],
                "section_title": section_title,
                "section_path": list(section_path),
                "part_number": part_number,
                "page_numbers": sorted(
                    {
                        item["page_number"]
                        for item in part
                        if item.get("page_number") is not None
                    }
                ),
                "source_urls": sorted(
                    {item["source_url"] for item in part if item.get("source_url")}
                ),
                "parent_text": parent_text,
                "token_count": token_count(parent_text),
                "element_ids": [item["element_id"] for item in part],
                "metadata": {},
            }
            all_parents[parent_id] = parent
            for item in part:
                if item["element_type"] == "image":
                    image_parent_ids[item["element_id"]] = parent_id
            parent_children = build_children(parent, part)
            if not parent_children:
                continue
            parent["child_count"] = len(parent_children)
            parents.append(parent)
            children.extend(parent_children)
    parents, children = recover_unattached_table_images(
        [item for item in elements if item["element_type"] == "image"],
        image_parent_ids,
        all_parents,
        parents,
        children,
    )
    return deduplicate_children(parents, children)


def build_csv_chunks(elements: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for item in elements:
        grouped[(item["metadata"]["profile_id"], item["metadata"]["facility_id"])].append(item)

    parents, children = [], []
    for (profile_id, facility_id), items in grouped.items():
        first = items[0]
        hospital_name = first["metadata"].get("hospital_name")
        categories = [item["section_title"] for item in items]
        parent_text = "\n".join(
            [
                f"Hospital: {hospital_name}",
                f"Facility ID: {facility_id}",
                f"State: {first['metadata'].get('state')}",
                "Available categories: " + ", ".join(categories),
            ]
        )
        parent = {
            "parent_id": profile_id,
            "document_id": first["document_id"],
            "domain": first["domain"],
            "source_type": "csv",
            "source_name": first["source_name"],
            "section_title": hospital_name,
            "section_path": [hospital_name],
            "part_number": 1,
            "page_numbers": [],
            "source_urls": [],
            "parent_text": parent_text,
            "token_count": token_count(parent_text),
            "element_ids": [item["element_id"] for item in items],
            "child_count": len(items),
            "metadata": {"facility_id": facility_id},
        }
        parents.append(parent)
        for item in items:
            chunk = {
                "chunk_id": item["element_id"],
                "retrieval_type": "hospital_category_doc",
                "parent_id": profile_id,
                "document_id": item["document_id"],
                "domain": item["domain"],
                "source_type": "csv",
                "source_name": item["source_name"],
                "section_title": item["section_title"],
                "section_path": item["section_path"],
                "page_numbers": [],
                "source_urls": [],
                "source_row_numbers": item["source_row_numbers"],
                "modalities": ["text", "table"],
                "text_content": item["text"],
                "table_markdown": item["table_markdown"],
                "table_html": None,
                "image_paths": [],
                "element_ids": [item["element_id"]],
                "metadata": {
                    **item["metadata"],
                    "exclude_table_from_retrieval": True,
                },
            }
            chunk["retrieval_text"] = retrieval_text(chunk)
            chunk["token_count"] = token_count(chunk["retrieval_text"])
            children.append(chunk)
    return parents, children


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def quality_report(parents: list[dict], children: list[dict]) -> dict:
    child_tokens = [child["token_count"] for child in children]
    parent_tokens = [parent["token_count"] for parent in parents]
    text_hashes = Counter(
        chunk_fingerprint(child)
        for child in children
    )
    missing_images = [
        path
        for child in children
        for path in child["image_paths"]
        if not (PROJECT_ROOT / path).exists()
    ]
    child_parent_ids = Counter(child["parent_id"] for child in children)
    image_only = [
        child["chunk_id"]
        for child in children
        if child["image_paths"]
        and not clean_text(child.get("text_content"))
        and not child.get("table_markdown")
    ]
    image_contexts = [
        context
        for child in children
        for context in child["metadata"].get("image_context", [])
    ]
    return {
        "tokenizer": TOKENIZER_NAME,
        "token_safety_factor": TOKEN_SAFETY_FACTOR,
        "parent_count": len(parents),
        "child_count": len(children),
        "chunk_modalities": dict(
            Counter("+".join(child["modalities"]) for child in children)
        ),
        "child_token_stats": {
            "min": min(child_tokens, default=0),
            "average": round(statistics.mean(child_tokens), 2) if child_tokens else 0,
            "p95": percentile(child_tokens, 0.95),
            "max": max(child_tokens, default=0),
        },
        "parent_token_stats": {
            "min": min(parent_tokens, default=0),
            "average": round(statistics.mean(parent_tokens), 2) if parent_tokens else 0,
            "p95": percentile(parent_tokens, 0.95),
            "max": max(parent_tokens, default=0),
        },
        "empty_chunks": [
            child["chunk_id"]
            for child in children
            if not child["retrieval_text"].strip()
        ],
        "image_only_chunks": image_only,
        "missing_image_paths": missing_images,
        "duplicate_chunk_count": sum(count - 1 for count in text_hashes.values() if count > 1),
        "parents_without_children": [
            parent["parent_id"]
            for parent in parents
            if not child_parent_ids[parent["parent_id"]]
        ],
        "oversized_non_table_chunks": [
            child["chunk_id"]
            for child in children
            if child["token_count"] > CHILD_MAX_TOKENS
            and "table" not in child["modalities"]
        ],
        "table_chunks": sum("table" in child["modalities"] for child in children),
        "tables_split": len(
            {
                child["metadata"]["source_table_element_id"]
                for child in children
                if child["metadata"].get("table_split_count")
            }
        ),
        "table_split_strategies": dict(
            Counter(
                child["metadata"]["table_split_strategy"]
                for child in children
                if child["metadata"].get("table_split_strategy")
            )
        ),
        "image_assignment_methods": dict(
            Counter(
                context.get("assignment_method") or "unknown"
                for context in image_contexts
            )
        ),
        "weighted_images_assigned_same_page": sum(
            context.get("assignment_method") == "weighted_geometry"
            and (context.get("assignment_details") or {}).get("same_page")
            for context in image_contexts
        ),
        "weighted_images_assigned_cross_page": sum(
            context.get("assignment_method") == "weighted_geometry"
            and not (context.get("assignment_details") or {}).get("same_page")
            for context in image_contexts
        ),
        "oversized_table_chunks": [
            child["chunk_id"]
            for child in children
            if child["token_count"] > CHILD_MAX_TOKENS
            and "table" in child["modalities"]
        ],
        "chunks_over_image_limit": [
            child["chunk_id"]
            for child in children
            if len(child["image_paths"]) > MAX_IMAGES_PER_CHUNK
        ],
    }


def create_chunks_for_file(elements_path: Path) -> dict:
    elements = read_jsonl(elements_path)
    if not elements:
        raise RuntimeError(f"No normalized elements in {elements_path}")
    first = elements[0]
    if first["source_type"] == "csv":
        parents, children = build_csv_chunks(elements)
    else:
        parents, children = build_document_chunks(elements)

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "chunks"
        / first["domain"]
        / first["source_type"]
    )
    source_name = first["source_name"]
    parents_path = output_dir / f"{source_name}.parents.jsonl"
    children_path = output_dir / f"{source_name}.children.jsonl"
    quality_path = output_dir / f"{source_name}.quality.json"
    write_jsonl(parents_path, parents)
    write_jsonl(children_path, children)
    report = quality_report(parents, children)
    report.update(
        {
            "document_id": first["document_id"],
            "domain": first["domain"],
            "source_type": first["source_type"],
            "source_name": source_name,
            "outputs": {
                "parents_path": relative_path(parents_path),
                "children_path": relative_path(children_path),
            },
        }
    )
    write_json(quality_path, report)
    return report


def create_all_chunks() -> list[dict]:
    reports = [
        create_chunks_for_file(path)
        for path in sorted(
            (PROJECT_ROOT / "data" / "normalized").glob("*/*/*.elements.jsonl")
        )
    ]
    totals = {
        "documents": len(reports),
        "parents": sum(report["parent_count"] for report in reports),
        "children": sum(report["child_count"] for report in reports),
        "empty_chunks": sum(len(report["empty_chunks"]) for report in reports),
        "image_only_chunks": sum(
            len(report["image_only_chunks"]) for report in reports
        ),
        "missing_image_paths": sum(
            len(report["missing_image_paths"]) for report in reports
        ),
        "duplicate_chunk_count": sum(
            report["duplicate_chunk_count"] for report in reports
        ),
        "oversized_non_table_chunks": sum(
            len(report["oversized_non_table_chunks"]) for report in reports
        ),
        "oversized_table_chunks": sum(
            len(report["oversized_table_chunks"]) for report in reports
        ),
        "tables_split": sum(report["tables_split"] for report in reports),
        "parents_without_children": sum(
            len(report["parents_without_children"]) for report in reports
        ),
        "chunks_over_image_limit": sum(
            len(report["chunks_over_image_limit"]) for report in reports
        ),
        "table_split_strategies": dict(
            sum(
                (
                    Counter(report["table_split_strategies"])
                    for report in reports
                ),
                Counter(),
            )
        ),
        "image_assignment_methods": dict(
            sum(
                (
                    Counter(report["image_assignment_methods"])
                    for report in reports
                ),
                Counter(),
            )
        ),
        "weighted_images_assigned_same_page": sum(
            report["weighted_images_assigned_same_page"] for report in reports
        ),
        "weighted_images_assigned_cross_page": sum(
            report["weighted_images_assigned_cross_page"] for report in reports
        ),
    }
    summary = {"totals": totals, "documents": reports}
    write_json(PROJECT_ROOT / "data" / "chunks" / "quality_summary.json", summary)
    return reports
