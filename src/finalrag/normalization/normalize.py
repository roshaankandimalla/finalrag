import hashlib
import json
import re
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError

from finalrag.discovery.file_discovery import create_document_id


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAGE_PATTERNS = [
    re.compile(r"_p(\d+)_", re.IGNORECASE),
    re.compile(r"page_(\d+)_", re.IGNORECASE),
]
NOISY_IMAGE_TERMS = {
    "logo", "icon", "favicon", "sprite", "avatar", "button", "social",
    "toplogo", "branding",
}
NOISY_HTML_TEXT = {
    "skip to main content",
    "home news dailymed announcements",
    "report adverse events | recalls",
}


def stable_id(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, ":".join(parts)))


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_jsonl(path: Path, values) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False, default=str))
            output.write("\n")
            count += 1
    return count


def page_from_filename(filename: str) -> int | None:
    for pattern in PAGE_PATTERNS:
        match = pattern.search(filename)
        if match:
            return int(match.group(1))
    return None


def bbox_y(item: dict) -> float:
    boxes = item.get("bbox") or []
    boxes = [boxes] if isinstance(boxes, dict) else boxes
    values = [box.get("y") for box in boxes if box.get("y") is not None]
    return min(values) if values else float("inf")


def image_metrics(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        grayscale = image.convert("L")
        stddev = float(ImageStat.Stat(grayscale).stddev[0])
        small = grayscale.resize((16, 16))
        pixels = list(small.getdata())
        mean = sum(pixels) / len(pixels)
        bits = "".join("1" if pixel >= mean else "0" for pixel in pixels)
        color_pixels = list(image.convert("RGB").resize((128, 128)).getdata())
        white_ratio = sum(
            red > 240 and green > 240 and blue > 240
            for red, green, blue in color_pixels
        ) / len(color_pixels)
        dark_ratio = sum(
            red < 180 and green < 180 and blue < 180
            for red, green, blue in color_pixels
        ) / len(color_pixels)
        red_ratio = sum(
            red > 150 and red > green * 1.25 and red > blue * 1.25
            for red, green, blue in color_pixels
        ) / len(color_pixels)

    return {
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": round(width / height, 4) if height else None,
        "grayscale_stddev": round(stddev, 4),
        "white_pixel_ratio": round(white_ratio, 4),
        "dark_pixel_ratio": round(dark_ratio, 4),
        "red_pixel_ratio": round(red_ratio, 4),
        "perceptual_hash": hashlib.sha256(bits.encode("ascii")).hexdigest()[:32],
    }


def assess_image(
    path: Path,
    category: str | None,
    page_number: int | None,
    page_signal: dict | None = None,
) -> dict:
    decision = {
        "source_image_path": relative_path(path),
        "filename": path.name,
        "page_number": page_number,
        "category": category,
        "decision": "keep",
        "reason": "unique meaningful-size embedded image",
    }
    try:
        decision.update(image_metrics(path))
    except (OSError, UnidentifiedImageError) as exc:
        return decision | {"decision": "reject", "reason": f"unreadable image: {exc}"}

    ratio = decision["aspect_ratio"]
    if decision["width"] < 100 or decision["height"] < 100 or decision["area"] < 20_000:
        decision.update(decision="reject", reason="small icon or label")
    elif ratio is not None and (ratio > 8 or ratio < 0.125):
        decision.update(decision="reject", reason="extreme decorative aspect ratio")
    elif decision["grayscale_stddev"] < 5:
        decision.update(decision="reject", reason="nearly blank or single-color image")
    elif (
        (
            decision["white_pixel_ratio"] > 0.85
            and decision["dark_pixel_ratio"] < 0.015
            and decision["red_pixel_ratio"] > 0.02
        )
        or (
            decision["white_pixel_ratio"] > 0.80
            and decision["dark_pixel_ratio"] < 0.02
            and decision["red_pixel_ratio"] > 0.10
        )
    ):
        decision.update(decision="reject", reason="watermark-only crop")
    elif category == "layout":
        filename = path.name.lower()
        signal = page_signal or {}
        if "_table_" in filename:
            decision["reason"] = "layout table or form crop"
        elif "_chart_" in filename:
            decision["reason"] = "layout chart crop"
        elif signal.get("has_table") or signal.get("has_parse_concerns") or signal.get("low_text"):
            decision["reason"] = "layout visual from complex or low-text page"
        else:
            decision.update(
                decision="reject",
                reason="layout image without page-complexity signal",
            )
    return decision


def build_page_signals(parsed: dict) -> dict[int, dict]:
    signals = {}
    for page in parsed.get("pages", []):
        items = page.get("items", [])
        useful_items = [
            item for item in items if item.get("type") not in {"header", "footer"}
        ]
        text_chars = sum(
            len(str(item.get("md") or item.get("value") or ""))
            for item in useful_items
        )
        signals[page.get("page_number")] = {
            "has_table": any(item.get("type") == "table" for item in useful_items),
            "has_parse_concerns": any(
                bool(item.get("parse_concerns")) for item in useful_items
            ),
            "low_text": text_chars < 250,
            "text_chars": text_chars,
        }
    return signals


def filter_pdf_images(
    domain: str,
    source_name: str,
    metadata_path: Path,
    page_signals: dict[int, dict],
):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = (metadata.get("images_content_metadata") or {}).get("images") or []
    image_dir = PROJECT_ROOT / "data" / "images" / domain / source_name
    decisions = []

    for record in records:
        filename = Path(record.get("filename") or "").name
        path = image_dir / filename
        page_number = page_from_filename(filename)
        if path.exists():
            decision = assess_image(
                path,
                record.get("category"),
                page_number,
                page_signals.get(page_number),
            )
        else:
            decision = {
                "source_image_path": relative_path(path),
                "filename": filename,
                "page_number": page_number,
                "category": record.get("category"),
                "decision": "reject",
                "reason": "downloaded image file is missing",
            }
        decision["bbox"] = record.get("bbox")
        decisions.append(decision)

    decisions_by_hash = defaultdict(list)
    for decision in decisions:
        if decision["decision"] == "keep" and decision.get("perceptual_hash"):
            decisions_by_hash[decision["perceptual_hash"]].append(decision)

    for matching in decisions_by_hash.values():
        pages = {item.get("page_number") for item in matching}
        if len(pages) >= 3:
            for decision in matching:
                decision.update(
                    decision="reject",
                    reason="repeated logo, watermark, or decorative image",
                )
            continue

        if len(matching) > 1:
            matching.sort(
                key=lambda item: (
                    item.get("category") == "embedded",
                    "_table_" in item["filename"].lower()
                    or "_chart_" in item["filename"].lower(),
                    item.get("area") or 0,
                ),
                reverse=True,
            )
            for duplicate in matching[1:]:
                duplicate.update(
                    decision="reject",
                    reason="duplicate embedded/layout visual",
                )

    kept_by_page_and_category = defaultdict(lambda: defaultdict(list))
    for decision in decisions:
        if decision["decision"] == "keep":
            kept_by_page_and_category[decision.get("page_number")][
                decision.get("category")
            ].append(decision)
    for categories in kept_by_page_and_category.values():
        embedded = categories.get("embedded", [])
        for layout in categories.get("layout", []):
            layout_bbox = layout.get("bbox") or {}
            layout_area = (layout_bbox.get("w") or 0) * (layout_bbox.get("h") or 0)
            if not layout_area:
                continue
            for source in embedded:
                source_bbox = source.get("bbox") or {}
                source_area = (source_bbox.get("w") or 0) * (source_bbox.get("h") or 0)
                if not source_area:
                    continue
                width = max(
                    0,
                    min(
                        layout_bbox.get("x", 0) + layout_bbox.get("w", 0),
                        source_bbox.get("x", 0) + source_bbox.get("w", 0),
                    )
                    - max(layout_bbox.get("x", 0), source_bbox.get("x", 0)),
                )
                height = max(
                    0,
                    min(
                        layout_bbox.get("y", 0) + layout_bbox.get("h", 0),
                        source_bbox.get("y", 0) + source_bbox.get("h", 0),
                    )
                    - max(layout_bbox.get("y", 0), source_bbox.get("y", 0)),
                )
                overlap = width * height
                area_ratio = layout_area / source_area
                if (
                    0.7 <= area_ratio <= 1.4
                    and overlap / min(layout_area, source_area) >= 0.9
                ):
                    layout.update(
                        decision="reject",
                        reason="layout crop duplicates embedded image",
                    )
                    break

    kept_layout_by_page = defaultdict(list)
    for decision in decisions:
        if decision["decision"] == "keep" and decision.get("category") == "layout":
            kept_layout_by_page[decision.get("page_number")].append(decision)
    for page_decisions in kept_layout_by_page.values():
        page_decisions.sort(key=lambda item: item.get("area") or 0, reverse=True)
        for extra in page_decisions[4:]:
            extra.update(
                decision="reject",
                reason="extra layout visual beyond per-page limit",
            )

    kept_by_page = defaultdict(list)
    for decision in decisions:
        if decision["decision"] == "keep" and decision.get("page_number"):
            kept_by_page[decision["page_number"]].append(decision)
    return kept_by_page, decisions


def element(
    *,
    element_id: str,
    document_id: str,
    domain: str,
    source_type: str,
    source_name: str,
    sequence: int,
    element_type: str,
    section_title: str | None,
    section_path: list[str],
    page_number: int | None = None,
    source_url: str | None = None,
    text: str | None = None,
    table_markdown: str | None = None,
    table_html: str | None = None,
    image_path: str | None = None,
    image_url: str | None = None,
    bbox=None,
    source_row_numbers: list[int] | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "element_id": element_id,
        "document_id": document_id,
        "domain": domain,
        "source_type": source_type,
        "source_name": source_name,
        "source_url": source_url,
        "page_number": page_number,
        "sequence": sequence,
        "section_title": section_title,
        "section_path": section_path,
        "element_type": element_type,
        "text": text,
        "table_markdown": table_markdown,
        "table_html": table_html,
        "image_path": image_path,
        "image_url": image_url,
        "bbox": bbox,
        "source_row_numbers": source_row_numbers,
        "metadata": metadata or {},
    }


def normalize_pdf(structured_path: Path) -> dict:
    domain = structured_path.parts[-3]
    source_name = structured_path.name.removesuffix(".structured.json")
    input_path = PROJECT_ROOT / "data" / "input" / domain / f"{source_name}.pdf"
    document_id = str(create_document_id(domain, "pdf", relative_path(input_path)))
    parsed = json.loads(structured_path.read_text(encoding="utf-8"))
    kept_images, image_decisions = filter_pdf_images(
        domain,
        source_name,
        structured_path.with_name(f"{source_name}.images.json"),
        build_page_signals(parsed),
    )
    output_dir = PROJECT_ROOT / "data" / "normalized" / domain / "pdf"
    elements_path = output_dir / f"{source_name}.elements.jsonl"
    manifest_path = output_dir / f"{source_name}.image_manifest.jsonl"
    metadata_path = output_dir / f"{source_name}.metadata.json"
    elements = []
    headings: list[tuple[int, str]] = []
    ignored = Counter()
    sequence = 0

    for page in parsed.get("pages", []):
        page_number = page.get("page_number")
        ordered = [(bbox_y(item), 0, i, "item", item) for i, item in enumerate(page.get("items", []))]
        for i, image in enumerate(kept_images.get(page_number, [])):
            ordered.append(((image.get("bbox") or {}).get("y", float("inf")), 1, i, "image", image))

        ordered = sorted(ordered)
        for ordered_index, (_, _, _, kind, record) in enumerate(ordered):
            if kind == "image":
                nearby_before = None
                nearby_after = None
                nearby_table = None
                for neighbor in reversed(ordered[:ordered_index]):
                    if neighbor[3] != "item":
                        continue
                    item = neighbor[4]
                    if item.get("type") in {"header", "footer", "heading"}:
                        continue
                    value = clean_text(item.get("md") or item.get("value"))
                    if value:
                        nearby_before = value[:1200]
                        if item.get("type") == "table":
                            nearby_table = item.get("md")
                        break
                for neighbor in ordered[ordered_index + 1:]:
                    if neighbor[3] != "item":
                        continue
                    item = neighbor[4]
                    if item.get("type") in {"header", "footer", "heading"}:
                        continue
                    value = clean_text(item.get("md") or item.get("value"))
                    if value:
                        nearby_after = value[:1200]
                        if nearby_table is None and item.get("type") == "table":
                            nearby_table = item.get("md")
                        break

                sequence += 1
                elements.append(element(
                    element_id=stable_id(document_id, str(page_number), record["filename"]),
                    document_id=document_id, domain=domain, source_type="pdf",
                    source_name=source_name, sequence=sequence, element_type="image",
                    section_title=headings[-1][1] if headings else None,
                    section_path=[entry[1] for entry in headings],
                    page_number=page_number, image_path=record["source_image_path"],
                    bbox=record.get("bbox"),
                    metadata={
                        "width": record.get("width"),
                        "height": record.get("height"),
                        "image_category": record.get("category"),
                        "selection_reason": record.get("reason"),
                        "nearby_text_before": nearby_before,
                        "nearby_text_after": nearby_after,
                        "nearby_table_markdown": nearby_table,
                    },
                ))
                continue

            parser_type = record.get("type")
            if parser_type in {"header", "footer"}:
                ignored[parser_type] += 1
                continue
            if parser_type == "heading":
                text = clean_text(record.get("md") or record.get("value"))
                if not text:
                    continue
                level = int(record.get("level") or 1)
                headings = [entry for entry in headings if entry[0] < level]
                headings.append((level, text))
                normalized_type = "heading"
            elif parser_type == "table":
                text = clean_text(record.get("value") or record.get("md"))
                normalized_type = "table"
            else:
                text = clean_text(record.get("md") or record.get("value"))
                normalized_type = "text"
            if not text:
                continue

            sequence += 1
            elements.append(element(
                element_id=stable_id(document_id, str(page_number), str(sequence), normalized_type),
                document_id=document_id, domain=domain, source_type="pdf",
                source_name=source_name, sequence=sequence, element_type=normalized_type,
                section_title=headings[-1][1] if headings else None,
                section_path=[entry[1] for entry in headings], page_number=page_number,
                text=text, table_markdown=record.get("md") if parser_type == "table" else None,
                table_html=record.get("html") if parser_type == "table" else None,
                bbox=record.get("bbox"),
                metadata={"parser_type": parser_type, "parse_concerns": record.get("parse_concerns")},
            ))

    write_jsonl(elements_path, elements)
    write_jsonl(manifest_path, image_decisions)
    metadata = {
        "document_id": document_id, "domain": domain, "source_type": "pdf",
        "source_name": source_name, "element_count": len(elements),
        "element_counts": dict(Counter(item["element_type"] for item in elements)),
        "ignored_counts": dict(ignored),
        "image_decisions": dict(Counter(item["decision"] for item in image_decisions)),
        "outputs": {"elements_path": relative_path(elements_path), "image_manifest_path": relative_path(manifest_path)},
    }
    write_json(metadata_path, metadata)
    return metadata


def html_image_noise(item: dict) -> str | None:
    combined = " ".join([item.get("image_url") or "", item.get("alt_text") or "", item.get("title") or ""]).lower()
    return "logo, icon, or interface image" if any(term in combined for term in NOISY_IMAGE_TERMS) else None


def download_html_image(item: dict, output_dir: Path):
    image_url = item.get("image_url")
    decision = {"image_url": image_url, "decision": "keep", "reason": "meaningful HTML content image"}
    noise = html_image_noise(item)
    if noise:
        return None, decision | {"decision": "reject", "reason": noise}

    suffix = Path(image_url.split("?", 1)[0]).suffix.lower()
    suffix = suffix if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"} else ".img"
    path = output_dir / f"html_{hashlib.sha256(image_url.encode()).hexdigest()[:16]}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists():
            urllib.request.urlretrieve(image_url, path)
        decision.update(image_metrics(path))
        if decision["width"] < 100 or decision["height"] < 100 or decision["area"] < 20_000:
            path.unlink(missing_ok=True)
            return None, decision | {"decision": "reject", "reason": "small icon or label"}
        if decision["grayscale_stddev"] < 5:
            path.unlink(missing_ok=True)
            return None, decision | {"decision": "reject", "reason": "nearly blank image"}
    except Exception as exc:
        path.unlink(missing_ok=True)
        return None, decision | {"decision": "reject", "reason": f"download or validation failed: {exc}"}
    decision["local_image_path"] = relative_path(path)
    return relative_path(path), decision


def normalize_html(structured_path: Path) -> dict:
    domain = structured_path.parts[-3]
    source_name = structured_path.name.removesuffix(".structured.json")
    parsed = json.loads(structured_path.read_text(encoding="utf-8"))
    document_id = str(create_document_id(domain, "html", parsed.get("root_url") or ""))
    output_dir = PROJECT_ROOT / "data" / "normalized" / domain / "html"
    image_dir = PROJECT_ROOT / "data" / "images" / domain / source_name
    elements_path = output_dir / f"{source_name}.elements.jsonl"
    manifest_path = output_dir / f"{source_name}.image_manifest.jsonl"
    metadata_path = output_dir / f"{source_name}.metadata.json"
    elements, decisions, seen_images, seen_content = [], [], set(), set()
    ignored = Counter()
    sequence = 0

    for page in parsed.get("pages", []):
        for section in page.get("sections", []):
            title = clean_text(section.get("heading"))
            path = [clean_text(value) for value in section.get("section_path", []) if clean_text(value)]
            for item in section.get("elements", []):
                kind = item.get("type")
                text = clean_text(item.get("text"))
                if kind == "text" and text.lower() in NOISY_HTML_TEXT:
                    ignored["navigation_text"] += 1
                    continue
                if kind in {"text", "table"}:
                    dedupe_value = (
                        item.get("table_markdown") if kind == "table" else text
                    ) or ""
                    content_hash = hashlib.sha256(
                        clean_text(dedupe_value).lower().encode("utf-8")
                    ).hexdigest()
                    if content_hash in seen_content:
                        ignored["duplicate_content"] += 1
                        continue
                    seen_content.add(content_hash)
                image_path = None
                image_url = item.get("image_url") if kind == "image" else None
                if kind == "image":
                    if image_url in seen_images:
                        ignored["duplicate_image"] += 1
                        continue
                    seen_images.add(image_url)
                    image_path, decision = download_html_image(item, image_dir)
                    decision.update(section_title=title, source_url=page.get("source_url"))
                    decisions.append(decision)
                    if not image_path:
                        continue
                if kind not in {"heading", "text", "table", "image"}:
                    ignored[kind or "unknown"] += 1
                    continue

                sequence += 1
                elements.append(element(
                    element_id=item.get("element_id") or stable_id(document_id, str(sequence)),
                    document_id=document_id, domain=domain, source_type="html",
                    source_name=source_name, sequence=sequence, element_type=kind,
                    section_title=title, section_path=path,
                    page_number=page.get("page_number"), source_url=page.get("source_url"),
                    text=text or None, table_markdown=item.get("table_markdown"),
                    table_html=item.get("table_html"), image_path=image_path, image_url=image_url,
                    metadata={
                        "tag": item.get("tag"),
                        "alt_text": item.get("alt_text"),
                        "title": item.get("title"),
                        "audience": (
                            "consumer"
                            if "audience=consumer" in (page.get("source_url") or "")
                            else "professional"
                        ),
                    },
                ))

    write_jsonl(elements_path, elements)
    write_jsonl(manifest_path, decisions)
    metadata = {
        "document_id": document_id, "domain": domain, "source_type": "html",
        "source_name": source_name, "element_count": len(elements),
        "element_counts": dict(Counter(item["element_type"] for item in elements)),
        "ignored_counts": dict(ignored),
        "image_decisions": dict(Counter(item["decision"] for item in decisions)),
        "outputs": {"elements_path": relative_path(elements_path), "image_manifest_path": relative_path(manifest_path)},
    }
    write_json(metadata_path, metadata)
    return metadata


def normalize_csv(category_docs_path: Path) -> dict:
    domain = category_docs_path.parts[-3]
    source_name = category_docs_path.name.removesuffix(".hospital_category_docs.jsonl")
    output_dir = PROJECT_ROOT / "data" / "normalized" / domain / "csv"
    elements_path = output_dir / f"{source_name}.elements.jsonl"
    metadata_path = output_dir / f"{source_name}.metadata.json"

    def values():
        with category_docs_path.open("r", encoding="utf-8") as source:
            for sequence, line in enumerate(source, start=1):
                item = json.loads(line)
                metadata = item.get("metadata") or {}
                yield element(
                    element_id=item["category_doc_id"], document_id=item["document_id"],
                    domain=domain, source_type="csv", source_name=source_name,
                    sequence=sequence, element_type="table", section_title=item["category"],
                    section_path=[metadata.get("hospital_name"), item["category"]],
                    text=item["retrieval_text"], table_markdown=item["table_markdown"],
                    source_row_numbers=item["source_row_numbers"],
                    metadata={**metadata, "profile_id": item["profile_id"], "facility_id": item["facility_id"],
                              "measure_ids": item["measure_ids"], "already_chunked": True},
                )

    count = write_jsonl(elements_path, values())
    metadata = {
        "domain": domain, "source_type": "csv", "source_name": source_name,
        "element_count": count, "element_counts": {"table": count},
        "outputs": {"elements_path": relative_path(elements_path)},
    }
    write_json(metadata_path, metadata)
    return metadata


def normalize_all() -> list[dict]:
    parsed_dir = PROJECT_ROOT / "data" / "parsed"
    results = []
    for path in sorted(parsed_dir.glob("*/*/*.structured.json")):
        results.append(normalize_pdf(path) if path.parts[-2] == "pdf" else normalize_html(path))
    for path in sorted(parsed_dir.glob("*/*/*.hospital_category_docs.jsonl")):
        results.append(normalize_csv(path))
    write_json(PROJECT_ROOT / "data" / "normalized" / "summary.json", {"documents": results})
    return results
