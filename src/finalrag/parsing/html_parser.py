import json
import re
import uuid
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag
from firecrawl import Firecrawl


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CONTENT_TAGS = HEADING_TAGS | {
    "p",
    "ul",
    "ol",
    "table",
    "img",
    "blockquote",
    "pre",
}


def stable_id(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, ":".join(parts)))


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def save_json(value, path: Path) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def save_jsonl(values: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False, default=str))
            output.write("\n")


def table_to_markdown(table: Tag) -> str:
    rows = []
    for row in table.find_all("tr"):
        cells = [
            clean_text(cell.get_text(" ", strip=True)).replace("|", r"\|")
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def element_from_tag(
    tag: Tag,
    source_url: str,
    page_id: str,
    sequence: int,
) -> dict | None:
    element_id = stable_id(page_id, str(sequence), tag.name)

    if tag.name in HEADING_TAGS:
        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            return None
        return {
            "element_id": element_id,
            "sequence": sequence,
            "type": "heading",
            "heading_level": int(tag.name[1]),
            "text": text,
        }

    if tag.name == "table":
        markdown = table_to_markdown(tag)
        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            return None
        return {
            "element_id": element_id,
            "sequence": sequence,
            "type": "table",
            "text": text,
            "table_html": str(tag),
            "table_markdown": markdown,
        }

    if tag.name == "img":
        source = tag.get("src")
        if not source:
            return None
        return {
            "element_id": element_id,
            "sequence": sequence,
            "type": "image",
            "image_url": urljoin(source_url, source),
            "alt_text": clean_text(tag.get("alt")),
            "title": clean_text(tag.get("title")),
        }

    text = clean_text(tag.get_text(" ", strip=True))
    if not text:
        return None

    return {
        "element_id": element_id,
        "sequence": sequence,
        "type": "text",
        "tag": tag.name,
        "text": text,
    }


def extract_structured_page(page: dict, page_number: int) -> dict:
    metadata = page.get("metadata") or {}
    source_url = (
        metadata.get("sourceURL")
        or metadata.get("source_url")
        or metadata.get("url")
        or ""
    )
    title = metadata.get("title") or f"Crawled page {page_number}"
    page_id = stable_id(source_url or title, str(page_number))
    html = page.get("html") or page.get("raw_html") or ""

    soup = BeautifulSoup(html, "lxml")
    for unwanted in soup(["script", "style", "noscript", "template"]):
        unwanted.decompose()

    sections = []
    current_section = {
        "section_id": stable_id(page_id, "root"),
        "heading": title,
        "heading_level": 0,
        "section_path": [title],
        "elements": [],
    }
    sections.append(current_section)
    heading_stack: list[tuple[int, str]] = []
    sequence = 0

    for tag in soup.find_all(CONTENT_TAGS):
        if not isinstance(tag, Tag):
            continue

        # Avoid duplicating content nested inside lists, tables, and blockquotes.
        if tag.name not in HEADING_TAGS and tag.find_parent(
            ["table", "ul", "ol", "blockquote", "pre"]
        ):
            continue

        sequence += 1
        element = element_from_tag(tag, source_url, page_id, sequence)
        if not element:
            continue

        if element["type"] == "heading":
            level = element["heading_level"]
            heading = element["text"]
            heading_stack = [
                item for item in heading_stack if item[0] < level
            ]
            heading_stack.append((level, heading))
            current_section = {
                "section_id": stable_id(
                    page_id,
                    str(sequence),
                    heading,
                ),
                "heading": heading,
                "heading_level": level,
                "section_path": [item[1] for item in heading_stack],
                "elements": [element],
            }
            sections.append(current_section)
        else:
            current_section["elements"].append(element)

    sections = [section for section in sections if section["elements"]]
    return {
        "page_id": page_id,
        "page_number": page_number,
        "source_url": source_url,
        "title": title,
        "description": metadata.get("description"),
        "status_code": metadata.get("statusCode") or metadata.get("status_code"),
        "language": metadata.get("language"),
        "sections": sections,
        "links": page.get("links") or [],
        "images": page.get("images") or [],
        "metadata": metadata,
    }


def save_combined_markdown(pages: list[dict], path: Path) -> None:
    blocks = []
    for index, page in enumerate(pages, start=1):
        metadata = page.get("metadata") or {}
        source_url = (
            metadata.get("sourceURL")
            or metadata.get("source_url")
            or metadata.get("url")
            or ""
        )
        blocks.append(
            "\n".join(
                [
                    f"# Crawled Page {index}",
                    f"Source URL: {source_url}",
                    "",
                    page.get("markdown") or "",
                ]
            )
        )
    path.write_text("\n\n---\n\n".join(blocks), encoding="utf-8")


def find_existing_html_outputs(name: str, domain: str) -> dict | None:
    output_dir = PROJECT_ROOT / "data" / "parsed" / domain / "html"
    paths = {
        "raw_json_path": output_dir / f"{name}.raw.json",
        "pages_jsonl_path": output_dir / f"{name}.pages.jsonl",
        "markdown_path": output_dir / f"{name}.md",
        "structured_json_path": output_dir / f"{name}.structured.json",
    }
    if not all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        return None

    structured = json.loads(paths["structured_json_path"].read_text(encoding="utf-8"))
    return {
        key: path.relative_to(PROJECT_ROOT).as_posix()
        for key, path in paths.items()
    } | {
        "page_count": len(structured.get("pages", [])),
        "section_count": sum(
            len(page.get("sections", [])) for page in structured.get("pages", [])
        ),
        "reused_existing_outputs": True,
    }


def parse_html_crawl(
    name: str,
    domain: str,
    config: dict,
    api_key: str,
) -> dict:
    output_dir = PROJECT_ROOT / "data" / "parsed" / domain / "html"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = Firecrawl(api_key=api_key, timeout=600, max_retries=5)
    scrape_config = config.get("scrapeOptions") or {}

    print(f"[{name}] Starting Firecrawl crawl")
    result = client.crawl(
        url=config["url"],
        limit=config.get("limit", 10),
        max_discovery_depth=config.get("maxDiscoveryDepth", 1),
        sitemap=config.get("sitemap", "skip"),
        allow_external_links=config.get("allowExternalLinks", False),
        allow_subdomains=config.get("allowSubdomains", False),
        crawl_entire_domain=config.get("crawlEntireDomain", False),
        ignore_query_parameters=config.get("ignoreQueryParameters", False),
        formats=scrape_config.get("formats", ["markdown", "html"]),
        only_main_content=scrape_config.get("onlyMainContent", False),
        remove_base64_images=True,
        poll_interval=5,
        timeout=3600,
        request_timeout=600,
    )

    raw_result = result.model_dump(mode="json")
    pages = raw_result.get("data") or []
    if not pages:
        raise RuntimeError(f"Firecrawl returned no pages for {name}")

    raw_path = output_dir / f"{name}.raw.json"
    pages_path = output_dir / f"{name}.pages.jsonl"
    markdown_path = output_dir / f"{name}.md"
    structured_path = output_dir / f"{name}.structured.json"

    save_json(raw_result, raw_path)
    save_jsonl(pages, pages_path)
    save_combined_markdown(pages, markdown_path)

    structured_pages = [
        extract_structured_page(page, index)
        for index, page in enumerate(pages, start=1)
    ]
    structured = {
        "name": name,
        "domain": domain,
        "source_type": "html",
        "root_url": config["url"],
        "pages": structured_pages,
    }
    save_json(structured, structured_path)

    section_count = sum(len(page["sections"]) for page in structured_pages)
    return {
        "page_count": len(pages),
        "section_count": section_count,
        "credits_used": raw_result.get("credits_used"),
        "raw_json_path": raw_path.relative_to(PROJECT_ROOT).as_posix(),
        "pages_jsonl_path": pages_path.relative_to(PROJECT_ROOT).as_posix(),
        "markdown_path": markdown_path.relative_to(PROJECT_ROOT).as_posix(),
        "structured_json_path": structured_path.relative_to(PROJECT_ROOT).as_posix(),
    }
