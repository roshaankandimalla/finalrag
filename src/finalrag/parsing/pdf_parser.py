import json
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

from llama_cloud import LlamaCloud


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parse_options() -> dict:
    """Return the shared LlamaParse v2 configuration for project PDFs."""
    return {
        "tier": "agentic",
        "version": "latest",
        "agentic_options": {
            "custom_prompt": (
                "Preserve document structure, headings, tables, charts, "
                "flowcharts, and text visible inside images. Keep related "
                "content together and do not summarize."
            ),
        },
        "processing_options": {
            "aggressive_table_extraction": True,
            "specialized_chart_parsing": "agentic",
            "ocr_parameters": {"languages": ["en"]},
        },
        "output_options": {
            "extract_printed_page_number": True,
            "images_to_save": ["embedded", "layout"],
            "markdown": {
                "annotate_links": True,
                "inline_images": False,
                "tables": {
                    "output_tables_as_markdown": True,
                    "compact_markdown_tables": False,
                    "merge_continued_tables": True,
                },
            },
        },
    }


def _save_model(result, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_markdown(result, output_path: Path) -> None:
    pages = []

    if result.markdown:
        for page in result.markdown.pages:
            markdown = getattr(page, "markdown", None)
            if markdown:
                pages.append(f"<!-- Page {page.page_number} -->\n\n{markdown}")

    output_path.write_text("\n\n---\n\n".join(pages), encoding="utf-8")


def _download_structured_json(result, output_path: Path) -> None:
    content_metadata = result.result_content_metadata or {}
    items_metadata = content_metadata.get("items")

    if not items_metadata or not items_metadata.presigned_url:
        raise RuntimeError("Structured JSON download URL is unavailable")

    urllib.request.urlretrieve(items_metadata.presigned_url, output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _safe_image_name(filename: str, index: int) -> str:
    clean_name = Path(filename).name
    clean_name = re.sub(r"[^A-Za-z0-9._-]", "_", clean_name)
    return clean_name or f"image_{index:04d}.png"


def _download_images(result, image_dir: Path) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)

    if not result.images_content_metadata:
        return 0

    downloaded = 0
    for index, image in enumerate(result.images_content_metadata.images, start=1):
        if not image.presigned_url:
            continue

        output_path = image_dir / _safe_image_name(image.filename, index)
        urllib.request.urlretrieve(image.presigned_url, output_path)
        downloaded += 1

    return downloaded


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def find_existing_pdf_outputs(file_path: str | Path, domain: str) -> dict | None:
    """Return a manifest when all required raw outputs already exist."""
    pdf_path = Path(file_path)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    parsed_dir = PROJECT_ROOT / "data" / "parsed" / domain / "pdf"
    image_dir = PROJECT_ROOT / "data" / "images" / domain / pdf_path.stem

    markdown_path = parsed_dir / f"{pdf_path.stem}.md"
    metadata_path = parsed_dir / f"{pdf_path.stem}.metadata.json"
    images_metadata_path = parsed_dir / f"{pdf_path.stem}.images.json"
    structured_path = parsed_dir / f"{pdf_path.stem}.structured.json"

    required_paths = [
        markdown_path,
        metadata_path,
        images_metadata_path,
        structured_path,
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required_paths):
        return None

    image_count = (
        sum(1 for path in image_dir.rglob("*") if path.is_file())
        if image_dir.exists()
        else 0
    )

    return {
        "markdown_path": _relative_path(markdown_path),
        "structured_json_path": _relative_path(structured_path),
        "metadata_json_path": _relative_path(metadata_path),
        "images_json_path": _relative_path(images_metadata_path),
        "images_directory": _relative_path(image_dir),
        "downloaded_image_count": image_count,
        "reused_existing_outputs": True,
    }


def parse_pdf(
    file_path: str | Path,
    domain: str,
    api_key: str,
    existing_job_id: str | None = None,
    on_job_created: Callable[[str], None] | None = None,
) -> dict:
    """Create/resume one LlamaParse job and save all raw parser outputs."""
    pdf_path = Path(file_path)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    client = LlamaCloud(api_key=api_key, timeout=600, max_retries=5)
    job_id = existing_job_id

    if job_id:
        print(f"[{pdf_path.name}] Resuming LlamaParse job {job_id}")
    else:
        print(f"[{pdf_path.name}] Creating LlamaParse job")
        with pdf_path.open("rb") as pdf_file:
            job = client.parsing.create(
                upload_file=pdf_file,
                **build_parse_options(),
            )

        job_id = job.id
        if on_job_created:
            on_job_created(job_id)

    print(f"[{pdf_path.name}] Waiting for job {job_id}")
    client.parsing.wait_for_completion(
        job_id,
        polling_interval=5,
        max_interval=30,
        timeout=14400,
        backoff="linear",
        verbose=True,
    )

    parsed_dir = PROJECT_ROOT / "data" / "parsed" / domain / "pdf"
    image_dir = PROJECT_ROOT / "data" / "images" / domain / pdf_path.stem
    parsed_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = parsed_dir / f"{pdf_path.stem}.md"
    metadata_path = parsed_dir / f"{pdf_path.stem}.metadata.json"
    images_metadata_path = parsed_dir / f"{pdf_path.stem}.images.json"
    structured_path = parsed_dir / f"{pdf_path.stem}.structured.json"

    print(f"[{pdf_path.name}] Downloading parse outputs")
    markdown_result = client.parsing.get(
        job_id, expand=["markdown"], timeout=600
    )
    metadata_result = client.parsing.get(
        job_id, expand=["metadata"], timeout=600
    )
    images_result = client.parsing.get(
        job_id, expand=["images_content_metadata"], timeout=600
    )
    items_result = client.parsing.get(
        job_id, expand=["items_content_metadata"], timeout=600
    )

    _save_markdown(markdown_result, markdown_path)
    _save_model(metadata_result, metadata_path)
    _save_model(images_result, images_metadata_path)
    _download_structured_json(items_result, structured_path)
    image_count = _download_images(images_result, image_dir)

    return {
        "job_id": job_id,
        "markdown_path": _relative_path(markdown_path),
        "structured_json_path": _relative_path(structured_path),
        "metadata_json_path": _relative_path(metadata_path),
        "images_json_path": _relative_path(images_metadata_path),
        "images_directory": _relative_path(image_dir),
        "downloaded_image_count": image_count,
    }
