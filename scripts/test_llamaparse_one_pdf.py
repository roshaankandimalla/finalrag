import argparse
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import LlamaCloud


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def save_model(result, output_path: Path):
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def pretty_print_json_file(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_markdown(result, output_path: Path):
    if not result.markdown:
        return

    pages = []

    for page in result.markdown.pages:
        markdown = getattr(page, "markdown", None)

        if markdown:
            pages.append(
                f"<!-- Page {page.page_number} -->\n\n{markdown}"
            )

    output_path.write_text(
        "\n\n---\n\n".join(pages),
        encoding="utf-8",
    )


def download_raw_structured_json(
    result,
    output_path: Path,
):
    metadata = result.result_content_metadata or {}
    items_metadata = metadata.get("items")

    if not items_metadata or not items_metadata.presigned_url:
        raise RuntimeError("Structured JSON download URL is unavailable")

    urllib.request.urlretrieve(
        items_metadata.presigned_url,
        output_path,
    )


def download_images(result, image_dir: Path):
    image_dir.mkdir(parents=True, exist_ok=True)

    if not result.images_content_metadata:
        return 0

    downloaded = 0

    for image in result.images_content_metadata.images:
        if not image.presigned_url:
            continue

        output_path = image_dir / image.filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(
                image.presigned_url,
                output_path,
            )
            downloaded += 1
            print(f"Downloaded image: {image.filename}")
        except Exception as exc:
            print(f"Failed to download {image.filename}: {exc}")

    return downloaded


def inspect_structured_json(structured_path: Path):
    data = json.loads(structured_path.read_text(encoding="utf-8"))

    pages = data.get("pages", [])
    tables = 0
    images = 0
    item_types = set()

    for page in pages:
        for item in page.get("items", []):
            item_type = item.get("type")
            item_types.add(item_type)

            if item_type == "table":
                tables += 1
            elif item_type == "image":
                images += 1

    print("\nStructured result summary")
    print(f"Pages:       {len(pages)}")
    print(f"Tables:      {tables}")
    print(f"Images:      {images}")
    print(f"Item types:  {sorted(item_types)}")


def main():
    args = parse_arguments()

    pdf_path = args.pdf_path
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    load_dotenv(PROJECT_ROOT / ".env", override=True)

    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is missing from .env")

    client = LlamaCloud(
        api_key=api_key,
        timeout=600,
        max_retries=5,
    )

    document_name = pdf_path.stem

    parsed_dir = (
        PROJECT_ROOT / "data" / "parsed" / args.domain / "pdf"
    )
    image_dir = (
        PROJECT_ROOT / "data" / "images" / args.domain / document_name
    )

    parsed_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading existing job: {args.job_id}")

    markdown_result = client.parsing.get(
        args.job_id,
        expand=["markdown"],
        timeout=600,
    )

    metadata_result = client.parsing.get(
        args.job_id,
        expand=["metadata"],
        timeout=600,
    )

    images_result = client.parsing.get(
        args.job_id,
        expand=["images_content_metadata"],
        timeout=600,
    )

    items_metadata_result = client.parsing.get(
        args.job_id,
        expand=["items_content_metadata"],
        timeout=600,
    )

    markdown_path = parsed_dir / f"{document_name}.md"
    metadata_path = parsed_dir / f"{document_name}.metadata.json"
    images_metadata_path = parsed_dir / f"{document_name}.images.json"
    structured_path = parsed_dir / f"{document_name}.structured.json"

    save_markdown(markdown_result, markdown_path)
    save_model(metadata_result, metadata_path)
    save_model(images_result, images_metadata_path)

    download_raw_structured_json(
        items_metadata_result,
        structured_path,
    )
    pretty_print_json_file(structured_path)

    image_count = download_images(images_result, image_dir)

    inspect_structured_json(structured_path)

    print("\nSaved outputs")
    print(f"Markdown:        {markdown_path}")
    print(f"Structured JSON: {structured_path}")
    print(f"Metadata JSON:   {metadata_path}")
    print(f"Images JSON:     {images_metadata_path}")
    print(f"Images folder:   {image_dir}")
    print(f"Downloaded images: {image_count}")


if __name__ == "__main__":
    main()
