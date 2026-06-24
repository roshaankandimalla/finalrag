import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

THUMBNAIL_SIZE = (180, 135)
CELL_SIZE = (200, 175)
COLUMNS = 5
IMAGES_PER_SHEET = 60


def read_manifest(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("decision") == "keep":
                records.append(record)
    return records


def output_path(source_name: str, sheet_number: int) -> Path:
    suffix = "" if sheet_number == 1 else f"_{sheet_number:03d}"
    return (
        PROJECT_ROOT
        / "data"
        / "normalized"
        / "image_previews"
        / f"{source_name}.selected_images{suffix}.jpg"
    )


def build_sheet(records: list[dict], source_name: str, sheet_number: int) -> Path:
    rows = (len(records) + COLUMNS - 1) // COLUMNS
    sheet = Image.new(
        "RGB",
        (COLUMNS * CELL_SIZE[0], rows * CELL_SIZE[1]),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, record in enumerate(records):
        row, column = divmod(index, COLUMNS)
        left = column * CELL_SIZE[0] + 10
        top = row * CELL_SIZE[1] + 8
        image_path = PROJECT_ROOT / record["source_image_path"]
        with Image.open(image_path) as image:
            thumbnail = ImageOps.contain(image.convert("RGB"), THUMBNAIL_SIZE)
        image_left = left + (THUMBNAIL_SIZE[0] - thumbnail.width) // 2
        image_top = top + (THUMBNAIL_SIZE[1] - thumbnail.height) // 2
        sheet.paste(thumbnail, (image_left, image_top))
        label = (
            f"p{record.get('page_number') or '-'} "
            f"{record.get('category', '')} "
            f"{record.get('filename', '')[:22]}"
        )
        draw.text((left, top + THUMBNAIL_SIZE[1] + 8), label, fill="black")

    destination = output_path(source_name, sheet_number)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=88, optimize=True)
    return destination


def main() -> None:
    manifests = sorted(
        (PROJECT_ROOT / "data" / "normalized").glob("*/*/*.image_manifest.jsonl")
    )
    created = []
    for manifest in manifests:
        records = read_manifest(manifest)
        source_name = manifest.name.removesuffix(".image_manifest.jsonl")
        preview_dir = PROJECT_ROOT / "data" / "normalized" / "image_previews"
        for stale in preview_dir.glob(f"{source_name}.selected_images*.jpg"):
            stale.unlink()
        for offset in range(0, len(records), IMAGES_PER_SHEET):
            created.append(
                build_sheet(
                    records[offset : offset + IMAGES_PER_SHEET],
                    source_name,
                    offset // IMAGES_PER_SHEET + 1,
                )
            )
    print(f"Created image preview sheets: {len(created)}")
    for path in created:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
