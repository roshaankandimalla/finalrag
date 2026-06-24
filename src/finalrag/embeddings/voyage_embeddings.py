import hashlib
import math
import os
from pathlib import Path

import voyageai
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "voyage-multimodal-3.5"
DEFAULT_DIMENSION = 2048
DEFAULT_BATCH_SIZE = 64
MAX_IMAGE_PIXELS = 16_000_000
MAX_INPUT_TOKENS = 32_000
MAX_BATCH_TOKENS = 280_000
IMAGE_PIXELS_PER_TOKEN = 560
INPUT_TOKEN_RESERVE = 2_000
EMBEDDING_INPUT_VERSION = "voyage-mm-v1-resize-32k"


def resolve_image_path(image_path: str, project_root: Path = PROJECT_ROOT) -> Path:
    resolved_root = project_root.resolve()
    resolved = (resolved_root / image_path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Image path leaves project directory: {image_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing chunk image: {image_path}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_input_hash(chunk: dict, project_root: Path = PROJECT_ROOT) -> str:
    digest = hashlib.sha256()
    digest.update(EMBEDDING_INPUT_VERSION.encode("utf-8"))
    digest.update((chunk.get("retrieval_text") or "").encode("utf-8"))
    for image_path in chunk.get("image_paths") or []:
        digest.update(image_path.encode("utf-8"))
        digest.update(file_sha256(resolve_image_path(image_path, project_root)).encode())
    return digest.hexdigest()


def prepare_multimodal_input(
    chunk: dict,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list, list[Image.Image]]:
    retrieval_text = (chunk.get("retrieval_text") or "").strip()
    if not retrieval_text:
        raise ValueError(f"Chunk {chunk.get('chunk_id')} has empty retrieval_text")

    opened_images = []
    for image_path in chunk.get("image_paths") or []:
        image = Image.open(resolve_image_path(image_path, project_root))
        image.load()
        opened_images.append(image)

    text_tokens = chunk.get("token_count") or max(1, math.ceil(len(retrieval_text) / 3))
    pixel_budget = max(
        IMAGE_PIXELS_PER_TOKEN,
        (MAX_INPUT_TOKENS - INPUT_TOKEN_RESERVE - text_tokens)
        * IMAGE_PIXELS_PER_TOKEN,
    )
    total_pixels = sum(image.width * image.height for image in opened_images)
    max_pixels = max(
        (image.width * image.height for image in opened_images),
        default=0,
    )
    scale = 1.0
    if total_pixels > pixel_budget:
        scale = min(scale, math.sqrt(pixel_budget / total_pixels))
    if max_pixels > MAX_IMAGE_PIXELS:
        scale = min(scale, math.sqrt(MAX_IMAGE_PIXELS / max_pixels))

    if scale < 1:
        resized_images = []
        for image in opened_images:
            target_size = (
                max(1, math.floor(image.width * scale)),
                max(1, math.floor(image.height * scale)),
            )
            resized_images.append(image.resize(target_size, Image.Resampling.LANCZOS))
            image.close()
        opened_images = resized_images

    content: list = [retrieval_text, *opened_images]
    return content, opened_images


def image_pixels(image_path: str, project_root: Path = PROJECT_ROOT) -> int:
    with Image.open(resolve_image_path(image_path, project_root)) as image:
        return image.width * image.height


def estimate_input_tokens(chunk: dict, project_root: Path = PROJECT_ROOT) -> int:
    retrieval_text = chunk.get("retrieval_text") or ""
    text_tokens = chunk.get("token_count") or max(1, math.ceil(len(retrieval_text) / 3))
    total_pixels = sum(
        image_pixels(image_path, project_root)
        for image_path in chunk.get("image_paths") or []
    )
    max_image_tokens = max(0, MAX_INPUT_TOKENS - INPUT_TOKEN_RESERVE - text_tokens)
    image_tokens = min(
        max_image_tokens,
        math.ceil(total_pixels / IMAGE_PIXELS_PER_TOKEN),
    )
    return min(MAX_INPUT_TOKENS - INPUT_TOKEN_RESERVE, text_tokens + image_tokens)


def select_safe_batch(
    chunks: list[dict],
    max_inputs: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = MAX_BATCH_TOKENS,
    project_root: Path = PROJECT_ROOT,
) -> list[dict]:
    selected = []
    total_tokens = 0
    for chunk in chunks:
        estimated_tokens = estimate_input_tokens(chunk, project_root)
        if selected and (
            len(selected) >= max_inputs
            or total_tokens + estimated_tokens > max_tokens
        ):
            break
        selected.append(chunk)
        total_tokens += estimated_tokens
    return selected


def create_client() -> voyageai.Client:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is not configured")
    return voyageai.Client(api_key=api_key, max_retries=3, timeout=120)


def embed_query(
    client: voyageai.Client,
    query: str,
    model: str = DEFAULT_MODEL,
    dimension: int = DEFAULT_DIMENSION,
) -> list[float]:
    value = query.strip()
    if not value:
        raise ValueError("Query cannot be empty")
    result = client.multimodal_embed(
        inputs=[[value]],
        model=model,
        input_type="query",
        truncation=False,
        output_dtype="float",
        output_dimension=dimension,
    )
    embedding = result.embeddings[0]
    if len(embedding) != dimension:
        raise RuntimeError(
            f"Voyage returned {len(embedding)} query dimensions; expected {dimension}"
        )
    return embedding


def embed_chunk_batch(
    client: voyageai.Client,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    dimension: int = DEFAULT_DIMENSION,
    project_root: Path = PROJECT_ROOT,
) -> list[dict]:
    inputs = []
    opened_images: list[Image.Image] = []
    try:
        for chunk in chunks:
            content, images = prepare_multimodal_input(chunk, project_root)
            inputs.append(content)
            opened_images.extend(images)

        result = client.multimodal_embed(
            inputs=inputs,
            model=model,
            input_type="document",
            truncation=False,
            output_dtype="float",
            output_dimension=dimension,
        )
        if len(result.embeddings) != len(chunks):
            raise RuntimeError("Voyage returned an unexpected embedding count")

        records = []
        for chunk, embedding in zip(chunks, result.embeddings):
            if len(embedding) != dimension:
                raise RuntimeError(
                    f"Voyage returned {len(embedding)} dimensions; expected {dimension}"
                )
            records.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "embedding": embedding,
                    "dense_model": model,
                    "dense_dimension": dimension,
                }
            )
        return records
    finally:
        for image in opened_images:
            image.close()
