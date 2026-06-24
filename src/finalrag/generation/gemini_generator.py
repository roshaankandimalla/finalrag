import base64
import mimetypes
import os
from pathlib import Path

from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-3.1-flash-lite"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAX_GENERATION_IMAGES = 8

SYSTEM_INSTRUCTION = """You are the grounded answer component of a RAG system.
Answer the user's question using only the retrieved evidence.
Every factual claim must cite one or more evidence labels such as [1] or [2].
Never invent citations or use labels that are not present in the evidence.
If the evidence is incomplete or conflicting, clearly say so.
Keep the answer direct and readable."""


def create_client(
    api_key: str | None = None,
    timeout_seconds: float = 340,
) -> genai.Client:
    selected_key = api_key or os.getenv("GEMINI_API_KEY")
    if not selected_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return genai.Client(
        api_key=selected_key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1_000)),
    )


def resolve_generation_image(image_path: str) -> Path:
    root = PROJECT_ROOT.resolve()
    resolved = (root / image_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Image path leaves project directory: {image_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing generation image: {image_path}")
    return resolved


def image_to_base64_part(image_path: str) -> types.Part:
    resolved = resolve_generation_image(image_path)
    mime_type = mimetypes.guess_type(resolved.name)[0] or "image/png"
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return types.Part.from_bytes(
        data=base64.b64decode(encoded),
        mime_type=mime_type,
    )


def build_generation_contents(
    prompt: str,
    image_paths: list[str] | None = None,
    max_images: int = DEFAULT_MAX_GENERATION_IMAGES,
) -> list[types.Content]:
    parts = [types.Part.from_text(text=prompt)]
    for image_path in (image_paths or [])[:max_images]:
        parts.append(image_to_base64_part(image_path))
    return [types.Content(role="user", parts=parts)]


def generate_grounded_answer(
    prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 340,
    image_paths: list[str] | None = None,
    max_images: int = DEFAULT_MAX_GENERATION_IMAGES,
) -> str:
    selected_model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    client = create_client(api_key=api_key, timeout_seconds=timeout_seconds)
    response = client.models.generate_content(
        model=selected_model,
        contents=build_generation_contents(
            prompt,
            image_paths=image_paths,
            max_images=max_images,
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=1_500,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty answer")
    return response.text.strip()
