import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


SUPPORTED_LOCAL_FILES = {
    ".pdf": "llamaparse",
    ".csv": "pandas",
}


@dataclass
class DiscoveredDocument:
    document_id: uuid.UUID
    domain: str
    file_name: str
    source_type: str
    file_path: str
    parser_used: str
    metadata: dict


def create_document_id(
    domain: str,
    source_type: str,
    source_location: str,
) -> uuid.UUID:
    identity = f"{domain}:{source_type}:{source_location.lower()}"
    return uuid.uuid5(uuid.NAMESPACE_URL, identity)


def create_url_name(url: str) -> str:
    parsed = urlparse(url)

    path_name = Path(parsed.path).stem
    if path_name and path_name not in {"drugInfo", "lookup"}:
        return path_name

    return parsed.netloc.replace(".", "_")


def discover_local_files(
    domain_dir: Path,
    project_root: Path,
) -> list[DiscoveredDocument]:
    documents = []

    for file_path in domain_dir.iterdir():
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_LOCAL_FILES:
            continue

        source_type = suffix.removeprefix(".")
        relative_path = file_path.relative_to(project_root).as_posix()

        documents.append(
            DiscoveredDocument(
                document_id=create_document_id(
                    domain_dir.name,
                    source_type,
                    relative_path,
                ),
                domain=domain_dir.name,
                file_name=file_path.name,
                source_type=source_type,
                file_path=relative_path,
                parser_used=SUPPORTED_LOCAL_FILES[suffix],
                metadata={
                    "file_size_bytes": file_path.stat().st_size,
                },
            )
        )

    return documents


def discover_html_sources(
    domain_dir: Path,
) -> list[DiscoveredDocument]:
    documents = []

    for manifest_path in sorted(domain_dir.glob("*.json")):
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = raw_manifest if isinstance(raw_manifest, list) else [raw_manifest]

        for source in sources:
            firecrawl_config = source.get("firecrawl_config", source)
            url = firecrawl_config.get("url")

            # Skip JSON files that are not Firecrawl source manifests.
            if not url:
                continue

            name = source.get("name") or manifest_path.stem

            documents.append(
                DiscoveredDocument(
                    document_id=create_document_id(
                        domain_dir.name,
                        "html",
                        url,
                    ),
                    domain=domain_dir.name,
                    file_name=name,
                    source_type="html",
                    file_path=url,
                    parser_used="firecrawl",
                    metadata={
                        "manifest_path": str(
                            manifest_path.relative_to(domain_dir.parent.parent)
                        ),
                        "firecrawl_mode": source.get("firecrawl_mode", "crawl"),
                        "firecrawl_config": firecrawl_config,
                    },
                )
            )

    return documents

def discover_documents(
    input_dir: Path,
    project_root: Path,
) -> list[DiscoveredDocument]:
    documents = []

    for domain_dir in sorted(input_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        documents.extend(discover_local_files(domain_dir, project_root))
        documents.extend(discover_html_sources(domain_dir))

    return documents
