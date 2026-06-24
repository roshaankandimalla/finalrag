import argparse
import json
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from finalrag.chunking.hierarchical_chunker import (
    create_all_chunks,
    retrieval_text,
    stable_id,
    token_count,
)
from finalrag.database.connection import connect_database
from finalrag.database.repository import upsert_child_chunks
from finalrag.embeddings.voyage_embeddings import embedding_input_hash


HOSPITAL_PROFILE_CHUNK_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "medical"
    / "csv"
    / "HCAHPS-Hospital.hospital_profiles.children.jsonl"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create parent/child chunks for parsed documents and hospital "
            "profile retrieval chunks for CSV profiles."
        )
    )
    parser.add_argument(
        "--skip-hospital-profiles",
        action="store_true",
        help="Create normal chunks only; skip hospital profile chunks.",
    )
    parser.add_argument(
        "--write-only-hospital-profiles",
        action="store_true",
        help=(
            "Write hospital profile chunks locally without storing those "
            "profile chunks in PostgreSQL."
        ),
    )
    return parser.parse_args()


def fetch_profile_chunks(connection: psycopg.Connection) -> list[dict]:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                profile.profile_id,
                profile.document_id,
                profile.facility_id,
                profile.hospital_name,
                profile.retrieval_text AS profile_text,
                profile.metadata AS profile_metadata,
                document.domain,
                document.file_name
            FROM hospital_profiles AS profile
            JOIN documents AS document
              ON document.document_id = profile.document_id
            ORDER BY document.domain, document.file_name, profile.facility_id;
            """
        )
        rows = cursor.fetchall()

    chunks = []
    for row in rows:
        source_name = Path(row["file_name"]).stem
        profile_text = (row["profile_text"] or "").strip()
        chunk = {
            "chunk_id": stable_id(str(row["profile_id"]), "hospital_profile_chunk"),
            "retrieval_type": "hospital_profile",
            "parent_id": str(row["profile_id"]),
            "document_id": str(row["document_id"]),
            "domain": row["domain"],
            "source_type": "csv",
            "source_name": source_name,
            "section_title": row["hospital_name"],
            "section_path": [row["hospital_name"], "profile"],
            "page_numbers": [],
            "source_urls": [],
            "source_row_numbers": (row["profile_metadata"] or {}).get(
                "source_row_numbers", []
            ),
            "modalities": ["text"],
            "text_content": profile_text,
            "table_markdown": None,
            "table_html": None,
            "image_paths": [],
            "element_ids": [],
            "metadata": {
                **(row["profile_metadata"] or {}),
                "profile_id": str(row["profile_id"]),
                "facility_id": row["facility_id"],
                "hospital_name": row["hospital_name"],
                "already_chunked": True,
            },
        }
        chunk["retrieval_text"] = retrieval_text(chunk)
        chunk["token_count"] = token_count(chunk["retrieval_text"])
        chunk["embedding_input_hash"] = embedding_input_hash(chunk, PROJECT_ROOT)
        chunks.append(chunk)

    return chunks


def write_jsonl(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def create_hospital_profile_chunks(write_only: bool) -> int:
    with connect_database() as connection:
        chunks = fetch_profile_chunks(connection)
        write_jsonl(HOSPITAL_PROFILE_CHUNK_PATH, chunks)

        print(f"\nHospital profile chunks: {len(chunks):,}")
        print(f"Output: {HOSPITAL_PROFILE_CHUNK_PATH}")

        if write_only:
            return len(chunks)

        for start in range(0, len(chunks), 250):
            upsert_child_chunks(connection, chunks[start : start + 250], compact=True)
            connection.commit()
            print(f"Stored profile chunks: {min(start + 250, len(chunks)):,}")

    return len(chunks)


def main() -> None:
    args = parse_arguments()

    reports = create_all_chunks()
    print(f"\nChunked sources: {len(reports)}")
    for report in reports:
        print(
            f"{report['domain']:<8} {report['source_type']:<4} "
            f"{report['source_name']}: parents={report['parent_count']:,} "
            f"children={report['child_count']:,} "
            f"duplicates={report['duplicate_chunk_count']} "
            f"image_only={len(report['image_only_chunks'])}"
        )

    if args.skip_hospital_profiles:
        print("\nSkipped hospital profile chunks.")
        return

    create_hospital_profile_chunks(
        write_only=args.write_only_hospital_profiles,
    )


if __name__ == "__main__":
    main()
