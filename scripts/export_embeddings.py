import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finalrag.database.connection import connect_database
from finalrag.embeddings.splade_embeddings import DEFAULT_MODEL as DEFAULT_SPLADE_MODEL
from finalrag.embeddings.voyage_embeddings import (
    DEFAULT_DIMENSION as DEFAULT_DENSE_DIMENSION,
    DEFAULT_MODEL as DEFAULT_DENSE_MODEL,
)


OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"
DENSE_PATH = OUTPUT_DIR / "dense_embeddings.npy"
MANIFEST_PATH = OUTPUT_DIR / "dense_manifest.jsonl"
SPLADE_PATH = OUTPUT_DIR / "splade_vectors.jsonl"


def current_chunk_ids() -> set[str]:
    ids = set()
    for path in sorted((PROJECT_ROOT / "data" / "chunks").glob("*/*/*.children.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                ids.add(str(json.loads(line)["chunk_id"]))
    return ids


def sparse_record(sparse_embedding) -> dict:
    return {
        "indices": [int(index) for index in sparse_embedding.indices()],
        "values": [float(value) for value in sparse_embedding.values()],
        "dimension": int(sparse_embedding.dimensions()),
    }


def replace_tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    dense_model = os.getenv("VOYAGE_EMBED_MODEL", DEFAULT_DENSE_MODEL)
    dense_dimension = int(
        os.getenv("VOYAGE_EMBED_DIMENSION", str(DEFAULT_DENSE_DIMENSION))
    )
    sparse_model = os.getenv("SPLADE_MODEL", DEFAULT_SPLADE_MODEL)

    chunk_ids = current_chunk_ids()
    if not chunk_ids:
        raise RuntimeError("No current child chunks found in data/chunks")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dense_tmp = replace_tmp(DENSE_PATH)
    manifest_tmp = replace_tmp(MANIFEST_PATH)
    splade_tmp = replace_tmp(SPLADE_PATH)

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (
                        WHERE dense_embedding IS NOT NULL
                          AND dense_model = %s
                          AND dense_dimension = %s
                    ) AS dense_ready,
                    COUNT(*) FILTER (
                        WHERE sparse_embedding IS NOT NULL
                          AND sparse_model = %s
                    ) AS sparse_ready
                FROM child_chunks
                WHERE chunk_id::text = ANY(%s);
                """,
                (dense_model, dense_dimension, sparse_model, list(chunk_ids)),
            )
            total, dense_ready, sparse_ready = cursor.fetchone()

            if total != len(chunk_ids):
                raise RuntimeError(
                    f"PostgreSQL has {total:,} current chunks, "
                    f"but data/chunks has {len(chunk_ids):,}"
                )
            if dense_ready != total or sparse_ready != total:
                raise RuntimeError(
                    "Cannot export incomplete embeddings: "
                    f"dense={dense_ready:,}/{total:,}, "
                    f"sparse={sparse_ready:,}/{total:,}"
                )

            dense_matrix = np.lib.format.open_memmap(
                dense_tmp,
                mode="w+",
                dtype=np.float32,
                shape=(total, dense_dimension),
            )

            cursor.execute(
                """
                SELECT
                    chunk_id::text,
                    document_id::text,
                    domain,
                    source_type,
                    file_name,
                    section_title,
                    page_numbers,
                    source_urls,
                    source_row_numbers,
                    modalities,
                    token_count,
                    embedding_input_hash,
                    dense_embedding,
                    dense_model,
                    dense_dimension,
                    sparse_embedding,
                    sparse_model,
                    sparse_dimension
                FROM child_chunks
                WHERE chunk_id::text = ANY(%s)
                ORDER BY domain, source_type, file_name, chunk_id;
                """,
                (list(chunk_ids),),
            )

            exported = 0
            with manifest_tmp.open("w", encoding="utf-8") as manifest, splade_tmp.open(
                "w",
                encoding="utf-8",
            ) as splade:
                while True:
                    rows = cursor.fetchmany(500)
                    if not rows:
                        break
                    for row in rows:
                        (
                            chunk_id,
                            document_id,
                            domain,
                            source_type,
                            file_name,
                            section_title,
                            page_numbers,
                            source_urls,
                            source_row_numbers,
                            modalities,
                            token_count,
                            embedding_input_hash,
                            dense_embedding,
                            row_dense_model,
                            row_dense_dimension,
                            sparse_embedding,
                            row_sparse_model,
                            row_sparse_dimension,
                        ) = row

                        dense_matrix[exported] = np.asarray(
                            dense_embedding,
                            dtype=np.float32,
                        )

                        manifest.write(
                            json.dumps(
                                {
                                    "row_index": exported,
                                    "chunk_id": chunk_id,
                                    "document_id": document_id,
                                    "domain": domain,
                                    "source_type": source_type,
                                    "file_name": file_name,
                                    "section_title": section_title,
                                    "page_numbers": page_numbers or [],
                                    "source_urls": source_urls or [],
                                    "source_row_numbers": source_row_numbers or [],
                                    "modalities": modalities or [],
                                    "token_count": token_count,
                                    "embedding_input_hash": embedding_input_hash,
                                    "dense_model": row_dense_model,
                                    "dense_dimension": row_dense_dimension,
                                },
                                ensure_ascii=False,
                            )
                        )
                        manifest.write("\n")

                        sparse_payload = sparse_record(sparse_embedding)
                        sparse_payload.update(
                            {
                                "chunk_id": chunk_id,
                                "model": row_sparse_model,
                                "dimension": row_sparse_dimension,
                            }
                        )
                        splade.write(json.dumps(sparse_payload, ensure_ascii=False))
                        splade.write("\n")

                        exported += 1
                        if exported % 5_000 == 0:
                            print(f"Exported: {exported:,}/{total:,}")

            dense_matrix.flush()
            del dense_matrix

    dense_tmp.replace(DENSE_PATH)
    manifest_tmp.replace(MANIFEST_PATH)
    splade_tmp.replace(SPLADE_PATH)

    print(f"Exported embeddings: {exported:,}")
    print(f"Dense matrix: {DENSE_PATH}")
    print(f"Dense manifest: {MANIFEST_PATH}")
    print(f"SPLADE vectors: {SPLADE_PATH}")


if __name__ == "__main__":
    main()
