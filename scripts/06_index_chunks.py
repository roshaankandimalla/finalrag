import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pgvector import SparseVector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from finalrag.database.connection import connect_database
from finalrag.database.repository import (
    create_dense_hnsw_index,
    create_sparse_hnsw_index,
    dense_indexing_counts,
    fetch_chunks_needing_dense_embeddings,
    fetch_chunks_needing_sparse_embeddings,
    mark_chunked_documents,
    mark_dense_indexed_documents,
    sparse_indexing_counts,
    update_dense_embeddings,
    update_sparse_embeddings,
    upsert_child_chunks,
    upsert_parent_sections,
)
from finalrag.embeddings.voyage_embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    create_client,
    embed_chunk_batch,
    embedding_input_hash,
    select_safe_batch,
)
from finalrag.embeddings.splade_embeddings import (
    DEFAULT_BATCH_SIZE as DEFAULT_SPLADE_BATCH_SIZE,
    DEFAULT_MAX_LENGTH as DEFAULT_SPLADE_MAX_LENGTH,
    DEFAULT_MODEL as DEFAULT_SPLADE_MODEL,
    DEFAULT_TOP_K as DEFAULT_SPLADE_TOP_K,
    SpladeEncoder,
)
from finalrag.retrieval.domain_router import rebuild_domain_centroids


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Store chunks, create dense/SPLADE embeddings, build HNSW indexes, "
            "and rebuild domain centroids."
        )
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Call Voyage and store dense embeddings. This consumes API credits.",
    )
    parser.add_argument(
        "--splade",
        action="store_true",
        help=(
            "Create SPLADE sparse embeddings locally and store them. If "
            "--import-splade-jsonl is also used, import runs first and SPLADE "
            "CPU encoding fills only remaining pending chunks."
        ),
    )
    parser.add_argument(
        "--import-splade-jsonl",
        type=Path,
        help="Import Colab-generated SPLADE JSONL instead of encoding locally.",
    )
    parser.add_argument(
        "--domain-centroids",
        action="store_true",
        help="Rebuild domain centroids after dense embeddings are available.",
    )
    parser.add_argument(
        "--skip-store",
        action="store_true",
        help="Skip parent/child upserts and resume dense embedding only.",
    )
    parser.add_argument(
        "--full-storage",
        action="store_true",
        help=(
            "Also store duplicate text_content and table_markdown payloads "
            "in PostgreSQL."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of child chunks to embed during this run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum Voyage inputs per request. Defaults to VOYAGE_EMBED_BATCH_SIZE or 64.",
    )
    parser.add_argument(
        "--splade-limit",
        type=int,
        help="Maximum number of chunks to SPLADE-embed during this run.",
    )
    parser.add_argument(
        "--splade-batch-size",
        type=int,
        default=None,
        help="Maximum SPLADE inputs per local batch.",
    )
    parser.add_argument(
        "--build-sparse-index",
        action="store_true",
        help="Build sparse HNSW index even when sparse chunks remain pending.",
    )
    parser.add_argument(
        "--skip-sparse-index",
        action="store_true",
        help="Skip sparse HNSW rebuild after importing SPLADE JSONL.",
    )
    return parser.parse_args()


def is_default_full_index_run(args: argparse.Namespace) -> bool:
    return not any(
        [
            args.embed,
            args.splade,
            args.import_splade_jsonl,
            args.domain_centroids,
            args.skip_store,
            args.full_storage,
            args.limit is not None,
            args.batch_size is not None,
            args.splade_limit is not None,
            args.splade_batch_size is not None,
            args.build_sparse_index,
            args.skip_sparse_index,
        ]
    )


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def batched(values: list[dict], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_local_chunks() -> tuple[list[dict], list[dict]]:
    parents = []
    children = []
    for path in sorted((PROJECT_ROOT / "data" / "chunks").glob("*/*/*.parents.jsonl")):
        parents.extend(read_jsonl(path))
    for path in sorted((PROJECT_ROOT / "data" / "chunks").glob("*/*/*.children.jsonl")):
        for chunk in read_jsonl(path):
            chunk["embedding_input_hash"] = embedding_input_hash(chunk, PROJECT_ROOT)
            children.append(chunk)
    return parents, children


def store_chunks(connection, compact: bool) -> tuple[int, int]:
    parents, children = load_local_chunks()
    for batch in batched(parents, 500):
        upsert_parent_sections(connection, batch)
        connection.commit()
    for batch in batched(children, 250):
        upsert_child_chunks(connection, batch, compact=compact)
        connection.commit()
    mark_chunked_documents(connection)
    connection.commit()
    return len(parents), len(children)


def embed_pending_chunks(
    connection,
    model: str,
    dimension: int,
    batch_size: int,
    limit: int | None,
) -> int:
    client = create_client()
    embedded = 0

    while limit is None or embedded < limit:
        candidate_limit = max(batch_size * 2, batch_size)
        if limit is not None:
            candidate_limit = min(candidate_limit, limit - embedded)
        candidates = fetch_chunks_needing_dense_embeddings(
            connection,
            model=model,
            dimension=dimension,
            limit=candidate_limit,
        )
        if not candidates:
            break
        chunks = select_safe_batch(
            candidates,
            max_inputs=batch_size,
            project_root=PROJECT_ROOT,
        )

        records = embed_chunk_batch(
            client,
            chunks,
            model=model,
            dimension=dimension,
            project_root=PROJECT_ROOT,
        )
        update_dense_embeddings(connection, records)
        connection.commit()
        embedded += len(records)
        print(f"Embedded and committed: {embedded:,}")

    mark_dense_indexed_documents(connection, model, dimension)
    connection.commit()
    counts = dense_indexing_counts(connection, model, dimension)
    if counts["pending"] == 0:
        print("Building dense HNSW index")
        create_dense_hnsw_index(connection)
        connection.commit()
    return embedded


def embed_pending_splade_chunks(
    connection,
    model_name: str,
    batch_size: int,
    max_length: int,
    top_k: int,
    limit: int | None,
    build_index: bool,
) -> int:
    print(f"Loading SPLADE model: {model_name}")
    encoder = SpladeEncoder(model_name, max_length=max_length, top_k=top_k)
    print(
        f"Device: {encoder.device} | dimension={encoder.dimension:,} | "
        f"top_k={top_k:,}"
    )
    if encoder.dimension != 30522:
        raise RuntimeError(
            f"schema.sql uses SPARSEVEC(30522), but {model_name} has "
            f"{encoder.dimension} vocabulary dimensions"
        )

    indexed = 0
    while limit is None or indexed < limit:
        current_size = batch_size
        if limit is not None:
            current_size = min(current_size, limit - indexed)
        chunks = fetch_chunks_needing_sparse_embeddings(
            connection,
            model=model_name,
            dimension=encoder.dimension,
            limit=current_size,
        )
        if not chunks:
            break

        vectors = encoder.encode_documents(
            [(chunk["retrieval_text"] or "").strip() for chunk in chunks]
        )
        records = [
            {
                "chunk_id": chunk["chunk_id"],
                "sparse_embedding": vector,
                "sparse_model": model_name,
                "sparse_dimension": encoder.dimension,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        update_sparse_embeddings(connection, records)
        connection.commit()
        indexed += len(records)
        print(f"SPLADE embedded and committed: {indexed:,}")

    counts = sparse_indexing_counts(connection, model_name, encoder.dimension)
    print(
        f"Sparse index: completed={counts['completed']:,} "
        f"pending={counts['pending']:,} total={counts['total']:,}"
    )
    if counts["pending"] == 0 or build_index:
        print("Building sparse HNSW index")
        create_sparse_hnsw_index(connection)
        connection.commit()
    return indexed


def splade_jsonl_records(path: Path):
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"chunk_id", "indices", "values", "dimension", "model"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Line {line_number} missing: {sorted(missing)}")
            if record["chunk_id"] in seen:
                continue
            if len(record["indices"]) != len(record["values"]):
                raise ValueError(f"Line {line_number} has mismatched indices/values")
            seen.add(record["chunk_id"])
            yield {
                "chunk_id": record["chunk_id"],
                "sparse_embedding": SparseVector(
                    dict(zip(record["indices"], record["values"])),
                    record["dimension"],
                ),
                "sparse_model": record["model"],
                "sparse_dimension": record["dimension"],
            }


def import_splade_jsonl(
    connection,
    path: Path,
    batch_size: int,
    rebuild_index: bool,
) -> int:
    if batch_size < 1:
        raise ValueError("splade-batch-size must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)

    imported = 0
    batch = []
    for record in splade_jsonl_records(path):
        batch.append(record)
        if len(batch) < batch_size:
            continue
        update_sparse_embeddings(connection, batch)
        connection.commit()
        imported += len(batch)
        batch.clear()
        print(f"Imported and committed: {imported:,}")

    if batch:
        update_sparse_embeddings(connection, batch)
        connection.commit()
        imported += len(batch)
        print(f"Imported and committed: {imported:,}")

    if rebuild_index:
        print("Building sparse HNSW index")
        create_sparse_hnsw_index(connection)
        connection.commit()
    return imported


def prompt_splade_import_path() -> Path | None:
    print("\nSPLADE import")
    print("Enter a SPLADE JSONL file path to import first.")
    print("Press Enter or type 'no' to skip import and create SPLADE locally on CPU.")
    try:
        value = input("SPLADE JSONL path [no]: ").strip().strip('"')
    except EOFError:
        value = ""

    if not value or value.lower() in {"n", "no", "none", "skip"}:
        return None

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    args = arguments()
    default_full_index = is_default_full_index_run(args)

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    model = os.getenv("VOYAGE_EMBED_MODEL", DEFAULT_MODEL)
    dimension = int(os.getenv("VOYAGE_EMBED_DIMENSION", str(DEFAULT_DIMENSION)))
    batch_size = args.batch_size or int(
        os.getenv("VOYAGE_EMBED_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))
    )
    splade_model = os.getenv("SPLADE_MODEL", DEFAULT_SPLADE_MODEL)
    splade_batch_size = args.splade_batch_size or int(
        os.getenv("SPLADE_BATCH_SIZE", str(DEFAULT_SPLADE_BATCH_SIZE))
    )
    splade_max_length = int(
        os.getenv("SPLADE_MAX_LENGTH", str(DEFAULT_SPLADE_MAX_LENGTH))
    )
    splade_top_k = int(os.getenv("SPLADE_TOP_K", str(DEFAULT_SPLADE_TOP_K)))
    if dimension != 2048:
        raise RuntimeError(
            "The current PostgreSQL schema uses VECTOR(2048); "
            "set VOYAGE_EMBED_DIMENSION=2048"
        )
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if splade_batch_size < 1:
        raise ValueError("splade-batch-size must be positive")

    with connect_database() as connection:
        if not args.skip_store:
            parents, children = store_chunks(
                connection,
                compact=not args.full_storage,
            )
            print(f"Stored parents: {parents:,}")
            print(f"Stored children: {children:,}")

        counts = dense_indexing_counts(connection, model, dimension)
        print(
            f"Dense index: completed={counts['completed']:,} "
            f"pending={counts['pending']:,} total={counts['total']:,}"
        )

        if args.embed or default_full_index:
            embedded = embed_pending_chunks(
                connection,
                model=model,
                dimension=dimension,
                batch_size=batch_size,
                limit=args.limit,
            )
            print(f"Embedded this run: {embedded:,}")
            counts = dense_indexing_counts(connection, model, dimension)
            print(
                f"Dense index now: completed={counts['completed']:,} "
                f"pending={counts['pending']:,} total={counts['total']:,}"
            )
        else:
            print("Voyage was not called. Add --embed when ready to spend API credits.")

        import_splade_path = args.import_splade_jsonl
        if default_full_index:
            count = rebuild_domain_centroids(connection, model, dimension)
            connection.commit()
            print(f"Rebuilt domain centroids: {count}")
            import_splade_path = prompt_splade_import_path()

        if import_splade_path:
            # Import Colab/GPU vectors first. When --splade is also set, delay
            # index creation until after the local CPU pass fills any gaps.
            imported = import_splade_jsonl(
                connection,
                path=import_splade_path,
                batch_size=splade_batch_size,
                rebuild_index=(
                    not args.skip_sparse_index
                    and not args.splade
                    and not default_full_index
                ),
            )
            print(f"SPLADE imported this run: {imported:,}")

        if args.splade or default_full_index:
            indexed = embed_pending_splade_chunks(
                connection,
                model_name=splade_model,
                batch_size=splade_batch_size,
                max_length=splade_max_length,
                top_k=splade_top_k,
                limit=args.splade_limit,
                build_index=args.build_sparse_index or default_full_index,
            )
            print(f"SPLADE embedded this run: {indexed:,}")

        if (
            not args.splade
            and not import_splade_path
            and not args.domain_centroids
            and not default_full_index
        ):
            sparse_counts = sparse_indexing_counts(
                connection,
                splade_model,
                30522,
            )
            print(
                f"Sparse index: completed={sparse_counts['completed']:,} "
                f"pending={sparse_counts['pending']:,} "
                f"total={sparse_counts['total']:,}"
            )
            print("SPLADE was not called. Add --splade or --import-splade-jsonl.")

        if args.domain_centroids:
            # Domain centroids are dense-only. They are averages of Voyage
            # embeddings per domain/retrieval_type and do not use SPLADE.
            count = rebuild_domain_centroids(connection, model, dimension)
            connection.commit()
            print(f"Rebuilt domain centroids: {count}")


if __name__ == "__main__":
    main()
