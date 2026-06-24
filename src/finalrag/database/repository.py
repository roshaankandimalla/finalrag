from importlib.metadata import PackageNotFoundError, version

import psycopg
from pgvector import SparseVector, Vector
from psycopg.types.json import Jsonb

from finalrag.discovery.file_discovery import DiscoveredDocument


PARSER_PACKAGES = {
    "llamaparse": "llama-cloud",
    "firecrawl": "firecrawl-py",
    "pandas": "pandas",
}


def get_parser_version(parser_used: str) -> str | None:
    package = PARSER_PACKAGES.get(parser_used)

    if not package:
        return None

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def upsert_document(
    connection: psycopg.Connection,
    document: DiscoveredDocument,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (
                document_id,
                domain,
                file_name,
                source_type,
                file_path,
                parser_used,
                parser_version,
                status,
                metadata
            )
            VALUES (
                %(document_id)s,
                %(domain)s,
                %(file_name)s,
                %(source_type)s,
                %(file_path)s,
                %(parser_used)s,
                %(parser_version)s,
                'discovered',
                %(metadata)s
            )
            ON CONFLICT (document_id) DO UPDATE SET
                domain = EXCLUDED.domain,
                file_name = EXCLUDED.file_name,
                source_type = EXCLUDED.source_type,
                file_path = EXCLUDED.file_path,
                parser_used = EXCLUDED.parser_used,
                parser_version = EXCLUDED.parser_version,
                metadata = EXCLUDED.metadata,
                updated_at = NOW();
            """,
            {
                "document_id": document.document_id,
                "domain": document.domain,
                "file_name": document.file_name,
                "source_type": document.source_type,
                "file_path": document.file_path,
                "parser_used": document.parser_used,
                "parser_version": get_parser_version(document.parser_used),
                "metadata": Jsonb(document.metadata),
            },
        )

def fetch_documents_for_parsing(
    connection: psycopg.Connection,
    source_type: str | None = None,
) -> list[dict]:
    query = """
        SELECT
            document_id,
            domain,
            file_name,
            source_type,
            file_path,
            parser_used,
            parser_version,
            status,
            metadata
        FROM documents
        WHERE status IN ('discovered', 'parsing', 'failed')
    """

    params = []

    if source_type:
        query += " AND source_type = %s"
        params.append(source_type)

    query += " ORDER BY domain, file_name;"

    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()


def update_document_status(
    connection: psycopg.Connection,
    document_id,
    status: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET status = %s,
                updated_at = NOW()
            WHERE document_id = %s;
            """,
            (status, document_id),
        )


def save_llamaparse_job(
    connection: psycopg.Connection,
    document_id,
    job_id: str,
    api_key_alias: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET metadata = jsonb_set(
                    metadata,
                    '{llamaparse}',
                    %s,
                    true
                ),
                updated_at = NOW()
            WHERE document_id = %s;
            """,
            (
                Jsonb(
                    {
                        "job_id": job_id,
                        "api_key_alias": api_key_alias,
                        "tier": "agentic",
                        "version": "latest",
                    }
                ),
                document_id,
            ),
        )


def mark_document_parsed(
    connection: psycopg.Connection,
    document_id,
    outputs: dict,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET status = 'parsed',
                metadata = jsonb_set(
                    metadata,
                    '{parse_outputs}',
                    %s,
                    true
                ),
                updated_at = NOW()
            WHERE document_id = %s;
            """,
            (Jsonb(outputs), document_id),
        )


def mark_document_failed(
    connection: psycopg.Connection,
    document_id,
    error: Exception | str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET status = 'failed',
                metadata = jsonb_set(
                    metadata,
                    '{parse_error}',
                    %s,
                    true
                ),
                updated_at = NOW()
            WHERE document_id = %s;
            """,
            (Jsonb(str(error)[:2000]), document_id),
        )


def clear_csv_data(
    connection: psycopg.Connection,
    document_id,
) -> None:
    """Remove derived CSV rows before a clean, repeatable re-import."""
    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM hcahps_records WHERE document_id = %s;",
            (document_id,),
        )
        cursor.execute(
            "DELETE FROM hospital_profiles WHERE document_id = %s;",
            (document_id,),
        )


def insert_hcahps_records(
    connection: psycopg.Connection,
    records: list[dict],
) -> None:
    if not records:
        return

    columns = [
        "document_id",
        "source_row_number",
        "facility_id",
        "facility_name",
        "address",
        "city",
        "state",
        "zip_code",
        "county",
        "telephone",
        "measure_id",
        "question",
        "answer_description",
        "star_rating",
        "star_rating_footnote",
        "answer_percent",
        "answer_percent_footnote",
        "linear_mean_value",
        "completed_surveys",
        "completed_surveys_footnote",
        "response_rate_percent",
        "response_rate_footnote",
        "survey_start_date",
        "survey_end_date",
    ]

    with connection.cursor() as cursor:
        with cursor.copy(
            """
            COPY hcahps_records (
                document_id,
                source_row_number,
                facility_id,
                facility_name,
                address,
                city,
                state,
                zip_code,
                county,
                telephone,
                measure_id,
                question,
                answer_description,
                star_rating,
                star_rating_footnote,
                answer_percent,
                answer_percent_footnote,
                linear_mean_value,
                completed_surveys,
                completed_surveys_footnote,
                response_rate_percent,
                response_rate_footnote,
                survey_start_date,
                survey_end_date
            )
            FROM STDIN
            """,
        ) as copy:
            for record in records:
                copy.write_row(tuple(record[column] for column in columns))


def upsert_hospital_profile(
    connection: psycopg.Connection,
    profile: dict,
) -> None:
    values = dict(profile)
    values["category_summaries"] = Jsonb(values["category_summaries"])
    values["metadata"] = Jsonb(values["metadata"])

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO hospital_profiles (
                profile_id,
                document_id,
                facility_id,
                hospital_name,
                address,
                city,
                state,
                zip_code,
                county,
                telephone,
                survey_start_date,
                survey_end_date,
                completed_surveys,
                response_rate_percent,
                category_summaries,
                retrieval_text,
                metadata
            )
            VALUES (
                %(profile_id)s,
                %(document_id)s,
                %(facility_id)s,
                %(hospital_name)s,
                %(address)s,
                %(city)s,
                %(state)s,
                %(zip_code)s,
                %(county)s,
                %(telephone)s,
                %(survey_start_date)s,
                %(survey_end_date)s,
                %(completed_surveys)s,
                %(response_rate_percent)s,
                %(category_summaries)s,
                %(retrieval_text)s,
                %(metadata)s
            )
            ON CONFLICT (document_id, facility_id) DO UPDATE SET
                hospital_name = EXCLUDED.hospital_name,
                address = EXCLUDED.address,
                city = EXCLUDED.city,
                state = EXCLUDED.state,
                zip_code = EXCLUDED.zip_code,
                county = EXCLUDED.county,
                telephone = EXCLUDED.telephone,
                survey_start_date = EXCLUDED.survey_start_date,
                survey_end_date = EXCLUDED.survey_end_date,
                completed_surveys = EXCLUDED.completed_surveys,
                response_rate_percent = EXCLUDED.response_rate_percent,
                category_summaries = EXCLUDED.category_summaries,
                retrieval_text = EXCLUDED.retrieval_text,
                metadata = EXCLUDED.metadata,
                updated_at = NOW();
            """,
            values,
        )


def upsert_hospital_category_docs(
    connection: psycopg.Connection,
    category_docs: list[dict],
) -> None:
    if not category_docs:
        return

    values = []
    for category_doc in category_docs:
        item = dict(category_doc)
        item["metadata"] = Jsonb(item["metadata"])
        values.append(item)

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO hospital_category_docs (
                category_doc_id,
                profile_id,
                document_id,
                facility_id,
                category,
                retrieval_text,
                table_markdown,
                measure_ids,
                source_row_numbers,
                metadata
            )
            VALUES (
                %(category_doc_id)s,
                %(profile_id)s,
                %(document_id)s,
                %(facility_id)s,
                %(category)s,
                %(retrieval_text)s,
                %(table_markdown)s,
                %(measure_ids)s,
                %(source_row_numbers)s,
                %(metadata)s
            )
            ON CONFLICT (document_id, facility_id, category) DO UPDATE SET
                retrieval_text = EXCLUDED.retrieval_text,
                table_markdown = EXCLUDED.table_markdown,
                measure_ids = EXCLUDED.measure_ids,
                source_row_numbers = EXCLUDED.source_row_numbers,
                metadata = EXCLUDED.metadata,
                updated_at = NOW();
            """,
            values,
        )


def upsert_parent_sections(
    connection: psycopg.Connection,
    parents: list[dict],
) -> None:
    if not parents:
        return

    values = [
        {
            "parent_id": parent["parent_id"],
            "document_id": parent["document_id"],
            "domain": parent["domain"],
            "source_type": parent["source_type"],
            "section_title": parent.get("section_title"),
            "section_path": parent.get("section_path") or [],
            "page_numbers": parent.get("page_numbers") or [],
            "row_ranges": Jsonb(parent.get("row_ranges"))
            if parent.get("row_ranges") is not None
            else None,
            "parent_text": parent.get("parent_text"),
            "token_count": parent.get("token_count"),
            "metadata": Jsonb(parent.get("metadata") or {}),
        }
        for parent in parents
    ]

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO parent_sections (
                parent_id,
                document_id,
                domain,
                source_type,
                section_title,
                section_path,
                page_numbers,
                row_ranges,
                parent_text,
                token_count,
                metadata
            )
            VALUES (
                %(parent_id)s,
                %(document_id)s,
                %(domain)s,
                %(source_type)s,
                %(section_title)s,
                %(section_path)s,
                %(page_numbers)s,
                %(row_ranges)s,
                %(parent_text)s,
                %(token_count)s,
                %(metadata)s
            )
            ON CONFLICT (parent_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                domain = EXCLUDED.domain,
                source_type = EXCLUDED.source_type,
                section_title = EXCLUDED.section_title,
                section_path = EXCLUDED.section_path,
                page_numbers = EXCLUDED.page_numbers,
                row_ranges = EXCLUDED.row_ranges,
                parent_text = EXCLUDED.parent_text,
                token_count = EXCLUDED.token_count,
                metadata = EXCLUDED.metadata,
                updated_at = NOW();
            """,
            values,
        )


def upsert_child_chunks(
    connection: psycopg.Connection,
    chunks: list[dict],
    compact: bool = True,
) -> None:
    if not chunks:
        return

    values = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        metadata["storage_mode"] = "compact" if compact else "full"
        values.append(
            {
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk.get("parent_id"),
                "document_id": chunk["document_id"],
                "domain": chunk["domain"],
                "source_type": chunk["source_type"],
                "file_name": chunk["source_name"],
                "retrieval_type": chunk.get("retrieval_type") or "child_chunk",
                "retrieval_text": chunk["retrieval_text"],
                "page_numbers": chunk.get("page_numbers") or [],
                "source_urls": chunk.get("source_urls") or [],
                "source_row_numbers": chunk.get("source_row_numbers") or [],
                "row_range": None,
                "section_title": chunk.get("section_title"),
                "section_path": chunk.get("section_path") or [],
                "modalities": chunk.get("modalities") or [],
                "text_content": None if compact else chunk.get("text_content"),
                "table_markdown": None if compact else chunk.get("table_markdown"),
                "table_html": chunk.get("table_html"),
                "image_paths": chunk.get("image_paths") or [],
                "token_count": chunk.get("token_count"),
                "embedding_input_hash": chunk["embedding_input_hash"],
                "metadata": Jsonb(metadata),
            }
        )

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO child_chunks (
                chunk_id,
                parent_id,
                document_id,
                domain,
                source_type,
                file_name,
                retrieval_type,
                retrieval_text,
                page_numbers,
                source_urls,
                source_row_numbers,
                row_range,
                section_title,
                section_path,
                modalities,
                text_content,
                table_markdown,
                table_html,
                image_paths,
                token_count,
                embedding_input_hash,
                metadata
            )
            VALUES (
                %(chunk_id)s,
                %(parent_id)s,
                %(document_id)s,
                %(domain)s,
                %(source_type)s,
                %(file_name)s,
                %(retrieval_type)s,
                %(retrieval_text)s,
                %(page_numbers)s,
                %(source_urls)s,
                %(source_row_numbers)s,
                %(row_range)s,
                %(section_title)s,
                %(section_path)s,
                %(modalities)s,
                %(text_content)s,
                %(table_markdown)s,
                %(table_html)s,
                %(image_paths)s,
                %(token_count)s,
                %(embedding_input_hash)s,
                %(metadata)s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                parent_id = EXCLUDED.parent_id,
                document_id = EXCLUDED.document_id,
                domain = EXCLUDED.domain,
                source_type = EXCLUDED.source_type,
                file_name = EXCLUDED.file_name,
                retrieval_type = EXCLUDED.retrieval_type,
                retrieval_text = EXCLUDED.retrieval_text,
                page_numbers = EXCLUDED.page_numbers,
                source_urls = EXCLUDED.source_urls,
                source_row_numbers = EXCLUDED.source_row_numbers,
                row_range = EXCLUDED.row_range,
                section_title = EXCLUDED.section_title,
                section_path = EXCLUDED.section_path,
                modalities = EXCLUDED.modalities,
                text_content = EXCLUDED.text_content,
                table_markdown = EXCLUDED.table_markdown,
                table_html = EXCLUDED.table_html,
                image_paths = EXCLUDED.image_paths,
                token_count = EXCLUDED.token_count,
                dense_embedding = CASE
                    WHEN child_chunks.embedding_input_hash IS DISTINCT FROM
                         EXCLUDED.embedding_input_hash
                    THEN NULL
                    ELSE child_chunks.dense_embedding
                END,
                dense_model = CASE
                    WHEN child_chunks.embedding_input_hash IS DISTINCT FROM
                         EXCLUDED.embedding_input_hash
                    THEN NULL
                    ELSE child_chunks.dense_model
                END,
                dense_dimension = CASE
                    WHEN child_chunks.embedding_input_hash IS DISTINCT FROM
                         EXCLUDED.embedding_input_hash
                    THEN NULL
                    ELSE child_chunks.dense_dimension
                END,
                dense_updated_at = CASE
                    WHEN child_chunks.embedding_input_hash IS DISTINCT FROM
                         EXCLUDED.embedding_input_hash
                    THEN NULL
                    ELSE child_chunks.dense_updated_at
                END,
                sparse_embedding = CASE
                    WHEN child_chunks.embedding_input_hash IS DISTINCT FROM
                         EXCLUDED.embedding_input_hash
                    THEN NULL
                    ELSE child_chunks.sparse_embedding
                END,
                sparse_model = CASE
                    WHEN child_chunks.embedding_input_hash IS DISTINCT FROM
                         EXCLUDED.embedding_input_hash
                    THEN NULL
                    ELSE child_chunks.sparse_model
                END,
                sparse_dimension = CASE
                    WHEN child_chunks.embedding_input_hash IS DISTINCT FROM
                         EXCLUDED.embedding_input_hash
                    THEN NULL
                    ELSE child_chunks.sparse_dimension
                END,
                sparse_updated_at = CASE
                    WHEN child_chunks.embedding_input_hash IS DISTINCT FROM
                         EXCLUDED.embedding_input_hash
                    THEN NULL
                    ELSE child_chunks.sparse_updated_at
                END,
                embedding_input_hash = EXCLUDED.embedding_input_hash,
                metadata = EXCLUDED.metadata,
                updated_at = NOW();
            """,
            values,
        )


def fetch_chunks_needing_dense_embeddings(
    connection: psycopg.Connection,
    model: str,
    dimension: int,
    limit: int,
) -> list[dict]:
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                chunk_id,
                retrieval_text,
                image_paths,
                token_count,
                embedding_input_hash
            FROM child_chunks
            WHERE dense_embedding IS NULL
               OR dense_model IS DISTINCT FROM %s
               OR dense_dimension IS DISTINCT FROM %s
            ORDER BY domain, source_type, file_name, chunk_id
            LIMIT %s;
            """,
            (model, dimension, limit),
        )
        return cursor.fetchall()


def update_dense_embeddings(
    connection: psycopg.Connection,
    records: list[dict],
) -> None:
    if not records:
        return

    values = [
        {
            **record,
            "embedding": Vector(record["embedding"]),
        }
        for record in records
    ]
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            UPDATE child_chunks
            SET dense_embedding = %(embedding)s,
                dense_model = %(dense_model)s,
                dense_dimension = %(dense_dimension)s,
                dense_updated_at = NOW(),
                updated_at = NOW()
            WHERE chunk_id = %(chunk_id)s;
            """,
            values,
        )


def dense_indexing_counts(
    connection: psycopg.Connection,
    model: str,
    dimension: int,
) -> dict:
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (
                    WHERE dense_embedding IS NOT NULL
                      AND dense_model = %s
                      AND dense_dimension = %s
                ) AS completed
            FROM child_chunks;
            """,
            (model, dimension),
        )
        result = cursor.fetchone()
    result["pending"] = result["total"] - result["completed"]
    return result


def mark_dense_indexed_documents(
    connection: psycopg.Connection,
    model: str,
    dimension: int,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents AS document
            SET status = 'indexed',
                updated_at = NOW()
            WHERE EXISTS (
                SELECT 1
                FROM child_chunks AS child
                WHERE child.document_id = document.document_id
            )
              AND NOT EXISTS (
                SELECT 1
                FROM child_chunks AS child
                WHERE child.document_id = document.document_id
                  AND (
                      child.dense_embedding IS NULL
                      OR child.dense_model IS DISTINCT FROM %s
                      OR child.dense_dimension IS DISTINCT FROM %s
                  )
            );
            """,
            (model, dimension),
        )


def mark_chunked_documents(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents AS document
            SET status = 'chunked',
                updated_at = NOW()
            WHERE EXISTS (
                SELECT 1
                FROM child_chunks AS child
                WHERE child.document_id = document.document_id
            )
              AND document.status IS DISTINCT FROM 'indexed';
            """
        )


def create_dense_hnsw_index(connection: psycopg.Connection) -> None:
    """Use half-precision HNSW because regular vector HNSW stops at 2,000 dims."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_child_chunks_dense_embedding
            ON child_chunks USING hnsw (
                (dense_embedding::halfvec(2048)) halfvec_cosine_ops
            );
            """
        )


def fetch_chunks_needing_sparse_embeddings(
    connection: psycopg.Connection,
    model: str,
    dimension: int,
    limit: int,
) -> list[dict]:
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(
            """
            SELECT chunk_id, retrieval_text
            FROM child_chunks
            WHERE retrieval_text IS NOT NULL
              AND BTRIM(retrieval_text) <> ''
              AND (
                  sparse_embedding IS NULL
                  OR sparse_model IS DISTINCT FROM %s
                  OR sparse_dimension IS DISTINCT FROM %s
              )
            ORDER BY domain, source_type, file_name, chunk_id
            LIMIT %s;
            """,
            (model, dimension, limit),
        )
        return cursor.fetchall()


def update_sparse_embeddings(
    connection: psycopg.Connection,
    records: list[dict],
) -> None:
    if not records:
        return
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            UPDATE child_chunks
            SET sparse_embedding = %(sparse_embedding)s,
                sparse_model = %(sparse_model)s,
                sparse_dimension = %(sparse_dimension)s,
                sparse_updated_at = NOW(),
                updated_at = NOW()
            WHERE chunk_id = %(chunk_id)s;
            """,
            records,
        )


def sparse_indexing_counts(
    connection: psycopg.Connection,
    model: str,
    dimension: int,
) -> dict:
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE retrieval_text IS NOT NULL
                      AND BTRIM(retrieval_text) <> ''
                ) AS total,
                COUNT(*) FILTER (
                    WHERE sparse_embedding IS NOT NULL
                      AND sparse_model = %s
                      AND sparse_dimension = %s
                ) AS completed
            FROM child_chunks;
            """,
            (model, dimension),
        )
        result = cursor.fetchone()
    result["pending"] = result["total"] - result["completed"]
    return result


def create_sparse_hnsw_index(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_child_chunks_sparse_embedding
            ON child_chunks USING hnsw (sparse_embedding sparsevec_ip_ops);
            """
        )
