import psycopg
from pgvector import SparseVector
from psycopg.rows import dict_row


def retrieve_sparse(
    connection: psycopg.Connection,
    query_embedding: SparseVector,
    domains: list[str],
    limit: int = 50,
    file_names: list[str] | None = None,
) -> list[dict]:
    source_filter = file_names or None
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order';")
        cursor.execute("SET LOCAL hnsw.ef_search = 200;")
        cursor.execute(
            """
            SELECT
                chunk_id,
                parent_id,
                document_id,
                domain,
                retrieval_type,
                file_name,
                section_title,
                page_numbers,
                source_urls,
                source_row_numbers,
                retrieval_text,
                table_html,
                image_paths,
                -(sparse_embedding <#> %s) AS sparse_score
            FROM child_chunks
            WHERE sparse_embedding IS NOT NULL
              AND domain = ANY(%s)
              AND (%s::text[] IS NULL OR file_name = ANY(%s::text[]))
            ORDER BY sparse_embedding <#> %s
            LIMIT %s;
            """,
            (
                query_embedding,
                domains,
                source_filter,
                source_filter,
                query_embedding,
                limit,
            ),
        )
        return cursor.fetchall()
