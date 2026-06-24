import os

import psycopg
from pgvector import Vector
from psycopg.rows import dict_row


ALL_DOMAINS = ["finance", "legal", "medical"]


def select_routed_domains(
    domain_scores: dict[str, float],
    minimum: float,
    margin: float,
) -> tuple[list[str], str]:
    if not domain_scores:
        return ALL_DOMAINS, "fallback_all"
    ordered = sorted(domain_scores.items(), key=lambda item: -item[1])
    top_score = ordered[0][1]
    second_score = ordered[1][1] if len(ordered) > 1 else float("-inf")
    if top_score < minimum:
        return ALL_DOMAINS, "low_all_domains"
    if top_score - second_score >= margin:
        return [ordered[0][0]], "high_single_domain"
    return [domain for domain, _ in ordered[:2]], "ambiguous_two_domains"


def rebuild_domain_centroids(
    connection: psycopg.Connection,
    model: str,
    dimension: int,
) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM domain_centroids
            WHERE dense_model = %s AND dense_dimension = %s;
            """,
            (model, dimension),
        )
        cursor.execute(
            """
            INSERT INTO domain_centroids (
                centroid_id,
                domain,
                retrieval_type,
                chunk_count,
                dense_model,
                dense_dimension,
                centroid_embedding,
                updated_at
            )
            SELECT
                domain || ':' || retrieval_type || ':' || %s || ':' || %s,
                domain,
                retrieval_type,
                COUNT(*),
                %s,
                %s,
                AVG(dense_embedding),
                NOW()
            FROM child_chunks
            WHERE dense_embedding IS NOT NULL
              AND dense_model = %s
              AND dense_dimension = %s
            GROUP BY domain, retrieval_type;
            """,
            (model, dimension, model, dimension, model, dimension),
        )
        return cursor.rowcount


def route_domains(
    connection: psycopg.Connection,
    query_embedding: list[float],
    model: str,
    dimension: int,
) -> dict:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                domain,
                retrieval_type,
                chunk_count,
                1 - (
                    centroid_embedding::halfvec(2048)
                    <=> %s::halfvec(2048)
                ) AS similarity
            FROM domain_centroids
            WHERE dense_model = %s
              AND dense_dimension = %s
            ORDER BY centroid_embedding::halfvec(2048) <=> %s::halfvec(2048);
            """,
            (Vector(query_embedding), model, dimension, Vector(query_embedding)),
        )
        centroid_scores = cursor.fetchall()

    if not centroid_scores:
        return {
            "domains": ALL_DOMAINS,
            "confidence": "fallback_all",
            "domain_scores": {},
            "centroid_scores": [],
        }

    domain_scores = {}
    for item in centroid_scores:
        domain_scores[item["domain"]] = max(
            domain_scores.get(item["domain"], float("-inf")),
            item["similarity"],
        )
    minimum = float(os.getenv("DOMAIN_ROUTER_MIN_SIMILARITY", "0.25"))
    margin = float(os.getenv("DOMAIN_ROUTER_SINGLE_DOMAIN_MARGIN", "0.03"))
    domains, confidence = select_routed_domains(domain_scores, minimum, margin)
    ordered = sorted(domain_scores.items(), key=lambda item: -item[1])
    return {
        "domains": domains,
        "confidence": confidence,
        "domain_scores": dict(ordered),
        "centroid_scores": centroid_scores,
    }
