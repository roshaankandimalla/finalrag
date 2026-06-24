from collections import defaultdict

import psycopg
from psycopg.rows import dict_row


DEFAULT_MATCHED_CHILDREN_PER_PARENT = 3


def expand_and_deduplicate_parents(
    connection: psycopg.Connection,
    fused_children: list[dict],
    matched_children_per_parent: int = DEFAULT_MATCHED_CHILDREN_PER_PARENT,
) -> list[dict]:
    grouped = defaultdict(list)
    for child in fused_children:
        grouped[str(child["parent_id"])].append(child)
    if not grouped:
        return []

    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                parent_id,
                document_id,
                domain,
                source_type,
                section_title,
                section_path,
                page_numbers,
                parent_text,
                metadata
            FROM parent_sections
            WHERE parent_id = ANY(%s::uuid[]);
            """,
            (list(grouped),),
        )
        parents = {str(item["parent_id"]): item for item in cursor.fetchall()}

    results = []
    for parent_id, matches in grouped.items():
        parent = parents.get(parent_id)
        if not parent:
            continue
        matches.sort(key=lambda item: -item["rrf_score"])
        selected = matches[:matched_children_per_parent]
        evidence = "\n\n".join(
            f"Matched child {index}:\n{item['retrieval_text']}"
            for index, item in enumerate(selected, start=1)
        )
        parent_text = parent.get("parent_text") or ""
        results.append(
            {
                **parent,
                "parent_id": parent_id,
                "matched_children": selected,
                "best_rrf_score": selected[0]["rrf_score"],
                "rerank_text": (
                    f"Section: {parent.get('section_title') or ''}\n\n"
                    f"Parent context:\n{parent_text}\n\n"
                    f"Matched evidence:\n{evidence}"
                ),
            }
        )
    return sorted(results, key=lambda item: -item["best_rrf_score"])
