def citation_for_parent(parent: dict, index: int) -> dict:
    children = parent.get("matched_children") or []
    first = children[0] if children else {}
    page_numbers = sorted(
        {
            page
            for child in children
            for page in (child.get("page_numbers") or [])
        }
    )
    row_numbers = sorted(
        {
            row
            for child in children
            for row in (child.get("source_row_numbers") or [])
        }
    )
    return {
        "id": index,
        "label": f"[{index}]",
        "parent_id": str(parent["parent_id"]),
        "document_id": str(parent["document_id"]),
        "domain": parent["domain"],
        "source_type": parent["source_type"],
        "file_name": first.get("file_name"),
        "section_title": parent.get("section_title"),
        "page_numbers": page_numbers,
        "source_row_numbers": row_numbers,
        "matched_chunk_ids": [child["chunk_id"] for child in children],
        "rerank_score": parent.get("rerank_score"),
    }


def format_source(citation: dict) -> str:
    location = []
    if citation["page_numbers"]:
        location.append(
            "pages " + ", ".join(str(value) for value in citation["page_numbers"])
        )
    if citation["source_row_numbers"]:
        rows = citation["source_row_numbers"]
        location.append(f"rows {rows[0]}-{rows[-1]}")
    details = " | ".join(
        value
        for value in [
            citation.get("file_name"),
            citation.get("section_title"),
            ", ".join(location) if location else None,
        ]
        if value
    )
    return f"{citation['label']} {details}"
