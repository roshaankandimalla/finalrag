from finalrag.generation.citations import citation_for_parent


def generation_assets(reranked_parents: list[dict]) -> dict:
    image_paths = []
    table_html = []
    seen_images = set()
    seen_tables = set()
    for parent in reranked_parents:
        for child in parent.get("matched_children") or []:
            for image_path in child.get("image_paths") or []:
                if image_path not in seen_images:
                    seen_images.add(image_path)
                    image_paths.append(image_path)
            html = (child.get("table_html") or "").strip()
            if html and html not in seen_tables:
                seen_tables.add(html)
                table_html.append(html)
    return {"image_paths": image_paths, "table_html": table_html}


def context_blocks(reranked_parents: list[dict]) -> list[str]:
    blocks = []
    for index, parent in enumerate(reranked_parents, start=1):
        citation = citation_for_parent(parent, index)
        children = parent.get("matched_children") or []
        evidence = "\n\n".join(
            child["retrieval_text"] for child in children if child.get("retrieval_text")
        )
        blocks.append(
            f"{citation['label']}\n"
            f"Domain: {citation['domain']}\n"
            f"Source: {citation.get('file_name') or 'unknown'}\n"
            f"Section: {citation.get('section_title') or 'unknown'}\n"
            f"Parent context:\n{parent.get('parent_text') or ''}\n\n"
            f"Matched evidence:\n{evidence}"
        )
    return blocks


def build_grounded_context(
    query: str,
    reranked_parents: list[dict],
    max_characters: int = 60_000,
) -> tuple[str, list[dict]]:
    blocks = []
    citations = []
    used = 0
    for index, (parent, block) in enumerate(
        zip(reranked_parents, context_blocks(reranked_parents)),
        start=1,
    ):
        citation = citation_for_parent(parent, index)
        if blocks and used + len(block) > max_characters:
            break
        blocks.append(block)
        citations.append(citation)
        used += len(block)

    prompt = (
        f"User question:\n{query}\n\n"
        "Retrieved evidence:\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    tables = generation_assets(reranked_parents)["table_html"]
    if tables:
        prompt += "\n\nRetrieved table HTML:\n\n" + "\n\n---\n\n".join(tables)
    return prompt, citations
