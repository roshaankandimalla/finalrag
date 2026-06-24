import os

import voyageai


DEFAULT_RERANK_MODEL = "rerank-2.5"


def rerank_parents(
    client: voyageai.Client,
    query: str,
    parents: list[dict],
    top_k: int = 8,
    model: str | None = None,
) -> list[dict]:
    if not parents:
        return []

    selected_model = model or os.getenv("VOYAGE_RERANK_MODEL", DEFAULT_RERANK_MODEL)
    result = client.rerank(
        query=query,
        documents=[parent["rerank_text"] for parent in parents],
        model=selected_model,
        top_k=min(top_k, len(parents)),
        truncation=True,
    )
    return [
        {
            **parents[item.index],
            "rerank_score": item.relevance_score,
            "rerank_model": selected_model,
            "rerank_provider": "voyage",
        }
        for item in result.results
    ]
