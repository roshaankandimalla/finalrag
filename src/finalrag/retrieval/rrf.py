def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    rank_constant: int = 60,
    limit: int = 50,
) -> list[dict]:
    fused: dict[str, dict] = {}
    for source_index, ranked in enumerate(ranked_lists):
        source = "dense" if source_index == 0 else "sparse"
        for rank, item in enumerate(ranked, start=1):
            chunk_id = str(item["chunk_id"])
            record = fused.setdefault(
                chunk_id,
                {
                    **item,
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "retrieval_sources": [],
                    "source_ranks": {},
                },
            )
            record["rrf_score"] += 1.0 / (rank_constant + rank)
            record["retrieval_sources"].append(source)
            record["source_ranks"][source] = rank
            if item.get("dense_score") is not None:
                record["dense_score"] = item["dense_score"]
            if item.get("sparse_score") is not None:
                record["sparse_score"] = item["sparse_score"]
    return sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], item["chunk_id"]),
    )[:limit]
