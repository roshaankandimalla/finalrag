import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from finalrag.database.connection import connect_database
from finalrag.embeddings.splade_embeddings import load_query_encoder
from finalrag.embeddings.voyage_embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    create_client,
    embed_query,
)
from finalrag.retrieval.dense import retrieve_dense
from finalrag.retrieval.domain_router import route_domains
from finalrag.retrieval.parent_expansion import (
    DEFAULT_MATCHED_CHILDREN_PER_PARENT,
    expand_and_deduplicate_parents,
)
from finalrag.retrieval.reranker import rerank_parents
from finalrag.retrieval.rrf import reciprocal_rank_fusion
from finalrag.retrieval.source_router import route_sources
from finalrag.retrieval.sparse import retrieve_sparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RetrievalSession:
    """Reusable query session that loads SPLADE only once per process."""

    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env", override=True)
        self.voyage_model = os.getenv("VOYAGE_EMBED_MODEL", DEFAULT_MODEL)
        self.dimension = int(
            os.getenv("VOYAGE_EMBED_DIMENSION", str(DEFAULT_DIMENSION))
        )
        self.voyage_client = create_client()
        self.splade = load_query_encoder()
        self.matched_children_per_parent = int(
            os.getenv(
                "MATCHED_CHILDREN_PER_PARENT",
                str(DEFAULT_MATCHED_CHILDREN_PER_PARENT),
            )
        )

    def retrieve(
        self,
        query: str,
        dense_limit: int = 50,
        sparse_limit: int = 50,
        fused_limit: int = 50,
        rerank_limit: int = 8,
    ) -> dict:
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                embed_query,
                self.voyage_client,
                query,
                self.voyage_model,
                self.dimension,
            )
            sparse_future = executor.submit(self.splade.encode_query, query)
            query_dense = dense_future.result()
            query_sparse = sparse_future.result()

        with connect_database() as connection:
            routing = route_domains(
                connection,
                query_dense,
                model=self.voyage_model,
                dimension=self.dimension,
            )
            source_routing = route_sources(query, routing["domains"])
            dense = retrieve_dense(
                connection,
                query_dense,
                routing["domains"],
                limit=dense_limit,
                file_names=source_routing["file_names"],
            )
            sparse = retrieve_sparse(
                connection,
                query_sparse,
                routing["domains"],
                limit=sparse_limit,
                file_names=source_routing["file_names"],
            )
            fused = reciprocal_rank_fusion([dense, sparse], limit=fused_limit)
            parents = expand_and_deduplicate_parents(
                connection,
                fused,
                matched_children_per_parent=self.matched_children_per_parent,
            )

        reranked = rerank_parents(
            self.voyage_client,
            query,
            parents,
            top_k=rerank_limit,
        )
        return {
            "query": query,
            "routing": routing,
            "source_routing": source_routing,
            "counts": {
                "dense": len(dense),
                "sparse": len(sparse),
                "fused_children": len(fused),
                "deduplicated_parents": len(parents),
                "matched_children_per_parent": self.matched_children_per_parent,
                "reranked_parents": len(reranked),
            },
            "results": reranked,
        }


def retrieve(
    query: str,
    dense_limit: int = 50,
    sparse_limit: int = 50,
    fused_limit: int = 50,
    rerank_limit: int = 8,
) -> dict:
    return RetrievalSession().retrieve(
        query,
        dense_limit=dense_limit,
        sparse_limit=sparse_limit,
        fused_limit=fused_limit,
        rerank_limit=rerank_limit,
    )
