from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from src.config import settings
from src.indexing.embedder import get_dense_model, get_sparse_model
from src.indexing.vectorstore import DENSE_VECTOR, SPARSE_VECTOR, get_qdrant_client
from src.rag.schemas import QueryFilters


class HybridRetriever:
    """Hybrid (dense and BM25) retriever over job chunks in Qdrant."""

    def __init__(self, top_k: int, client: QdrantClient | None = None):
        self.top_k = top_k
        self.client = client or get_qdrant_client()
        self.dense_model = get_dense_model()
        self.sparse_model = get_sparse_model()

    def search(self, query: str, filters: QueryFilters | None = None) -> list[Document]:
        """Return the best-matching chunk for each of the top_k jobs, within the filters."""
        query_filter = self.build_filter(filters) if filters else None

        dense_vector = next(self.dense_model.query_embed(query)).tolist()
        sparse = next(self.sparse_model.query_embed(query))
        sparse_vector = models.SparseVector(
            indices=sparse.indices.tolist(), values=sparse.values.tolist()
        )

        result = self.client.query_points_groups(
            collection_name=settings.qdrant_collection,
            # Run both searches, then merge with Reciprocal Rank Fusion
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR,
                    filter=query_filter,
                    limit=settings.prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=SPARSE_VECTOR,
                    filter=query_filter,
                    limit=settings.prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            # One best-scoring chunk per job
            group_by="job_id",
            group_size=1,
            limit=self.top_k,
            with_payload=True,
        )

        documents = []
        for group in result.groups:
            hit = group.hits[0]
            payload = dict(hit.payload)
            text = payload.pop("text")
            payload["retrieval_score"] = hit.score
            documents.append(Document(page_content=text, metadata=payload))

        return documents

    @staticmethod
    def build_filter(filters: QueryFilters) -> models.Filter | None:
        """Turn query filters into a Qdrant filter (all conditions must match)."""
        conditions = []
        if filters.work_type:
            conditions.append(
                models.FieldCondition(
                    key="work_type", match=models.MatchValue(value=filters.work_type)
                )
            )
        if filters.country:
            conditions.append(
                models.FieldCondition(
                    key="country", match=models.MatchValue(value=filters.country)
                )
            )
        if filters.seniority:
            conditions.append(
                models.FieldCondition(
                    key="seniority",
                    match=models.MatchAny(any=[filters.seniority, "unspecified"]),
                )
            )
        return models.Filter(must=conditions) if conditions else None
