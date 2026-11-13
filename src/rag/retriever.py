from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from src.config import settings
from src.indexing.embedder import get_dense_model, get_sparse_model
from src.indexing.vectorstore import DENSE_VECTOR, SPARSE_VECTOR, get_qdrant_client


class HybridRetriever:
    """Hybrid (dense and BM25) retriever over job chunks in Qdrant."""

    def __init__(self, top_k: int, client: QdrantClient | None = None):
        self.top_k = top_k
        self.client = client or get_qdrant_client()
        self.dense_model = get_dense_model()
        self.sparse_model = get_sparse_model()

    def search(self, query: str) -> list[Document]:
        """Return the best-matching chunk for each of the top_k jobs."""
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
                    limit=settings.prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=SPARSE_VECTOR,
                    limit=settings.prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
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
