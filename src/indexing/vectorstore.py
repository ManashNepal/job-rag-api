import logging
import time

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from src.config import settings
from src.indexing.embedder import get_dense_model, get_sparse_model

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
# Payload fields indexed for filtering and grouping
KEYWORD_FIELDS = ["job_id", "country", "work_type", "job_level", "skills"]
BATCH_SIZE = 128


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client."""
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def create_collection(client: QdrantClient, dense_size: int) -> None:
    """Recreate the collection with a dense and a BM25 sparse vector per point."""
    name = settings.qdrant_collection

    if client.collection_exists(name):
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=dense_size, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            # Qdrant computes the IDF part of BM25 at query time.
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    for field in KEYWORD_FIELDS:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def index_chunks(chunks: list[Document], client: QdrantClient | None = None) -> None:
    """Embed chunks (dense and BM25) and upload them to Qdrant in batches."""
    client = client or get_qdrant_client()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    create_collection(client, dense_size=dense_model.embedding_size)
    logger.info(
        "Indexing %d chunks into '%s'...", len(chunks), settings.qdrant_collection
    )

    start = time.time()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch]

        dense_vectors = dense_model.passage_embed(texts)
        sparse_vectors = sparse_model.passage_embed(texts)

        points = [
            models.PointStruct(
                id=chunk.metadata["chunk_id"],
                vector={
                    DENSE_VECTOR: dense.tolist(),
                    SPARSE_VECTOR: models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={**chunk.metadata, "text": chunk.page_content},
            )
            for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)

        done = min(i + BATCH_SIZE, len(chunks))
        if done % (BATCH_SIZE * 20) == 0 or done == len(chunks):
            logger.info(
                "Indexed %d/%d chunks (%.0fs)", done, len(chunks), time.time() - start
            )
