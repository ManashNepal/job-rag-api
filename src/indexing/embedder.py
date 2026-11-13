from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.config import settings


# Each model is loaded once and reused.
@lru_cache
def get_dense_model() -> TextEmbedding:
    """Dense embedding model for semantic search."""
    return TextEmbedding(model_name=settings.dense_model)


@lru_cache
def get_sparse_model() -> SparseTextEmbedding:
    """BM25 sparse model for keyword search."""
    return SparseTextEmbedding(model_name=settings.sparse_model)


@lru_cache
def get_reranker_model() -> TextCrossEncoder:
    """Cross-encoder model for reranking retrieved jobs."""
    return TextCrossEncoder(model_name=settings.reranker_model)
