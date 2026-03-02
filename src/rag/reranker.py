from sentence_transformers import CrossEncoder
from src.config import settings


class Reranker:
    """Simple cross-encoder reranker."""

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.model = CrossEncoder(settings.reranker_model)

    def rerank(self, query: str, documents):
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(documents, scores))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[: self.top_k]]

        return reranked_docs
