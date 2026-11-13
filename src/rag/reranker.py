from langchain_core.documents import Document

from src.config import settings
from src.indexing.embedder import get_reranker_model


class Reranker:
    """Cross-encoder reranker."""

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.model = get_reranker_model()

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        scores = self.model.rerank(
            query,
            [doc.page_content for doc in documents],
            batch_size=settings.model_batch_size,
        )
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        reranked_docs = []
        for doc, score in scored_docs[: self.top_k]:
            doc.metadata["rerank_score"] = float(score)
            reranked_docs.append(doc)

        return reranked_docs
