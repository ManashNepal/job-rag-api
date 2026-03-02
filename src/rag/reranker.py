from sentence_transformers import CrossEncoder


class Reranker:
    """Simple cross-encoder reranker."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
