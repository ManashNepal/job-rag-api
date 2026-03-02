from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_classic.schema import Document

from src.config import settings
from src.indexing.embedder import get_embedding_model


class HybridRetriever:
    """Hybrid retriever using Pinecone dense and sparse search."""

    def __init__(self, top_k: int):
        self.top_k = top_k
        # Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index_name)
        # Embeddings
        self.embedding_model = get_embedding_model()
        # Load fitted BM25
        self.bm25 = BM25Encoder().load("bm25.pkl")

    def search(self, query: str):
        dense_vector = self.embedding_model.embed_query(query)
        sparse_vector = self.bm25.encode_queries(query)

        results = self.index.query(
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=self.top_k,
            include_metadata=True,
        )

        documents = []

        for match in results["matches"]:
            metadata = match["metadata"] or {}
            text = metadata.get("text", "")
            doc = Document(
                page_content=text,
                metadata=metadata,
            )
            documents.append(doc)

        return documents
