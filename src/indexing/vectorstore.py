from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_classic.schema import Document
from pinecone_text.sparse import BM25Encoder
from src.config import settings
from src.indexing.embedder import get_embedding_model


def initialize_pinecone_index(dimension: int):
    """Initialize pinecone and create index if it doesn't exist already."""

    # Initialize pinecone client
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    existing_indexes = [index.name for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(index_name)


def create_vectorstore(documents: List[Document]):
    """
    Create and populate pinecone vectorstore with hybrid (dense and sparse) vectors.
    Uses batch upsert to avoid request size limits.
    """

    embedding_model = get_embedding_model()
    index = initialize_pinecone_index(
        dimension=384,
    )  # hugging face embedding model used returns vectors of length 384

    print(f"Indexing {len(documents)} documents...")

    # Prepare corpus for BM25
    corpus = [doc.page_content for doc in documents]
    bm25 = BM25Encoder()
    bm25.fit(corpus=corpus)
    bm25.dump("bm25.pkl")

    vectors = []

    for doc in documents:

        job_id = doc.metadata.get("id")

        dense_vector = embedding_model.embed_query(doc.page_content)
        sparse_vector = bm25.encode_documents(doc.page_content)

        vectors.append(
            {
                "id": str(job_id),
                "values": dense_vector,
                "sparse_values": sparse_vector,
                "metadata": {
                    **doc.metadata,
                    "text": doc.page_content,  # needed for retrieval reconstruction
                },
            }
        )

    # Batch Upsert
    batch_size = 100

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    print("Hybrid indexing completed!")
    return index
