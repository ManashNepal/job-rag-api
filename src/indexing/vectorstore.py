from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_classic.schema import Document
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
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(index_name)


def create_vectorstore(documents: List[Document]):
    """Create and populate pinecone vectorstore with the documents."""

    embedding_model = get_embedding_model()
    index = initialize_pinecone_index(
        dimension=3072
    )  # gemini embedding model used returns vectors of length 3072

    vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

    vectorstore.add_documents(documents=documents)

    return vectorstore
