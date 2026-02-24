from typing import List
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
    )

    chunked_documents = splitter.split_documents(documents=documents)
    return chunked_documents
