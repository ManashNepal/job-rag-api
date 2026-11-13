import uuid
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

from src.config import settings

# For the model's special tokens and the blank line
TOKEN_MARGIN = 10


@lru_cache
def get_tokenizer() -> Tokenizer:
    """Tokenizer of the embedding model"""
    return Tokenizer.from_pretrained(settings.dense_model)


def count_tokens(text: str) -> int:
    """Number of model tokens in a text."""
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids)


def build_header(metadata: dict) -> str:
    """Short job header repeated at the top of every chunk."""
    skills = metadata["skills"][: settings.max_header_skills]
    return (
        f"Job Title: {metadata['title']}\n"
        f"Company: {metadata['company']}\n"
        f"Location: {metadata['location']}, {metadata['country']}\n"
        f"Work Type: {metadata['work_type']} | Job Level: {metadata['job_level']}\n"
        f"Skills: {', '.join(skills) if skills else 'Not listed'}"
    )


def chunk_id(job_link: str, chunk_index: int) -> str:
    """Deterministic ID for one chunk of a job."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{job_link}#{chunk_index}"))


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split each job description into chunks, each prefixed with the job header.

    Chunk sizes are measured in tokens, so header + chunk always fits the model limit.
    """
    chunks = []
    for doc in documents:
        header = build_header(doc.metadata)

        # Token budget left for the description after this job's header.
        budget = min(
            settings.chunk_size,
            settings.max_chunk_tokens - count_tokens(header) - TOKEN_MARGIN,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=settings.chunk_overlap,
            length_function=count_tokens,
        )

        for index, piece in enumerate(splitter.split_text(doc.page_content)):
            chunks.append(
                Document(
                    page_content=f"{header}\n\n{piece}",
                    metadata={
                        **doc.metadata,
                        "chunk_id": chunk_id(doc.metadata["job_link"], index),
                        "chunk_index": index,
                    },
                )
            )

    return chunks
