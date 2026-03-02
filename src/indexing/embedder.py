from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings


def get_embedding_model():
    """Initialize and return hugging face embedding model."""

    return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)
