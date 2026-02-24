from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Initialize and return gemini embedding model."""

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=settings.google_api_key
    )
