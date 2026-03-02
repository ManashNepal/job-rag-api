from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    # API keys
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")
    pinecone_api_key: str = Field(..., validation_alias="PINECONE_API_KEY")
    # Pinecone
    pinecone_index_name: str = "job-rag-index"
    pinecone_dimension: int = 384
    # Retrieval
    retriever_top_k: int = 8
    reranker_top_k: int = 3
    # LLM
    groq_model: str = "llama-3.3-70b-versatile"
    # Embedding
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


settings = Settings()
