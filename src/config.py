from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    # API keys
    groq_api_key: str = Field(..., validation_alias="GROQ_API_KEY")
    hf_token: str | None = None
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None  # only needed for Qdrant Cloud
    qdrant_collection: str = "jobs"
    # Models
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Chunking (in tokens)
    max_chunk_tokens: int = 512  # bge-small and the reranker limit
    chunk_size: int = 256  # description tokens per chunk
    chunk_overlap: int = 32
    max_header_skills: int = 30
    # Retrieval
    prefetch_limit: int = 50  # candidates per search type (dense / BM25) before fusion
    retriever_top_k: int = 30  # jobs passed to the reranker
    reranker_top_k: int = 5  # jobs passed to the LLM
    # LLM
    groq_model: str = "llama-3.3-70b-versatile"


settings = Settings()
