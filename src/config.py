from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    # API keys
    anthropic_api_key: str | None = None
    hf_token: str | None = None
    # Langfuse tracing
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    app_environment: str = "development"  # shown in Langfuse to tell traces apart
    trace_llm_content: bool = True  # include prompts and responses in LLM call traces
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None  # only needed for Qdrant Cloud
    qdrant_collection: str = "jobs"
    # Models
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Texts per model call. Larger batches use much more memory
    model_batch_size: int = 16
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
    llm_model: str = "claude-sonnet-5-5"
    llm_effort: str = "low"
    llm_max_tokens: int = 8000  # covers thinking and the answer
    # Query filters
    filter_model: str = "claude-haiku-5-5"
    filter_effort: str = "low"
    filter_max_tokens: int = 2000
    countries: list[str] = ["United States", "United Kingdom", "Canada", "Australia"]
    min_filtered_results: int = (
        5  # if fewer jobs than this, search again without filters
    )
    # Search result cache (off when REDIS_URL is not set)
    redis_url: str | None = None
    cache_ttl_seconds: int = 3600

    @field_validator(
        "anthropic_api_key",
        "hf_token",
        "qdrant_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        "redis_url",
        mode="before",
    )
    @classmethod
    def empty_to_none(cls, value):
        """Treat an empty key in .env as no key."""
        return value or None


settings = Settings()
