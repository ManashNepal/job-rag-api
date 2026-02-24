from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    google_api_key: str = Field(..., validation_alias="GOOGLE_API_KEY")
    pinecone_api_key: str = Field(..., validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(validation_alias="PINECONE_INDEX_NAME")


settings = Settings()
