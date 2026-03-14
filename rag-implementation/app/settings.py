from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


from app.core.llm_settings import LLMSettings
from app.core.qdrant_settings import QdrantSettings
from app.core.embedding_settings import EmbeddingSettings
from app.core.reranker_settings import RerankerSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    llm: LLMSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    qdrant: QdrantSettings


settings = Settings() 