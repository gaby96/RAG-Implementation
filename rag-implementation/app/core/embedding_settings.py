from pydantic_settings import BaseSettings
from app.models.enums import EmbeddingType, EmbeddingStoreType

class EmbeddingSettings(BaseSettings):
    endpoint : str
    api_key : str
    deployment_name : str

    type: EmbeddingType

    max_chunk_tokens: int
    chunk_overlap_tokens: int
    vector_size: int

    dense_vector_weight: float = 0.7

    store_type: EmbeddingStoreType

    model_config = {
        "env_file": ".env",
        "env_prefix": "EMBEDDING_"
    }
