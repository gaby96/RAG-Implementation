from pydantic_settings import BaseSettings

class QdrantSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6334
    api_key: str | None = None
    use_tls: bool = False
    collection_name: str = "documents"
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "QDRANT_"
    }