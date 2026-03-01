from pydantic_settings import BaseSettings
from app.models.enums import RerankerType

class RerankerSettings(BaseSettings):
    endpoint: str
    api_key: str | None = None
    deployment_name: str | None = None
    type: RerankerType

    model_config = {
        "env_file": ".env",
        "env_prefix": "RERANKER_"
    }