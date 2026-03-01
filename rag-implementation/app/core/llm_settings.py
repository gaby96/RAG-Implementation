from pydantic_settings import BaseSettings
from app.models.enums import LlmType

class LLMSettings(BaseSettings):
    endpoint: str
    api_key: str
    chat_deployment_name: str
    temperature: float
    type: LlmType

    model_config = {
        "env_file": ".env",
        "env_prefix": "LLM_"
    }