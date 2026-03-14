from enum import Enum

class EmbeddingType(str, Enum):
    http = "http"
    azure_openai = "azure_openai"

class EmbeddingStoreType(str, Enum):
    file_system = "file_system"
    qdrant = "qdrant"
    qdrant_hybrid = "qdrant_hybrid"
    qdrant_hybrid_idf = "qdrant_hybrid_idf"

class LlmType(str, Enum):
    azure_openai = "azure_openai"
    openai = "openai"


class RerankerType(str, Enum):
   azure_openai = "azure_openai"
   cross_encoder = "cross_encoder"