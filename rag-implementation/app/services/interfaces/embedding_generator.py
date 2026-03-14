# app/services/interfaces/embedding_generator.py

from __future__ import annotations

from abc import ABC, abstractmethod
from app.models.document_chunk import DocumentChunk

class EmbeddingGenerator(ABC):
    """
    Interface for generating text embeddings.
    """

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""

    @abstractmethod
    async def generate_embeddings(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Generate embedding vectors for multiple chunks at once."""