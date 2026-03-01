# app/services/interfaces/embedding_generator.py

from __future__ import annotations

from typing import Protocol, List
from app.models.document_chunk import DocumentChunk


class EmbeddingGenerator(Protocol):
    """
    Interface for generating text embeddings.
    """

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text."""
        ...

    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Generate embedding vectors for multiple chunks at once."""
        ...