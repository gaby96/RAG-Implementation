# app/services/interfaces/embedding_store.py

from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Iterable, Optional

from app.models.document_chunk import DocumentChunk


class EmbeddingStore(ABC):
    """
    Interface for storing and retrieving embedding indexes.
    """
    @abstractmethod
    async def store_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        """Store multiple chunks with their embeddings."""

    @abstractmethod
    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """Search for similar chunks using cosine similarity (or vector DB search)."""

    @abstractmethod
    async def search_with_adjacent_chunks(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        adjacent_chunk_count: int = 1,
    ) -> list[DocumentChunk]:
        """Search for similar chunks and include adjacent chunks for better context."""

    @abstractmethod
    async def get_chunks_by_document_and_indices(
        self,
        document_name: str,
        chunk_indices: Iterable[int],
        section_path: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Retrieve specific chunks by document name and chunk indices (optionally filtered by section path)."""

    @abstractmethod
    async def delete_document(self, document_name: str) -> None:
        """Delete all chunks from a specific document."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored chunks."""