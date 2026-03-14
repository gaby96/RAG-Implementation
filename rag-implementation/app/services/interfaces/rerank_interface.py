from __future__ import annotations

from typing import Protocol

from app.models.document_chunk import DocumentChunk
from app.models.reranked_result import RerankedResult
from abc import ABC, abstractmethod

class Reranker(ABC):
    """
    Equivalent of IReranker in C#.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[DocumentChunk],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """
        Re-rank the given search results based on relevance to the query.
        Returns reranked results with relevance scores.
        """