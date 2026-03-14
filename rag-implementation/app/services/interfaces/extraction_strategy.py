from __future__ import annotations

from typing import Protocol, Dict, Iterable, BinaryIO
from app.models.document_chunk import DocumentChunk
from app.models.document_section import DocumentSection
from abc import ABC, abstractmethod


class ExtractionStrategy(ABC):
    """
    Equivalent to IExtractionStrategy
    """

    @abstractmethod
    async def extract(self, file: BinaryIO) -> list[DocumentSection]:
        """
        Extract document sections from a file stream
        """
        ...


class DocumentExtractor(ABC):
    """
    Equivalent to IDocumentExtractor
    """

    @abstractmethod
    async def process_document(
        self,
        file_path: str,
        extraction_strategy: ExtractionStrategy
    ) -> list[DocumentChunk]:
        """
        Process a document file and create chunks
        """
        ...

    @abstractmethod
    async def process_documents(
        self,
        file_paths: Iterable[str],
        extraction_strategy: ExtractionStrategy
    ) -> Dict[str, list[DocumentChunk]]:
        """
        Process multiple documents
        """
        ...