from __future__ import annotations

from dataclasses import dataclass
import logging

from app.services.rag_service import RagService
logger = logging.getLogger(__name__)

@dataclass(slots=True)
class RagPlugin:
    """
    Exposes high-level RAG operations on top of RagService.
    """
    rag_service: RagService

    async def search_documents(self, query: str, top_k: int = 5) -> str:
        
        results = await self.rag_service.search(query=query, top_k=top_k)
        return self.rag_service.get_context_from_results(results)

    async def index_pdf(self, pdf_file_name: str) -> str:
        try:
            result = await self.rag_service.index_pdf(pdf_file_name=pdf_file_name)
            return f"Indexed {pdf_file_name}: {result.chunk_count} chunks created"
        except Exception as ex:
            return f"Failed to index PDF: {ex}"

    async def delete_document(self, document_name: str) -> str:
        try:
            await self.rag_service.delete_document(document_name=document_name)
            return f"Deleted: {document_name}"
        except Exception as ex:
            return f"Failed to delete: {ex}"

    async def clear_index(self) -> str:
        try:
            await self.rag_service.clear_index()
            return "All documents cleared"
        except Exception as ex:
            return f"Failed to clear: {ex}"