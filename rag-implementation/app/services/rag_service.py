from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.models.document_chunk import DocumentChunk
from app.models.index_result import IndexResult
from app.services.interfaces.extraction_strategy import DocumentExtractor, ExtractionStrategy
from app.services.interfaces.embedding_generator  import EmbeddingGenerator
from app.services.interfaces.embedding_store import EmbeddingStore
from app.services.interfaces.rerank_interface import Reranker

# your extraction strategies
from app.document_extraction.pdf_format_based_extractor import PdfFormatBasedExtractor
# optional:
# from app.document_extraction.strategies.pdf_bookmark_extractor import PdfBookmarkExtractor
# from app.document_extraction.strategies.pdf_simple_extractor import PdfSimpleExtractor


@dataclass(slots=True)
class RagService:
    embedding_store: EmbeddingStore
    embedding_generator: EmbeddingGenerator
    reranker: Reranker
    document_extractor: DocumentExtractor
    adjacent_chunk_count: int = 1

    def _resolve_pdf_path(self, pdf_file_name: str) -> Path:
        """
        Mirrors:
          Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "Data", pdfFileName))
        """
        base_dir = Path(__file__).resolve().parents[2]  # adjust if your structure differs
        pdf_path = (base_dir / "data" / pdf_file_name).resolve()
        return pdf_path

    async def index_pdf(self, pdf_file_name: str) -> IndexResult:
        pdf_path = self._resolve_pdf_path(pdf_file_name)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Indexing {pdf_file_name}")

        # Choose extraction strategy (same idea as your C#)
        # strategy: ExtractionStrategy = PdfBookmarkExtractor(max_bookmark_depth=2, skip_pages=10)
        strategy: ExtractionStrategy = PdfFormatBasedExtractor(max_heading_depth=2)
        # strategy: ExtractionStrategy = PdfSimpleExtractor(skip_pages=10)

        chunks: list[DocumentChunk] = await self.document_extractor.process_document(
            file_path=str(pdf_path),
            extraction_strategy=strategy,
        )

        print(f"Extracted {pdf_file_name} into {len(chunks)} chunks")

        embeddings = await self.embedding_generator.generate_embeddings(chunks)

        # attach embeddings back to chunks
        # (assuming embeddings[i] is list[float])
        for i in range(min(len(chunks), len(embeddings))):
            chunks[i].embedding = embeddings[i]

        await self.embedding_store.store_chunks(chunks)

        return IndexResult(chunk_count=len(chunks))

    async def index_pdfs(self, pdf_file_names: Iterable[str]) -> None:
        for name in pdf_file_names:
            await self.index_pdf(name)

    async def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        # Generate embedding for query
        query_embedding = await self.embedding_generator.generate_embedding(query)

        # Retrieve more initially for reranking
        retrieval_count = top_k * 2

        # Search with adjacent chunk expansion handled by store
        results = await self.embedding_store.search_with_adjacent_chunks(
            query=query,
            query_embedding=query_embedding,
            top_k=retrieval_count,
            adjacent_chunk_count=self.adjacent_chunk_count,
        )

        if self.reranker:
            reranked = await self.reranker.rerank(query=query, results=results, top_k=top_k)
            return [r.chunk for r in reranked]
        else:
            return results[:top_k]

    def get_context_from_results(self, search_results: list[DocumentChunk]) -> str:
        if not search_results:
            return "No relevant information found."

        lines: list[str] = []
        lines.append("Context from documents:")
        lines.append("")

        for r in search_results:
            header = f"[Source: {r.source_document}"
            if r.section_path and r.section_path != r.source_document:
                header += f", Section: {r.section_path}"
            elif r.section and r.section != r.source_document:
                header += f", Section: {r.section}"
            header += f", Pages: {r.start_page}-{r.end_page}]"

            lines.append(header)
            lines.append(r.text or "")
            lines.append("")

        return "\n".join(lines)

    async def delete_document(self, document_name: str) -> None:
        await self.embedding_store.delete_document(document_name)
        print(f"Deleted document: {document_name}")

    async def clear_index(self) -> None:
        await self.embedding_store.clear()
        print("Cleared all indexed documents")