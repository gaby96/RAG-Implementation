# app/services/document_extractor.py

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Set

import tiktoken

from app.core.embedding_settings import EmbeddingSettings
from app.models.document_chunk import DocumentChunk
from app.models.document_section import DocumentSection


_SENTENCE_ENDINGS = (".", "!", "?")


@dataclass(slots=True)
class BookmarkDocumentExtractor:
    """
    Processes PDFs by using an ExtractionStrategy that returns DocumentSections
    (e.g., derived from bookmarks/outlines), then chunks section text by token limits.
    """

    max_chunk_tokens: int
    chunk_overlap_tokens: int
    tokenizer_model: str = "gpt-4"

    def __post_init__(self) -> None:
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be > 0")

        if self.chunk_overlap_tokens < 0 or self.chunk_overlap_tokens >= self.max_chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be between 0 and max_chunk_tokens")

        # Closest equivalent to ML.NET Tokenizers TiktokenTokenizer
        self._enc = tiktoken.encoding_for_model(self.tokenizer_model)

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def process_document(
        self,
        file_path: str,
        extraction_strategy,
    ) -> List[DocumentChunk]:
        source_document = os.path.basename(file_path)
        sections = await self._extract_sections(file_path, extraction_strategy)
        return self._create_chunks_from_sections(sections, source_document)

    async def process_documents(
        self,
        file_paths: Iterable[str],
        extraction_strategy,
    ) -> Dict[str, List[DocumentChunk]]:
        results: Dict[str, List[DocumentChunk]] = {}
        for fp in file_paths:
            file_name = os.path.basename(fp)
            results[file_name] = await self.process_document(fp, extraction_strategy)
        return results

    async def _extract_sections(self, file_path: str, extraction_strategy) -> List[DocumentSection]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        with open(file_path, "rb") as f:
            return await extraction_strategy.extract(f)

    def _create_chunks_from_sections(self, sections: List[DocumentSection], source_document: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for section in sections:
            chunks.extend(self._create_chunks_from_section(section, source_document))
        return chunks

    def _create_chunks_from_section(self, section: DocumentSection, source_document: str) -> List[DocumentChunk]:
        if not section.page_texts:
            return []

        full_text = section.text.strip()
        if not full_text:
            return []

        token_count = self._count_tokens(full_text)

        # Fits in one chunk
        if token_count <= self.max_chunk_tokens:
            pages = [p.page_number for p in section.page_texts]
            return [self._create_chunk(
                text=full_text,
                section=section,
                pages=sorted(set(pages)),
                source_document=source_document,
                chunk_index=0,
                total_sub_chunks=1,
            )]

        # Needs splitting
        sub_chunks = self._split_large_section_with_pages(section)
        total = len(sub_chunks)

        chunks: List[DocumentChunk] = []
        for i, (text, pages) in enumerate(sub_chunks):
            chunks.append(self._create_chunk(
                text=text,
                section=section,
                pages=pages,
                source_document=source_document,
                chunk_index=i,
                total_sub_chunks=total,
            ))
        return chunks

    def _split_large_section_with_pages(self, section: DocumentSection) -> List[Tuple[str, List[int]]]:
        result: List[Tuple[str, List[int]]] = []

        current_text_parts: List[str] = []
        current_pages: Set[int] = set()
        current_token_count = 0

        for page in section.page_texts:
            page_content = page.text or ""
            idx = 0

            while idx < len(page_content):
                remaining = page_content[idx:]

                # space between parts if needed
                space_needed = 1 if current_text_parts else 0
                available_tokens = self.max_chunk_tokens - current_token_count - space_needed

                if available_tokens <= 0:
                    # flush
                    if current_text_parts:
                        chunk_text = " ".join(current_text_parts).strip()
                        result.append((chunk_text, sorted(current_pages)))

                        overlap_text = self._get_overlap_text(chunk_text)
                        last_page = max(current_pages) if current_pages else None

                        current_text_parts = []
                        current_pages = set()

                        if overlap_text:
                            current_text_parts = [overlap_text]
                            current_token_count = self._count_tokens(overlap_text)
                            if last_page is not None:
                                current_pages.add(last_page)
                        else:
                            current_token_count = 0
                    continue

                remaining_tokens = self._count_tokens(remaining)

                if remaining_tokens <= available_tokens:
                    # fits all
                    if current_text_parts:
                        current_token_count += 1  # for the joining space
                    current_text_parts.append(remaining)
                    current_token_count += remaining_tokens
                    current_pages.add(page.page_number)
                    break
                else:
                    # take as much as fits (prefer sentence boundary)
                    text_to_take = self._get_text_within_token_limit(remaining, available_tokens)

                    if current_text_parts:
                        current_token_count += 1
                    current_text_parts.append(text_to_take)
                    current_pages.add(page.page_number)

                    # flush
                    chunk_text = " ".join(current_text_parts).strip()
                    result.append((chunk_text, sorted(current_pages)))

                    overlap_text = self._get_overlap_text(chunk_text)
                    last_page = max(current_pages) if current_pages else None

                    current_text_parts = []
                    current_pages = set()

                    if overlap_text:
                        current_text_parts = [overlap_text]
                        current_token_count = self._count_tokens(overlap_text)
                        if last_page is not None:
                            current_pages.add(last_page)
                    else:
                        current_token_count = 0

                    idx += len(text_to_take)

        if current_text_parts:
            chunk_text = " ".join(current_text_parts).strip()
            result.append((chunk_text, sorted(current_pages)))

        return result

    def _get_text_within_token_limit(self, text: str, max_tokens: int) -> str:
        # binary search on character length to approximate token boundary
        low, high = 0, len(text)
        best = 0

        while low <= high:
            mid = (low + high) // 2
            sub = text[:mid]
            tokens = self._count_tokens(sub)

            if tokens <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best <= 0:
            return text[:1] if text else ""

        candidate = text[:best]

        # prefer sentence boundary near the end (last quarter)
        search_start = max(0, best - best // 4)
        window = candidate[search_start:]
        last_end = -1
        for punct in _SENTENCE_ENDINGS:
            pos = window.rfind(punct)
            if pos > last_end:
                last_end = pos

        if last_end > 0:
            cut = search_start + last_end + 1
            return text[:cut]

        return candidate

    def _get_overlap_text(self, text: str) -> str:
        if self.chunk_overlap_tokens <= 0:
            return ""

        total_tokens = self._count_tokens(text)
        if total_tokens <= self.chunk_overlap_tokens:
            return ""

        # binary search for start index whose suffix has >= overlap tokens
        low, high = 0, len(text)
        best_start = len(text)

        while low <= high:
            mid = (low + high) // 2
            suffix = text[mid:]
            tokens = self._count_tokens(suffix)

            if tokens >= self.chunk_overlap_tokens:
                best_start = mid
                low = mid + 1
            else:
                high = mid - 1

        if best_start >= len(text):
            return ""

        overlap_region = text[best_start:]

        # start overlap at a clean sentence boundary if possible
        sentence_start = -1
        for i in range(len(overlap_region) - 1):
            c = overlap_region[i]
            if c in _SENTENCE_ENDINGS and overlap_region[i + 1].isspace():
                sentence_start = i + 2

        if 0 < sentence_start < len(overlap_region):
            return overlap_region[sentence_start:]

        return overlap_region

    def _create_chunk(
        self,
        text: str,
        section: DocumentSection,
        pages: List[int],
        source_document: str,
        chunk_index: int,
        total_sub_chunks: int,
    ) -> DocumentChunk:
        start_page = min(pages) if pages else section.start_page
        end_page = max(pages) if pages else section.end_page

        metadata = {
            "ChunkType": "Bookmark",
            "BookmarkLevel": str(section.level),
        }
        if pages:
            metadata["Pages"] = ",".join(map(str, pages))

        # stable section hash (avoid Python's salted hash)
        heading_path = (section.full_heading_path or "").strip()
        section_hash = "nosection"
        if heading_path:
            section_hash = hashlib.sha1(heading_path.encode("utf-8")).hexdigest()[:10]

        chunk_id = f"{source_document}_bookmark_{section_hash}_chunk_{chunk_index}"

        return DocumentChunk(
            id=chunk_id,
            text=text,
            embedding=None,  # to be generated later
            source_document=source_document,
            start_page=start_page,
            end_page=end_page,
            chunk_index=chunk_index,
            chunk_total=total_sub_chunks,
            section=section.heading_text or source_document,
            section_path=section.full_heading_path or section.heading_text or source_document,
            metadata=metadata,
        )


def create_document_extractor(settings: EmbeddingSettings) -> BookmarkDocumentExtractor:
    return BookmarkDocumentExtractor(
        max_chunk_tokens=settings.max_chunk_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        tokenizer_model="gpt-4",  # match your C# intent
    )