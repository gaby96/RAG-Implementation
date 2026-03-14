from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Optional, Set, Tuple

import tiktoken

from app.models.document_chunk import DocumentChunk
from app.models.document_section import DocumentSection, PageText
from app.services.interfaces.extraction_strategy import DocumentExtractor as DocumentExtractor
from app.services.interfaces.extraction_strategy import ExtractionStrategy


_SENTENCE_ENDINGS = {".", "!", "?"}


@dataclass(slots=True)
class DocumentExtractor(DocumentExtractor):
    max_chunk_tokens: int
    chunk_overlap_tokens: int
    model_name: str = "gpt-4"

    def __post_init__(self) -> None:
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be greater than 0")
        if self.chunk_overlap_tokens < 0 or self.chunk_overlap_tokens >= self.max_chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be between 0 and max_chunk_tokens")

        # Similar spirit to C# TiktokenTokenizer.CreateForModel("gpt-4")
        # If the exact model isn't known to tiktoken, fallback to cl100k_base.
        try:
            self._enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    # --------------------------
    # Public API (interface)
    # --------------------------

    async def process_document(
        self,
        file_path: str,
        extraction_strategy: ExtractionStrategy,
    ) -> list[DocumentChunk]:
        file_name = Path(file_path).name
        sections = await self._extract_sections(file_path, extraction_strategy)
        return self._create_chunks_from_sections(sections, file_name)

    async def process_documents(
        self,
        file_paths: Iterable[str],
        extraction_strategy: ExtractionStrategy,
    ) -> Dict[str, list[DocumentChunk]]:
        results: Dict[str, list[DocumentChunk]] = {}
        for fp in file_paths:
            file_name = Path(fp).name
            chunks = await self.process_document(fp, extraction_strategy)
            results[file_name] = chunks
        return results

    # --------------------------
    # Internals
    # --------------------------

    async def _extract_sections(
        self,
        file_path: str,
        extraction_strategy: ExtractionStrategy,
    ) -> list[DocumentSection]:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {p.resolve()}")

        # ExtractionStrategy expects a stream
        with p.open("rb") as f:
            return await extraction_strategy.extract(f)

    def _create_chunks_from_sections(self, sections: list[DocumentSection], source_document: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for section in sections:
            chunks.extend(self._create_chunks_from_section(section, source_document))
        return chunks

    def _create_chunks_from_section(self, section: DocumentSection, source_document: str) -> list[DocumentChunk]:
        if not section.page_texts:
            return []

        full_text = section.text or ""
        if not full_text.strip():
            return []

        token_count = self._count_tokens(full_text)
        if token_count <= self.max_chunk_tokens:
            pages = [p.page_number for p in section.page_texts]
            return [
                self._create_chunk(
                    text=full_text,
                    section=section,
                    pages=pages,
                    source_document=source_document,
                    chunk_index=0,
                    total_subchunks=1,
                )
            ]

        subchunks = self._split_large_section_with_pages(section)
        out: list[DocumentChunk] = []
        for i, (text, pages) in enumerate(subchunks):
            out.append(
                self._create_chunk(
                    text=text,
                    section=section,
                    pages=pages,
                    source_document=source_document,
                    chunk_index=i,
                    total_subchunks=len(subchunks),
                )
            )
        return out

    def _split_large_section_with_pages(self, section: DocumentSection) -> list[Tuple[str, list[int]]]:
        result: list[Tuple[str, list[int]]] = []

        current_text_parts: list[str] = []
        current_pages: Set[int] = set()
        current_token_count = 0

        for page_text in section.page_texts:
            page_content = page_text.text or ""
            page_content_index = 0

            while page_content_index < len(page_content):
                remaining = page_content[page_content_index:]
                space_needed = 1 if current_text_parts else 0
                available_tokens = self.max_chunk_tokens - current_token_count - space_needed

                if available_tokens <= 0:
                    # flush current chunk
                    if current_text_parts:
                        chunk_text = " ".join(current_text_parts).strip()
                        result.append((chunk_text, sorted(current_pages)))

                        overlap = self._get_overlap_text(chunk_text)
                        last_page = max(current_pages) if current_pages else None

                        current_text_parts = []
                        current_pages = set()

                        if overlap:
                            current_text_parts = [overlap]
                            current_token_count = self._count_tokens(overlap)
                            if last_page is not None:
                                current_pages.add(last_page)
                        else:
                            current_token_count = 0
                    continue

                remaining_tokens = self._count_tokens(remaining)

                if remaining_tokens <= available_tokens:
                    # remaining fits fully
                    if current_text_parts:
                        current_token_count += 1  # for the join space
                    current_text_parts.append(remaining)
                    current_token_count += remaining_tokens
                    current_pages.add(page_text.page_number)
                    break
                else:
                    # take what fits, prefer sentence boundary
                    text_to_take = self._get_text_within_token_limit(remaining, available_tokens)

                    if current_text_parts:
                        current_token_count += 1
                    current_text_parts.append(text_to_take)
                    current_pages.add(page_text.page_number)

                    # flush
                    chunk_text = " ".join(current_text_parts).strip()
                    result.append((chunk_text, sorted(current_pages)))

                    overlap = self._get_overlap_text(chunk_text)
                    last_page = max(current_pages) if current_pages else None

                    current_text_parts = []
                    current_pages = set()

                    if overlap:
                        current_text_parts = [overlap]
                        current_token_count = self._count_tokens(overlap)
                        if last_page is not None:
                            current_pages.add(last_page)
                    else:
                        current_token_count = 0

                    page_content_index += len(text_to_take)

        # final chunk
        if current_text_parts:
            chunk_text = " ".join(current_text_parts).strip()
            result.append((chunk_text, sorted(current_pages)))

        return result

    def _get_text_within_token_limit(self, text: str, max_tokens: int) -> str:
        # Binary search best character cut within token limit
        low, high = 0, len(text)
        best_fit = 0

        while low <= high:
            mid = (low + high) // 2
            sub = text[:mid]
            tok = self._count_tokens(sub)
            if tok <= max_tokens:
                best_fit = mid
                low = mid + 1
            else:
                high = mid - 1

        if best_fit == 0:
            return text[:1] if text else ""

        text_to_search = text[:best_fit]

        # find sentence ending within last quarter
        search_start = max(0, best_fit - best_fit // 4)
        last_sentence_end = -1
        for i in range(best_fit - 1, search_start - 1, -1):
            if text_to_search[i] in _SENTENCE_ENDINGS:
                last_sentence_end = i
                break

        if last_sentence_end > 0:
            return text[: last_sentence_end + 1]

        return text_to_search

    def _get_overlap_text(self, text: str) -> str:
        if self.chunk_overlap_tokens <= 0:
            return ""

        total_tokens = self._count_tokens(text)
        if total_tokens <= self.chunk_overlap_tokens:
            return ""

        # binary search start position from which suffix has >= overlap_tokens
        low, high = 0, len(text)
        best_start = len(text)

        while low <= high:
            mid = (low + high) // 2
            suffix = text[mid:]
            tok = self._count_tokens(suffix)
            if tok >= self.chunk_overlap_tokens:
                best_start = mid
                low = mid + 1
            else:
                high = mid - 1

        if best_start >= len(text):
            return ""

        overlap_region = text[best_start:]

        # try to start at clean sentence boundary
        sentence_start = -1
        for i in range(0, len(overlap_region) - 1):
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
        pages: list[int],
        source_document: str,
        chunk_index: int,
        total_subchunks: int,
    ) -> DocumentChunk:
        start_page = min(pages) if pages else section.start_page
        end_page = max(pages) if pages else section.end_page

        metadata: Dict[str, str] = {
            "ChunkType": "Bookmark",
            "BookmarkLevel": str(section.level),
        }
        if pages:
            metadata["Pages"] = ",".join(str(p) for p in pages)

        full_path = section.full_heading_path or ""
        section_hash = self._stable_hash(full_path) if full_path else "nosection"

        chunk_id = f"{source_document}_bookmark_{section_hash}_chunk_{chunk_index}"

        return DocumentChunk(
            id=chunk_id,
            text=text,
            embedding=[],
            source_document=source_document,
            start_page=start_page,
            end_page=end_page,
            chunk_index=chunk_index,
            chunk_total=total_subchunks,
            section=(section.heading_text or source_document),
            section_path=(section.full_heading_path or section.heading_text or source_document),
            metadata=metadata,
        )

    # --------------------------
    # Helpers
    # --------------------------

    def _count_tokens(self, text: str) -> int:
        # tiktoken: token count = len(encode)
        return len(self._enc.encode(text or ""))

    @staticmethod
    def _stable_hash(value: str) -> str:
        # deterministic across processes/machines
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]