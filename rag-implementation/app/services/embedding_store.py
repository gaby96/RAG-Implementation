# app/integrations/fs/embedding_store.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Dict

import aiofiles

from app.models.document_chunk import DocumentChunk
from app.models.chunk_identifier import ChunkIdentifier


@dataclass(slots=True)
class FileSystemEmbeddingStore:
    """
    File system-based implementation of embedding storage.
    Stores all chunks in ./embeddings/index.json
    """

    storage_directory: str = "./embeddings"
    index_filename: str = "index.json"

    def __post_init__(self) -> None:
        os.makedirs(self.storage_directory, exist_ok=True)

    @property
    def _index_file_path(self) -> str:
        return os.path.join(self.storage_directory, self.index_filename)

    # --------------------------
    # Public API (like IEmbeddingStore)
    # --------------------------

    async def store_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        print(f"Storing {len(list(chunks))} chunks to file system embedding store...")
        existing = await self._load_all_chunks()

        # Upsert by id (remove then add)
        by_id: Dict[str, DocumentChunk] = {c.id: c for c in existing if c.id}
        for chunk in chunks:
            by_id[chunk.id] = chunk

        await self._save_all_chunks(list(by_id.values()))

    async def search(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[DocumentChunk]:
        chunks = await self._load_all_chunks()
        if not chunks:
            return []

        scored: list[tuple[float, DocumentChunk]] = []
        for c in chunks:
            if not c.embedding:
                continue
            sim = cosine_similarity(query_embedding, c.embedding)
            scored.append((sim, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    async def search_with_adjacent_chunks(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        adjacent_chunk_count: int = 1,
    ) -> list[DocumentChunk]:
        results = await self.search(query, query_embedding, top_k=top_k)

        if adjacent_chunk_count <= 0 or not results:
            return results

        return await self._expand_with_adjacent_chunks(results, adjacent_chunk_count)

    async def delete_document(self, document_name: str) -> None:
        chunks = await self._load_all_chunks()
        chunks = [c for c in chunks if c.source_document != document_name]
        await self._save_all_chunks(chunks)

    async def clear(self) -> None:
        await self._save_all_chunks([])

    async def get_chunks_by_document_and_indices(
        self,
        document_name: str,
        chunk_indices: Iterable[int],
        section_path: Optional[str] = None,
    ) -> list[DocumentChunk]:
        indices_set = set(chunk_indices)
        if not indices_set:
            return []

        chunks = await self._load_all_chunks()

        # NOTE: your C# method ignores sectionPath in filtering even though it’s passed.
        # If you want to match C# exactly, don’t filter by section_path.
        # If you want correctness (recommended), filter by section_path when provided.
        filtered = [
            c for c in chunks
            if c.source_document == document_name and c.chunk_index in indices_set
        ]
        if section_path is not None:
            filtered = [c for c in filtered if c.section_path == section_path]

        return filtered

    # --------------------------
    # Internal helpers
    # --------------------------

    async def _expand_with_adjacent_chunks(
        self,
        results: list[DocumentChunk],
        adjacent_chunk_count: int,
    ) -> list[DocumentChunk]:
        chunks_to_fetch: Set[ChunkIdentifier] = set()

        for r in results:
            for offset in range(-adjacent_chunk_count, adjacent_chunk_count + 1):
                if offset == 0:
                    continue

                adjacent_index = r.chunk_index + offset

                # bounds checks
                if adjacent_index < 0:
                    continue
                if r.chunk_total and (adjacent_index + 1) > r.chunk_total:
                    continue

                already_in_results = any(
                    e.source_document == r.source_document
                    and e.chunk_index == adjacent_index
                    and e.section_path == r.section_path
                    for e in results
                )

                if not already_in_results:
                    chunks_to_fetch.add(
                        ChunkIdentifier(
                            source_document=r.source_document,
                            section_path=r.section_path,
                            chunk_index=adjacent_index,
                        )
                    )

        if not chunks_to_fetch:
            return results

        expanded = list(results)

        # group by (source_document, section_path)
        grouped: Dict[tuple[str, Optional[str]], list[int]] = {}
        for ci in chunks_to_fetch:
            key = (ci.source_document, ci.section_path)
            grouped.setdefault(key, []).append(ci.chunk_index)

        for (source_doc, sec_path), indices in grouped.items():
            adjacent = await self.get_chunks_by_document_and_indices(
                document_name=source_doc,
                chunk_indices=indices,
                section_path=sec_path,
            )
            expanded.extend(adjacent)

        return expanded

    async def _load_all_chunks(self) -> list[DocumentChunk]:
        if not os.path.exists(self._index_file_path):
            return []

        async with aiofiles.open(self._index_file_path, "r", encoding="utf-8") as f:
            raw = await f.read()

        if not raw.strip():
            return []

        data = json.loads(raw)

        # Expecting a list[Dict] representing DocumentChunk
        # If DocumentChunk is pydantic, prefer model_validate
        chunks: list[DocumentChunk] = []
        for item in data:
            try:
                chunks.append(DocumentChunk.model_validate(item))  # pydantic v2
            except Exception:
                # fallback if you’re on pydantic v1 or dataclass-like input
                chunks.append(DocumentChunk(**item))  # type: ignore[arg-type]

        return chunks

    async def _save_all_chunks(self, chunks: list[DocumentChunk]) -> None:
        # Serialize to JSON (camelCase isn’t needed unless you require it)
        payload = []
        for c in chunks:
            try:
                payload.append(c.model_dump())  # pydantic v2
            except Exception:
                payload.append(c.Dict())  # pydantic v1

        text = json.dumps(payload, indent=2, ensure_ascii=False)

        async with aiofiles.open(self._index_file_path, "w", encoding="utf-8") as f:
            await f.write(text)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / ((norm1 ** 0.5) * (norm2 ** 0.5))