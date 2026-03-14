# app/integrations/qdrant/embedding_store.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Set, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.models.document_chunk import DocumentChunk
from app.models.chunk_identifier import ChunkIdentifier
from app.integrations.qdrant.mappers import chunk_to_point, payload_to_chunk
import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QdrantEmbeddingStore:
    """
    Qdrant-based embedding store (vector search).
    Equivalent to the .NET QdrantEmbeddingStore.
    """

    client: AsyncQdrantClient
    collection_name: str
    vector_size: int

    async def ensure_collection(self) -> None:
        """
        Equivalent to InitializeCollectionAsync() in C#.
        Creates the collection + payload indexes if missing.
        """
        try:
            logger.info("🔍 Checking Qdrant connection...")
            cols = await self.client.get_collections()
            logger.info(f"✅ Connected to Qdrant! Collections: {[c.name for c in cols.collections]}")
            exists = any(c.name == self.collection_name for c in cols.collections)
            logger.info(f"📦 Collection '{self.collection_name}' exists: {exists}")

            if not exists:
                logger.info(f"📝 Creating collection '{self.collection_name}'...")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qm.VectorParams(
                        size=self.vector_size,
                        distance=qm.Distance.COSINE,
                    ),
                )
                logger.info("✅ Collection created successfully")

                # payload indexes for filtering
                logger.info("📑 Creating payload indexes...")
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="sourceDocument",
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="sectionPath",
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="chunkIndex",
                    field_schema=qm.PayloadSchemaType.INTEGER,
                )
                logger.info("✅ Payload indexes created")

        except Exception as ex:
            logger.error(f"❌ Qdrant connection failed: {ex}")
            raise RuntimeError(
                f"Failed to initialize Qdrant collection '{self.collection_name}': {ex}"
            ) from ex

    # --------------------------
    # IEmbeddingStore methods
    # --------------------------

    async def store_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        print("This one was called")
        chunk_list = list(chunks)
        logger.info(f"💾 Storing {len(chunk_list)} chunks to Qdrant...")
        if not chunk_list:
            logger.warning("⚠️  No chunks to store")
            return

        points = [chunk_to_point(c) for c in chunk_list]
        logger.info(f"📍 Created {len(points)} points")
        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except Exception as ex:
            logger.error(f"❌ Failed to store chunks in Qdrant: {ex}")
            raise RuntimeError(f"Failed to store chunks in Qdrant: {ex}") from ex

    async def search(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[DocumentChunk]:
        res = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [payload_to_chunk(p.payload or {}) for p in res]

    async def delete_document(self, document_name: str) -> None:
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="sourceDocument",
                        match=qm.MatchValue(value=document_name),
                    )
                ]
            ),
        )

    async def clear(self) -> None:
        # Drop and recreate
        await self.client.delete_collection(collection_name=self.collection_name)
        await self.ensure_collection()

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

    async def get_chunks_by_document_and_indices(
        self,
        document_name: str,
        chunk_indices: Iterable[int],
        section_path: Optional[str] = None,
    ) -> list[DocumentChunk]:
        indices = list(chunk_indices)
        if not indices:
            return []

        all_chunks: list[DocumentChunk] = []

        # Mirrors your C# approach: query each index separately (simple & reliable)
        for idx in indices:
            must: list[qm.Condition] = [
                qm.FieldCondition(key="sourceDocument", match=qm.MatchValue(value=document_name)),
                qm.FieldCondition(key="chunkIndex", match=qm.MatchValue(value=idx)),
            ]

            if section_path:
                must.append(qm.FieldCondition(key="sectionPath", match=qm.MatchValue(value=section_path)))

            scroll_res, _next = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qm.Filter(must=must),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            for p in scroll_res:
                all_chunks.append(payload_to_chunk(p.payload or {}))

        return all_chunks

    # --------------------------
    # Adjacent expansion
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
        grouped: Dict[Tuple[str, Optional[str]], list[int]] = {}
        for ci in chunks_to_fetch:
            grouped.setdefault((ci.source_document, ci.section_path), []).append(ci.chunk_index)

        for (doc, sec), indices in grouped.items():
            adjacent = await self.get_chunks_by_document_and_indices(
                document_name=doc,
                chunk_indices=indices,
                section_path=sec,
            )
            expanded.extend(adjacent)

        return expanded