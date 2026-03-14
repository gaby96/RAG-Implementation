# app/integrations/qdrant/hybrid_embedding_store.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.models.document_chunk import DocumentChunk
from app.models.chunk_identifier import ChunkIdentifier
from app.integrations.qdrant.mappers import payload_to_chunk
from app.services.bm25_sparse_vectorizer import Bm25SparseVectorizer
import logging
logger = logging.getLogger(__name__)


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@dataclass(slots=True)
class QdrantHybridEmbeddingStore:
    """
    Qdrant hybrid embedding store: dense vectors + BM25 sparse vectors with RRF fusion.
    Equivalent to .NET QdrantHybridEmbeddingStore.
    """

    client: AsyncQdrantClient
    collection_name: str
    dense_vector_size: int
    bm25: Bm25SparseVectorizer
    dense_weight: float = 0.7  # influences prefetch sizes
    sparse_weight: float = 0.3

    async def ensure_collection(self) -> None:
        try:
            cols = await self.client.get_collections()
            exists = any(c.name == self.collection_name for c in cols.collections)

            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: qm.VectorParams(
                            size=self.dense_vector_size,
                            distance=qm.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: qm.SparseVectorParams()
                    },
                )

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

        except Exception as ex:
            raise RuntimeError(
                f"Failed to initialize Qdrant hybrid collection '{self.collection_name}': {ex}"
            ) from ex

    # --------------------------
    # IEmbeddingStore methods
    # --------------------------

    async def store_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return

        # Update DF stats for BM25 using chunk text
        await self.bm25.add_documents([c.text for c in chunk_list])

        points: list[qm.PointStruct] = []
        for c in chunk_list:
            if not c.embedding:
                # If you want strict parity with C#, raise here.
                # But many pipelines embed before store; this protects you.
                continue

            sparse = self.bm25.compute_sparse_vector(c.text)

            payload = {
                "chunkId": c.id,
                "text": c.text,
                "sourceDocument": c.source_document,
                "chunkIndex": c.chunk_index,
                "chunkTotal": c.chunk_total,
                "startPage": c.start_page,
                "endPage": c.end_page,
                "section": c.section,
                "sectionPath": c.section_path,
                "metadata": c.metadata or {},
            }

            points.append(
                qm.PointStruct(
                    id=c.id,  # you can use chunk.id as the point id (simpler than random uuid)
                    payload=payload,
                    vector={
                        DENSE_VECTOR_NAME: c.embedding,
                        SPARSE_VECTOR_NAME: qm.SparseVector(indices=sparse.indices, values=sparse.values),
                    },
                )
            )

        if not points:
            return

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def search(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[DocumentChunk]:
        logger.info(f"Performing dense search with query embedding of length {len(query_embedding)}")
        return await self.search_hybrid(query_embedding=query_embedding, query_text=query, top_k=top_k)

    async def search_hybrid(self, query_embedding: list[float], query_text: str, top_k: int = 5) -> list[DocumentChunk]:

        logger.info(f"Performing hybrid search with query: '{query_text}' and embedding of length {len(query_embedding)}")
        # Compute query sparse vector
        sparse = self.bm25.compute_sparse_vector(query_text)

        # Prefetch sizes weighted (same idea as your C#)
        total = max(1e-6, (self.dense_weight + self.sparse_weight))
        dense_prefetch = max(top_k, int(top_k * 4 * (self.dense_weight / total)))
        sparse_prefetch = max(top_k, int(top_k * 4 * (self.sparse_weight / total)))

        # Hybrid query with RRF fusion
        #
        # qdrant-client supports Query API via `query_points` in newer versions.
        # If your installed version uses a different name, tell me your qdrant-client version.
        res = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                qm.Prefetch(
                    query=qm.NearestQuery(
                        nearest=qm.NamedVector(
                            name=DENSE_VECTOR_NAME,
                            vector=query_embedding,
                        )
                    ),
                    limit=dense_prefetch,
                ),
                qm.Prefetch(
                    query=qm.NearestQuery(
                        nearest=qm.NamedSparseVector(
                            name=SPARSE_VECTOR_NAME,
                            vector=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                        )
                    ),
                    limit=sparse_prefetch,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        # res.points is typical; some versions return directly a list
        points = getattr(res, "points", res)
        return [payload_to_chunk(p.payload or {}) for p in points]

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
        # Note: BM25 DF stats are not decremented (same as your C# note)

    async def clear(self) -> None:
        await self.client.delete_collection(collection_name=self.collection_name)
        await self.bm25.clear()
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

        out: list[DocumentChunk] = []

        for idx in indices:
            must: list[qm.Condition] = [
                qm.FieldCondition(key="sourceDocument", match=qm.MatchValue(value=document_name)),
                qm.FieldCondition(key="chunkIndex", match=qm.MatchValue(value=idx)),
            ]
            if section_path:
                must.append(qm.FieldCondition(key="sectionPath", match=qm.MatchValue(value=section_path)))

            pts, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qm.Filter(must=must),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            for p in pts:
                out.append(payload_to_chunk(p.payload or {}))

        return out

    # --------------------------
    # Adjacent expansion (same logic)
    # --------------------------

    async def _expand_with_adjacent_chunks(
        self,
        results: list[DocumentChunk],
        adjacent_chunk_count: int,
    ) -> list[DocumentChunk]:
        to_fetch: Set[ChunkIdentifier] = set()

        for r in results:
            for offset in range(-adjacent_chunk_count, adjacent_chunk_count + 1):
                if offset == 0:
                    continue

                adj = r.chunk_index + offset
                if adj < 0:
                    continue
                if r.chunk_total and (adj + 1) > r.chunk_total:
                    continue

                already = any(
                    e.source_document == r.source_document
                    and e.chunk_index == adj
                    and e.section_path == r.section_path
                    for e in results
                )
                if not already:
                    to_fetch.add(
                        ChunkIdentifier(
                            source_document=r.source_document,
                            section_path=r.section_path,
                            chunk_index=adj,
                        )
                    )

        if not to_fetch:
            return results

        expanded = list(results)

        grouped: Dict[Tuple[str, Optional[str]], list[int]] = {}
        for ci in to_fetch:
            grouped.setdefault((ci.source_document, ci.section_path), []).append(ci.chunk_index)

        for (doc, sec), idxs in grouped.items():
            expanded.extend(
                await self.get_chunks_by_document_and_indices(
                    document_name=doc,
                    chunk_indices=idxs,
                    section_path=sec,
                )
            )

        return expanded