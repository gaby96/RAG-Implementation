# app/integrations/qdrant/hybrid_idf_embedding_store.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple
import uuid
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.models.document_chunk import DocumentChunk
from app.models.chunk_identifier import ChunkIdentifier
from app.integrations.qdrant.mappers import payload_to_chunk
from app.integrations.qdrant.tf_sparse import TfSparseVectorizer
import logging
logger = logging.getLogger(__name__)


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@dataclass(slots=True)
class QdrantHybridIdfEmbeddingStore:
    """
    Hybrid search store using:
      - dense named vectors (cosine)
      - sparse named vectors with modifier=IDF (Qdrant applies IDF at query time)
      - RRF fusion over prefetch results
    Mirrors the .NET QdrantHybridIdfEmbeddingStore.
    """

    client: AsyncQdrantClient
    collection_name: str
    dense_vector_size: int

    tf_vectorizer: TfSparseVectorizer

    dense_weight: float = 0.7
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
                        SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                            modifier=qm.Modifier.IDF  # <-- the key difference
                        )
                    },
                )

                # payload indexes (for filtering / scroll)
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
        logger.info(f"📦 Processing {len(chunk_list)} chunks")
        if not chunk_list:
            logger.warning("⚠️  No chunks to process")
            return

        points: list[qm.PointStruct] = []

        for c in chunk_list:
            if not c.embedding:
                logger.warning(f"⚠️  Chunk {c.id} has no embedding, skipping")
                continue

            # TF-only sparse vector (Qdrant applies IDF)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c.id))
            logger.debug(f"Converting chunk ID '{c.id}' → point ID '{point_id}'")
            indices, values = self.tf_vectorizer.compute_tf_sparse_vector(c.text)
            logger.debug(f"📊 TF sparse vector for chunk {c.id}: {len(indices)} indices")
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
            try:
                points.append(
                    qm.PointStruct(
                        id=point_id,
                        payload=payload,
                        vector={
                            DENSE_VECTOR_NAME: c.embedding,
                            SPARSE_VECTOR_NAME: qm.SparseVector(indices=indices, values=values),
                        },
                    )
                )
            except Exception as ex:
                logger.error(f"❌ Failed to create point for chunk {c.id}: {type(ex).__name__}: {ex}", exc_info=True)
                raise

        logger.info(f"📍 Created {len(points)} points from {len(chunk_list)} chunks")
        
        if not points:
            logger.warning("⚠️  No valid points to upsert")
            return

        if points:
            first_point = points[0]
            logger.info(f"First point ID: {first_point.id}")
            logger.info(f"First point payload keys: {list(first_point.payload.keys())}")
            logger.info(f"First point vector keys: {list(first_point.vector.keys())}")
        
            # Check sparse vector structure
            for vec_name, vec_value in first_point.vector.items():
                logger.info(f"Vector '{vec_name}': type={type(vec_value).__name__}")
                if hasattr(vec_value, 'indices'):
                    logger.info(f"  - Sparse vector indices: {vec_value.indices[:10] if len(vec_value.indices) > 10 else vec_value.indices}")
                    logger.info(f"  - Sparse vector values: {vec_value.values[:10] if len(vec_value.values) > 10 else vec_value.values}")
                else:
                    logger.info(f"  - Dense vector length: {len(vec_value)}")
    
        # Batch upsert
        batch_size = 10
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            logger.info(f"🚀 Upserting batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} ({len(batch)} points)...")
        
            try:
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                logger.info(f"✅ Batch {i//batch_size + 1} upserted successfully")
            except Exception as ex:
                logger.error(f"❌ Batch {i//batch_size + 1} failed: {type(ex).__name__}: {str(ex)}", exc_info=True)
                # Try to get response body
                if hasattr(ex, '__cause__') and hasattr(ex.__cause__, 'response'):
                    logger.error(f"Response: {ex.__cause__.response}")
                raise RuntimeError(f"Failed to store chunks in Qdrant: {type(ex).__name__}: {str(ex)}") from ex

        logger.info(f"✅ All {len(points)} points successfully upserted to Qdrant")

    async def search(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[DocumentChunk]:
        """Search using dense vector."""
        
        logger.info(f"🔍 Searching for: '{query}'")
        
        try:
            # Use HTTP search API with named vector
            res = await self.client.http.search_api.search_points(
                collection_name=self.collection_name,
                search_request=qm.SearchRequest(
                    vector=qm.NamedVector(
                        name=DENSE_VECTOR_NAME,
                        vector=query_embedding,
                    ),
                    limit=top_k,
                    with_payload=True,
                ),
            )

            points = res.result if hasattr(res, 'result') else res
            logger.info(f"✅ Search returned {len(points)} results")
            return [payload_to_chunk(p.payload or {}) for p in points]
            
        except Exception as ex:
            logger.error(f"❌ Search failed: {type(ex).__name__}: {str(ex)}", exc_info=True)
            raise RuntimeError(f"Search failed: {type(ex).__name__}: {str(ex)}") from ex

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

    # --------------------------
    # Adjacent expansion
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