# app/integrations/qdrant/mappers.py

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from qdrant_client.http import models as qm

from app.models.document_chunk import DocumentChunk


def chunk_to_point(chunk: DocumentChunk, *, vector: list[float] | None = None) -> qm.PointStruct:
    """
    Convert DocumentChunk -> Qdrant PointStruct (id + vector + payload).
    Mirrors C# ToPointStruct().
    """
    vec = vector if vector is not None else (chunk.embedding or [])
    payload: Dict[str, Any] = {
        "chunkId": chunk.id,
        "text": chunk.text,
        "sourceDocument": chunk.source_document,
        "chunkIndex": chunk.chunk_index,
        "chunkTotal": chunk.chunk_total,
        "startPage": chunk.start_page,
        "endPage": chunk.end_page,
        "section": chunk.section,
        "sectionPath": chunk.section_path,
        # In Python/Qdrant you can store Dict payload directly.
        # If you want to match your C# exactly (metadata JSON string), change to: json.dumps(chunk.metadata)
        "metadata": chunk.metadata or {},
    }

    return qm.PointStruct(
        id=str(uuid4()),  # Qdrant point id (separate from chunkId)
        vector=vec,
        payload=payload,
    )


def payload_to_chunk(payload: Dict[str, Any]) -> DocumentChunk:
    """
    Convert Qdrant payload -> DocumentChunk.
    Mirrors C# ToDocumentChunk().
    """
    # metadata may be Dict (recommended) or JSON string (if you chose to store it that way)
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, str):
        # optionally parse JSON string if you stored it that way
        import json
        try:
            metadata = json.loads(metadata) or {}
        except Exception:
            metadata = {}

    return DocumentChunk(
        id=str(payload.get("chunkId", "")),
        text=str(payload.get("text", "")),
        embedding=None,  # embeddings are typically not returned unless you request vectors
        source_document=str(payload.get("sourceDocument", "")),
        start_page=int(payload.get("startPage") or 0),
        end_page=int(payload.get("endPage") or 0),
        chunk_index=int(payload.get("chunkIndex") or 0),
        chunk_total=int(payload.get("chunkTotal") or 0),
        section=(payload.get("section") if payload.get("section") is not None else None),
        section_path=(payload.get("sectionPath") if payload.get("sectionPath") is not None else None),
        metadata=metadata,
    )