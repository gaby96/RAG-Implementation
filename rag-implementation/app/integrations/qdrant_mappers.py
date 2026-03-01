from __future__ import annotations
import json
from typing import Any, dict, Optional

from qdrant_client.models import PointStruct

from app.models.document_chunk import DocumentChunk

def chunk_to_point(chunk: DocumentChunk, *, point_id: Optional[str] = None) -> PointStruct:
    """Convert a DocumentChunk to a Qdrant PointStruct."""

    pid = point_id or chunk.id

    if chunk.embedding is None:
        raise ValueError("Chunk must have an embedding to be converted to a Qdrant point.")
    
    payload = {
        "chunkId": chunk.id,
        "text": chunk.text,
        "source_document": chunk.source_document,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk.chunk_total,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "section": chunk.section,
        "section_path": chunk.section_path,
        "metadata": chunk.metadata,
    }
    
    return PointStruct(
        id=pid,
        vector=chunk.embedding,
        payload=payload
    )

def payload_to_chunk(payload: dict[str, Any]) -> DocumentChunk:
    """Convert a Qdrant payload back to a DocumentChunk."""

    md = payload.get("metadata") or {}

    if isinstance(md, dict):
        # If metadata is a dict, we can merge it with the main payload
        md = json.loads(md) or {}

    return DocumentChunk(
        id=payload.get("chunkId", ""),
        text=payload.get("text", ""),
        source_document=payload.get("source_document", ""),
        chunk_index=payload.get("chunk_index", 0),
        chunk_total=payload.get("chunk_total", 0),
        start_page=payload.get("start_page"),
        end_page=payload.get("end_page"),
        section=payload.get("section"),
        section_path=payload.get("section_path"),
        embedding=None,
        metadata={str(k): str(v) for k, v in md.items()}  # Ensure metadata values are strings
    )

def _opt_int(value: Any) -> Optional[int]:
    """Helper to convert a value to an optional int."""
    return None if value is None else int(value)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    return s if s else None