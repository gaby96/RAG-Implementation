# app/integrations/embeddings/http_generator.py

from typing import List
import httpx
from pydantic import BaseModel

from app.models.document_chunk import DocumentChunk


class EmbeddingRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int


class HttpEmbeddingGenerator:
    """
    HTTP implementation of embedding generation.
    Designed for FastAPI dependency injection.
    """

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self._client = client
        self._base_url = base_url.rstrip("/") + "/"

    async def generate_embedding(self, text: str) -> List[float]:
        req = EmbeddingRequest(texts=[text])

        resp = await self._client.post(
            f"{self._base_url}embed",
            json=req.model_dump(),
        )
        resp.raise_for_status()

        data = EmbeddingResponse(**resp.json())
        return data.embeddings[0]

    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        if not chunks:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = 50

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.generate_embedding_text() for c in batch]

            req = EmbeddingRequest(texts=texts)

            resp = await self._client.post(
                f"{self._base_url}embed",
                json=req.model_dump(),
            )
            resp.raise_for_status()

            data = EmbeddingResponse(**resp.json())
            all_embeddings.extend(data.embeddings)

            print(f"Created embeddings for {len(all_embeddings)}/{len(chunks)} chunks")

        return all_embeddings