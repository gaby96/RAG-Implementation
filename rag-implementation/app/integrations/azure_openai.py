# app/integrations/embeddings/azure_openai.py

from __future__ import annotations

from typing import List, Optional

from openai import AsyncAzureOpenAI

from app.core.embedding_settings import EmbeddingSettings
from app.models.document_chunk import DocumentChunk


class AzureOpenAIEmbeddingGenerator:
    """
    Azure OpenAI implementation of embedding generation.
    Mirrors the .NET AzureOpenAIEmbeddingGenerator behavior.
    """

    def __init__(self, settings: EmbeddingSettings, client: Optional[AsyncAzureOpenAI] = None):
        self._deployment_name = settings.deployment_name

        self._client = client or AsyncAzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=getattr(settings, "api_version", "2024-02-01"),
        )

    async def generate_embedding(self, text: str) -> List[float]:
        resp = await self._client.embeddings.create(
            model=self._deployment_name, 
            input=text,
        )
        return list(resp.data[0].embedding)

    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        if not chunks:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = 50 

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            inputs = [c.generate_embedding_text() for c in batch]

            resp = await self._client.embeddings.create(
                model=self._deployment_name,
                input=inputs,
            )

            all_embeddings.extend([list(item.embedding) for item in resp.data])

            print(f"Created embeddings for {len(all_embeddings)}/{len(chunks)} chunks")

        return all_embeddings