from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import httpx

from app.models.document_chunk import DocumentChunk
from app.models.reranked_result import RerankedResult
from app.services.interfaces.rerank_interface import Reranker


@dataclass(slots=True)
class CrossEncoderReranker(Reranker):
    """
    Equivalent of the C# CrossEncoderReranker.
    Expects a reranker HTTP service:
      POST {endpoint}rerank
      body: { "query": "...", "documents": ["...", "..."] }
      resp: { "scores": [0.0..1.0] }
    """
    endpoint: str  # e.g. "http://localhost:8001/" or "https://.../"
    api_key: str | None = None
    timeout_s: float = 60.0

    async def rerank(self, query: str, results: list[DocumentChunk], top_k: int = 5) -> list[RerankedResult]:
        if not results:
            return []

        # Ensure endpoint ends with /
        base = self.endpoint if self.endpoint.endswith("/") else f"{self.endpoint}/"
        url = f"{base}rerank"

        payload = {
            "query": query,
            "documents": [c.generate_embedding_text() for c in results],
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # adjust if your reranker expects a different header
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        scores = data.get("scores")
        if not isinstance(scores, list) or len(scores) != len(results):
            # defensive fallback: keep original order with neutral scores
            out = []
            for i, chunk in enumerate(results[:top_k]):
                out.append(
                    RerankedResult(
                        chunk=chunk,
                        relevance_score=5.0 - (i * 0.1),
                        original_rank=i + 1,
                        new_rank=i + 1,
                    )
                )
            return out

        reranked = [
            RerankedResult(
                chunk=results[i],
                relevance_score=float(scores[i]) * 10.0,
                original_rank=i + 1,
                new_rank=0,
            )
            for i in range(len(results))
        ]

        reranked.sort(key=lambda r: r.relevance_score, reverse=True)

        for i, r in enumerate(reranked):
            r.new_rank = i + 1

        return reranked[:top_k]