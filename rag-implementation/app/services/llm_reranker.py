from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, TypedDict

from openai import AsyncOpenAI

from app.models.document_chunk import DocumentChunk
from app.models.reranked_result import RerankedResult
from app.services.interfaces.rerank_interface import Reranker


class _ScoreItem(TypedDict, total=False):
    id: int
    score: float
    explanation: str


@dataclass(slots=True)
class LlmReranker(Reranker):
    client: AsyncOpenAI
    model: str 
    minimum_relevance_score: float = 5.0
    request_timeout_s: float = 60.0

    async def rerank(self, query: str, results: list[DocumentChunk], top_k: int = 5) -> list[RerankedResult]:
        if not results:
            return []

        system_prompt = (
            "You are a relevance scoring assistant. Your task is to evaluate how relevant each document chunk is to a given query.\n\n"
            "You must respond with ONLY a JSON array in the following format:\n"
            "[\n"
            '  { "id": 0, "score": <number between 0 and 10> },\n'
            '  { "id": 1, "score": <number between 0 and 10> },\n'
            "  ...\n"
            "]\n\n"
            "Scoring guidelines:\n"
            "- 0-2: Not relevant at all, the content does not address the query\n"
            "- 3-4: Slightly relevant, tangentially related to the query\n"
            "- 5-6: Moderately relevant, partially addresses the query\n"
            "- 7-8: Highly relevant, directly addresses the query\n"
            "- 9-10: Perfectly relevant, comprehensive answer to the query\n\n"
            "Be strict and objective in your scoring. Focus on semantic relevance, not just keyword matches.\n"
            "Return scores for ALL chunks in the same order they were provided."
        )

        chunks_text = "\n\n".join(
            f"Chunk: {idx}\n{chunk.generate_embedding_text()}"
            for idx, chunk in enumerate(results)
        )

        user_prompt = (
            f"Query: {query}\n\n"
            "Document chunks to evaluate:\n\n"
            f"{chunks_text}\n\n"
            f"Evaluate the relevance of each document chunk to the query. "
            f"Return a JSON array with scores for all {len(results)} chunks."
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                # Keep it high like your C# (20k). Actual allowed max depends on the model.
                max_tokens=4000,
                timeout=self.request_timeout_s,
            )

            content = (resp.choices[0].message.content or "").strip()
            scores = self._parse_batch_score_response(content, expected_count=len(results))

            reranked = [
                RerankedResult(
                    chunk=chunk,
                    relevance_score=float(scores[i]["score"]),
                    original_rank=i + 1,
                    new_rank=0,
                )
                for i, chunk in enumerate(results)
            ]

            filtered_sorted = [
                r for r in sorted(reranked, key=lambda x: x.relevance_score, reverse=True)
                if r.relevance_score >= self.minimum_relevance_score
            ][: top_k]

            for i, r in enumerate(filtered_sorted):
                r.new_rank = i + 1

            return filtered_sorted

        except Exception as ex:
            # Same idea as your C#: log + fallback
            print(f"Warning: Failed to rerank chunks: {ex}")

            fallback = []
            for i, chunk in enumerate(results[:top_k]):
                fallback.append(
                    RerankedResult(
                        chunk=chunk,
                        relevance_score=5.0 - (i * 0.1),
                        original_rank=i + 1,
                        new_rank=i + 1,
                    )
                )
            return fallback

    # --------------------------
    # Parsing helpers
    # --------------------------

    def _parse_batch_score_response(self, content: str, expected_count: int) -> list[_ScoreItem]:
        cleaned = content.strip()

        # Handle markdown fences ```json ... ```
        if cleaned.startswith("```"):
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if 0 <= start < end:
                cleaned = cleaned[start : end + 1]

        try:
            data = json.loads(cleaned)
            if isinstance(data, list) and len(data) == expected_count:
                # validate shape + clamp
                out: list[_ScoreItem] = []
                for i, item in enumerate(data):
                    score = float(item.get("score", 5.0)) if isinstance(item, Dict) else 5.0
                    score = max(0.0, min(10.0, score))
                    out.append({"id": i, "score": score})
                return out
            return self._default_scores(expected_count)
        except Exception:
            return self._default_scores(expected_count)

    @staticmethod
    def _default_scores(count: int) -> list[_ScoreItem]:
        return [{"id": i, "score": 5.0, "explanation": "Could not parse response"} for i in range(count)]