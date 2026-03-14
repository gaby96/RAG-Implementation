# app/services/bm25_sparse_vectorizer.py

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import aiofiles
import asyncio

from app.services.text_tokenizer import TextTokenizer


@dataclass(slots=True)
class DocumentFrequencyData:
    total_documents: int
    total_document_length: int
    frequencies: Dict[int, int] = field(default_factory=Dict)  # term_hash(uint32)->df


@dataclass(slots=True)
class Bm25SparseVectorizer:
    """
    Computes BM25-weighted sparse vectors for text using hash-based token indices.
    Maintains document frequency statistics for IDF calculation.
    Mirrors your C# Bm25SparseVectorizer.
    """

    storage_path: Optional[str] = None
    k1: float = 1.2
    b: float = 0.75
    tokenizer: TextTokenizer = field(default_factory=TextTokenizer)

    # DF stats
    document_frequencies: Dict[int, int] = field(default_factory=Dict)  # term_hash -> df
    total_documents: int = 0
    total_document_length: int = 0

    # async lock for FastAPI concurrency
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def unique_term_count(self) -> int:
        return len(self.document_frequencies)

    @property
    def average_document_length(self) -> float:
        return (self.total_document_length / self.total_documents) if self.total_documents > 0 else 1.0

    async def load_document_frequencies(self) -> None:
        if not self.storage_path:
            return

        try:
            if not os.path.exists(self.storage_path):
                return

            async with aiofiles.open(self.storage_path, "r", encoding="utf-8") as f:
                raw = await f.read()

            if not raw.strip():
                return

            data = json.loads(raw)
            async with self._lock:
                self.total_documents = int(data.get("total_documents") or data.get("TotalDocuments") or 0)
                self.total_document_length = int(data.get("total_document_length") or data.get("TotalDocumentLength") or 0)

                freqs = data.get("frequencies") or data.get("Frequencies") or {}
                # keys may be strings in JSON
                self.document_frequencies = {int(k): int(v) for k, v in freqs.items()}

        except Exception as ex:
            # keep same “warn and continue” behavior
            print(f"Warning: Failed to load document frequencies: {ex}")

    async def save_document_frequencies(self) -> None:
        if not self.storage_path:
            return

        try:
            async with self._lock:
                data = {
                    "total_documents": self.total_documents,
                    "total_document_length": self.total_document_length,
                    "frequencies": self.document_frequencies,
                }

            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)

            async with aiofiles.open(self.storage_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data))

        except Exception as ex:
            print(f"Warning: Failed to save document frequencies: {ex}")

    async def clear(self) -> None:
        async with self._lock:
            self.document_frequencies.clear()
            self.total_documents = 0
            self.total_document_length = 0
        await self.save_document_frequencies()

    async def add_document(self, text: str) -> None:
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return

        # unique term hashes
        unique_hashes = {self.tokenizer.hash_term_to_index(t) for t in tokens}

        async with self._lock:
            self.total_documents += 1
            self.total_document_length += len(tokens)
            for h in unique_hashes:
                self.document_frequencies[h] = self.document_frequencies.get(h, 0) + 1

    async def add_documents(self, texts: Iterable[str]) -> None:
        for t in texts:
            await self.add_document(t)
        await self.save_document_frequencies()

    def compute_sparse_vector(self, text: str) -> Tuple[list[int], list[float]]:
        """
        Returns (indices, values) like C#: (uint[] Indices, float[] Values)
        - indices are uint32 hashes (stored as Python int)
        - values are BM25 weights
        """
        term_freqs, token_count = self.tokenizer.get_term_frequencies(text)
        if token_count == 0:
            return [], []

        avg_doc_len = self.average_document_length
        total_docs = max(1, self.total_documents)

        sparse_entries: Dict[int, float] = {}

        for term_hash, tf in term_freqs.items():
            df = self.document_frequencies.get(term_hash, 0)

            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log(((total_docs - df + 0.5) / (df + 0.5)) + 1.0)

            # TF normalized:
            # (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLen / avgDocLen)))
            denom = tf + self.k1 * (1.0 - self.b + self.b * (token_count / avg_doc_len))
            tf_norm = (tf * (self.k1 + 1.0)) / denom

            sparse_entries[term_hash] = float(idf * tf_norm)

        # sort by term hash for stable output
        items = sorted(sparse_entries.items(), key=lambda kv: kv[0])
        indices = [k for k, _ in items]
        values = [v for _, v in items]
        return indices, values