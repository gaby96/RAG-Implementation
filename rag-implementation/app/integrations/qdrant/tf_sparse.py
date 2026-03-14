from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from app.services.text_tokenizer import TextTokenizer


@dataclass(slots=True)
class TfSparseVectorizer:
    """
    Computes TF-only BM25 weights (no IDF).
    IDF is applied by Qdrant at query time when sparse vector modifier is IDF.
    """

    tokenizer: TextTokenizer
    k1: float = 1.2
    b: float = 0.75
    avg_doc_len: float = 512.0  # you used MaxChunkTokens as proxy

    def compute_tf_sparse_vector(self, text: str) -> Tuple[list[int], list[float]]:
        term_freqs, token_count = self.tokenizer.get_term_frequencies(text)
        if token_count == 0:
            return [], []

        sparse = {}
        for term_hash, tf in term_freqs.items():
            # TF component:
            # (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLen / avgDocLen)))
            denom = tf + self.k1 * (1.0 - self.b + self.b * (token_count / self.avg_doc_len))
            tf_norm = (tf * (self.k1 + 1.0)) / denom
            sparse[term_hash] = float(tf_norm)

        items = sorted(sparse.items(), key=lambda kv: kv[0])
        indices = [k for k, _ in items]
        values = [v for _, v in items]
        return indices, values