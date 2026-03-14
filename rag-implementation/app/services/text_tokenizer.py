# app/services/text_tokenizer.py

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Dict, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class TextTokenizer:
    """
    Simple tokenizer + deterministic 32-bit hashing for term -> uint index.
    Mirrors the idea of HashTermToIndex() in your C#.
    """

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return [t.lower() for t in _TOKEN_RE.findall(text)]

    def hash_term_to_index(self, term: str) -> int:
        """
        Deterministic uint32 hash of the term (0..2^32-1).
        Using CRC32 for speed + determinism.
        """
        # zlib.crc32 returns signed in some contexts; mask to uint32
        return zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF

    def get_term_frequencies(self, text: str) -> Tuple[Dict[int, int], int]:
        """
        Returns ({term_hash: tf}, token_count)
        """
        tokens = self.tokenize(text)
        tf: Dict[int, int] = {}
        for t in tokens:
            h = self.hash_term_to_index(t)
            tf[h] = tf.get(h, 0) + 1
        return tf, len(tokens)