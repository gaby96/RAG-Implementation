from dataclasses import dataclass
from .document_chunk import DocumentChunk


@dataclass(slots=True)
class RerankedResult:
    """
    Represents a search result that has been re-ranked by the LLM.
    """

    chunk: DocumentChunk  # forward reference
    relevance_score: float  # 0–10
    original_rank: int
    new_rank: int