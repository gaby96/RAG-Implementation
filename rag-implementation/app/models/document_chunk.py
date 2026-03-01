from dataclasses import dataclass, field
from typing import List, dict, Optional

@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    source_document: str
    
    chunk_index: int
    chunk_total: int

    start_page: Optional[int] = None
    end_page: Optional[int] = None

    section: Optional[str] = None
    section_path: Optional[str] = None

    embedding: Optional[List[float]] = None
    metadata: dict[str, str] = field(default_factory=dict)

    #--------- RAG Methods ---------

    def generate_embedding_text(self) -> str:
        header = self._build_path_header()
        if header:
            return f"{header}\n\n{self.text}"
        return self.text
    
    def _build_path_header(self) -> str:
        parts = []

        if self.source_document:
            parts.append(f"Document: {self.source_document}")
        
        if self.section_path:
            parts.append(f"Section: {self.section_path}")
        elif self.section:
            parts.append(f"Section: {self.section}")

        return "\n".join(parts)
    

    #--------- Qdrant Helper ----------
    def to_qdrant_payload(self) -> dict:
        """Convert to Qdrant payload (metadata stored alongside vector)."""
        return {
            "source_document": self.source_document,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "chunk_index": self.chunk_index,
            "chunk_total": self.chunk_total,
            "section": self.section,
            "section_path": self.section_path,
            **self.metadata  # Include any additional metadata 
        }