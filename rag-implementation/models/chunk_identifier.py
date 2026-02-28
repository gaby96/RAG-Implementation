from dataclasses import dataclass
import uuid


@dataclass(frozen=True, slots=True)
class ChunkIdentifier:
    source_document: str
    section_path: str
    chunk_index: int

    def as_key(self) -> str:
        return f"{self.source_document}||{self.section_path}||{self.chunk_index}"

    def as_qdrant_uuid(self, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
        """Deterministic UUID based on content."""
        return str(uuid.uuid5(namespace, self.as_key()))