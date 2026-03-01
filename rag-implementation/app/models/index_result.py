from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class IndexResult:
    chunk_count: int