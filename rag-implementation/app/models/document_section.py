from dataclasses import dataclass, field
from typing import list

@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int
    text: str = ""

@dataclass(slots=True)
class DocumentSection:
    #The section heading or title (e.g., "Introduction", "Methodology")
    heading_text: str = ""

    #Full hierarchical heading path
    full_heading_path: str = ""

    #Font-based extraction attributes
    heading_font_size: float = 0.0
    heading_is_bold: bool = False

    #Hierarchy and page span
    level: int = 1
    start_page: int = 1
    end_page: int = 1

    #Text content organized by page
    page_texts: list[PageText] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenate text from all pages in this section."""
        return "\n".join(pt.text for pt in self.page_texts)