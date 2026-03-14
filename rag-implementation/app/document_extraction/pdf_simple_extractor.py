from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import pdfplumber

from app.models.document_section import DocumentSection, PageText


@dataclass(slots=True)
class PdfSimpleExtractor:
    """
    A simple PDF extraction strategy that extracts all content into a single section.
    Equivalent to the .NET PdfSimpleExtractor.
    """
    skip_pages: int = 0

    def __post_init__(self) -> None:
        if self.skip_pages < 0:
            raise ValueError("skip_pages cannot be negative")

    async def extract(self, file: BinaryIO) -> list[DocumentSection]:
        # Ensure we read from the start
        try:
            file.seek(0)
        except Exception:
            pass

        page_texts: list[PageText] = []
        start_page = 0

        with pdfplumber.open(file) as pdf:
            end_page = len(pdf.pages)

            for page_index0, page in enumerate(pdf.pages):
                page_number = page_index0 + 1

                if page_number <= self.skip_pages:
                    continue

                if start_page == 0:
                    start_page = page_number

                text = (page.extract_text() or "").strip()

                if text:
                    page_texts.append(PageText(page_number=page_number, text=text))

        if not page_texts:
            return []

        return [
            DocumentSection(
                heading_text="",
                full_heading_path="",
                heading_font_size=0.0,
                heading_is_bold=False,
                level=1,
                start_page=start_page,
                end_page=end_page,
                page_texts=page_texts,
            )
        ]