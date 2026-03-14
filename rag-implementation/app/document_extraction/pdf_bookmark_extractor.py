# app/document_extraction/extraction_strategies/pdf_bookmark_extractor.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO, Dict, Optional, Tuple

import pdfplumber
from pypdf import PdfReader

from app.models.document_section import DocumentSection, PageText


@dataclass(slots=True)
class FlattenedBookmark:
    title: str
    full_title: str
    page_number: int  # 1-based
    level: int
    y_position: Optional[float] = None  # not always available in Python tooling


@dataclass(slots=True)
class PdfBookmarkExtractor:
    """
    ExtractionStrategy that uses PDF bookmarks/outlines to create DocumentSections.

    - max_bookmark_depth: 1 = only top level, 2 = include children, etc.
    - skip_pages: ignore pages before this index (0 means include from page 1)
    """
    max_bookmark_depth: int = 1
    skip_pages: int = 0

    def __post_init__(self) -> None:
        if self.max_bookmark_depth < 1:
            raise ValueError("max_bookmark_depth must be at least 1")
        if self.skip_pages < 0:
            raise ValueError("skip_pages cannot be negative")

    async def extract(self, file: BinaryIO) -> list[DocumentSection]:
        # pypdf wants a seekable stream
        try:
            file.seek(0)
        except Exception:
            pass

        reader = PdfReader(file)
        total_pages = len(reader.pages)

        outlines = getattr(reader, "outlines", None)
        if not outlines:
            raise ValueError("Failed to read bookmarks/outlines from PDF document.")

        flattened = self._flatten_outlines(reader, outlines, depth=1, parent_titles=[])
        flattened.sort(key=lambda b: (b.page_number, b.level))

        # Extract per-page text (line-break rich) and "search text" (space-joined)
        # pdfplumber reads from path/bytes; easiest is to reopen stream bytes
        try:
            file.seek(0)
        except Exception:
            pass
        pdf_bytes = file.read()

        page_text_with_breaks: Dict[int, str] = {}
        page_text_space_joined: Dict[int, str] = {}

        with pdfplumber.open(pdf_bytes) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_index0 = i - 1
                if page_index0 < self.skip_pages:
                    continue

                txt = page.extract_text() or ""
                txt = txt.strip()

                page_text_with_breaks[i] = txt
                # for searching headings we normalize to spaces
                page_text_space_joined[i] = self._normalize_for_comparison(txt)

        return self._create_sections_from_bookmarks(
            bookmarks=flattened,
            page_full_texts=page_text_space_joined,
            page_texts_with_line_breaks=page_text_with_breaks,
            total_pages=total_pages,
        )

    # ---------------------------
    # Outline / bookmark handling
    # ---------------------------

    def _flatten_outlines(
        self,
        reader: PdfReader,
        nodes,
        depth: int,
        parent_titles: list[str],
    ) -> list[FlattenedBookmark]:
        """
        pypdf outlines are nested lists/objects.
        We flatten them similarly to your C# FlattenBookmarks.
        """
        result: list[FlattenedBookmark] = []

        for node in nodes:
            # node can be a list (subtree) or a Destination-like object
            if isinstance(node, list):
                # recurse into nested list at same depth (structure quirk)
                result.extend(self._flatten_outlines(reader, node, depth, parent_titles))
                continue

            title = getattr(node, "title", None) or getattr(node, "/Title", None) or ""
            title = str(title).strip()
            if not title:
                continue

            # Only include within depth limit
            if depth <= self.max_bookmark_depth:
                page_number = self._resolve_outline_page(reader, node) or 1

                # apply skip_pages
                if page_number - 1 < self.skip_pages:
                    # still recurse into children, but skip this node itself
                    pass
                else:
                    full_title = " > ".join(parent_titles + [title]) if parent_titles else title
                    result.append(
                        FlattenedBookmark(
                            title=title,
                            full_title=full_title,
                            page_number=page_number,
                            level=depth,
                            y_position=None,
                        )
                    )

            # Recurse into children if present (pypdf uses `.children` sometimes)
            children = getattr(node, "children", None)
            if children:
                result.extend(self._flatten_outlines(reader, children, depth + 1, parent_titles + [title]))

        return result

    def _resolve_outline_page(self, reader: PdfReader, outline_node) -> Optional[int]:
        """
        Try to resolve an outline node to a 1-based page number.
        """
        try:
            # pypdf can resolve outline destinations to page indices
            page_index0 = reader.get_destination_page_number(outline_node)
            return int(page_index0) + 1
        except Exception:
            return None

    # ---------------------------
    # Section building
    # ---------------------------

    def _create_sections_from_bookmarks(
        self,
        bookmarks: list[FlattenedBookmark],
        page_full_texts: Dict[int, str],
        page_texts_with_line_breaks: Dict[int, str],
        total_pages: int,
    ) -> list[DocumentSection]:
        sections: list[DocumentSection] = []

        for i, bm in enumerate(bookmarks):
            bookmark_page = bm.page_number

            next_bm = bookmarks[i + 1] if i + 1 < len(bookmarks) else None
            next_page_guess = next_bm.page_number if next_bm else total_pages + 1

            # Find where this heading appears (search expected page .. expected page+2, plus one page back)
            actual_start_page, heading_start_idx = self._find_heading_location(
                page_full_texts,
                bm.title,
                expected_page=bookmark_page,
                max_page=min(bookmark_page + 2, total_pages),
            )

            if actual_start_page == -1:
                actual_start_page = bookmark_page
                heading_start_idx = 0

            content_start_idx = heading_start_idx
            start_page_text = page_full_texts.get(actual_start_page, "")
            heading_end_idx = self._find_heading_end_index(start_page_text, bm.title, heading_start_idx)
            if heading_end_idx > heading_start_idx:
                content_start_idx = heading_end_idx

            # Determine next heading boundary
            if next_bm:
                next_heading_page, next_heading_idx = self._find_heading_location(
                    page_full_texts,
                    next_bm.title,
                    expected_page=next_page_guess,
                    max_page=min(next_page_guess + 2, total_pages),
                )
                if next_heading_page == -1:
                    next_heading_page, next_heading_idx = next_page_guess, 0
            else:
                next_heading_page, next_heading_idx = -1, -1

            end_page = next_heading_page if next_heading_page > 0 else total_pages

            page_texts: list[PageText] = []

            for page_num in range(actual_start_page, end_page + 1):
                page_full = page_full_texts.get(page_num, "")
                if not page_full:
                    continue

                linebreak_text = page_texts_with_line_breaks.get(page_num, "")
                text_to_use = linebreak_text or page_full

                # First page: trim after heading
                if page_num == actual_start_page and content_start_idx > 0:
                    adjusted = self._map_index_to_linebreak_text(page_full, text_to_use, content_start_idx)
                    if 0 <= adjusted < len(text_to_use):
                        text_to_use = text_to_use[adjusted:].lstrip()

                # Last page boundary: stop before next heading
                if next_heading_page > 0 and page_num == next_heading_page:
                    if next_heading_idx == 0:
                        # next section starts at beginning of this page
                        continue

                    adjusted_next = self._map_index_to_linebreak_text(page_full, text_to_use, next_heading_idx)

                    if page_num == actual_start_page and content_start_idx > 0:
                        adjusted_start = self._map_index_to_linebreak_text(page_full, text_to_use, content_start_idx)
                        relative_next = adjusted_next - adjusted_start
                        if relative_next <= 0:
                            continue
                        text_to_use = text_to_use[:relative_next].rstrip()
                    else:
                        if 0 < adjusted_next < len(text_to_use):
                            text_to_use = text_to_use[:adjusted_next].rstrip()

                if text_to_use.strip():
                    page_texts.append(PageText(page_number=page_num, text=text_to_use))

            if page_texts:
                sections.append(
                    DocumentSection(
                        heading_text=bm.title,
                        full_heading_path=bm.full_title,
                        heading_font_size=0.0,   # bookmark-based extraction doesn't use font metrics
                        heading_is_bold=False,
                        level=bm.level,
                        start_page=actual_start_page,
                        end_page=max(p.page_number for p in page_texts),
                        page_texts=page_texts,
                    )
                )

        return sections

    # ---------------------------
    # Heading search helpers
    # ---------------------------

    def _find_heading_location(
        self,
        page_full_texts: Dict[int, str],
        heading_title: str,
        expected_page: int,
        max_page: int,
    ) -> Tuple[int, int]:
        if not heading_title:
            return -1, -1

        # forward search
        for p in range(expected_page, max_page + 1):
            txt = page_full_texts.get(p, "")
            if not txt:
                continue
            idx = self._find_heading_in_text(txt, heading_title)
            if idx >= 0:
                return p, idx

        # one page back
        if expected_page > 1:
            txt = page_full_texts.get(expected_page - 1, "")
            idx = self._find_heading_in_text(txt, heading_title)
            if idx >= 0:
                return expected_page - 1, idx

        return -1, -1

    def _find_heading_in_text(self, page_text: str, heading_title: str) -> int:
        if not page_text or not heading_title:
            return -1

        # try exact
        exact = page_text.lower().find(heading_title.lower())
        if exact >= 0:
            return exact

        # fuzzy word-sequence search
        heading_norm = self._normalize_for_comparison(heading_title)
        words = [w for w in heading_norm.split(" ") if w]
        if not words:
            return -1

        search_idx = 0
        first_word_idx = -1
        wi = 0

        while search_idx < len(page_text) and wi < len(words):
            word = words[wi]
            found = page_text.lower().find(word.lower(), search_idx)
            if found < 0:
                return -1

            if wi == 0:
                first_word_idx = found
            else:
                gap = found - search_idx
                if gap > len(word) + 10:
                    search_idx = first_word_idx + len(words[0])
                    wi = 0
                    first_word_idx = -1
                    continue

            search_idx = found + len(word)
            wi += 1

        return first_word_idx

    def _find_heading_end_index(self, page_text: str, heading_title: str, heading_start_index: int) -> int:
        if not page_text or not heading_title or heading_start_index < 0:
            return heading_start_index

        # exact match
        end = heading_start_index + len(heading_title)
        if end <= len(page_text):
            if page_text[heading_start_index:end].lower() == heading_title.lower():
                while end < len(page_text) and page_text[end].isspace():
                    end += 1
                return end

        # fallback: word-by-word
        heading_norm = self._normalize_for_comparison(heading_title)
        words = [w for w in heading_norm.split(" ") if w]
        if not words:
            return heading_start_index

        search_idx = heading_start_index
        last_end = heading_start_index

        lower = page_text.lower()
        for w in words:
            found = lower.find(w.lower(), search_idx)
            if found < 0:
                break
            last_end = found + len(w)
            search_idx = last_end

        while last_end < len(page_text) and page_text[last_end].isspace():
            last_end += 1

        return last_end

    def _map_index_to_linebreak_text(self, space_joined_text: str, linebreak_text: str, space_idx: int) -> int:
        # Very close to your C# logic: match non-whitespace count
        if space_idx <= 0:
            return 0
        if not space_joined_text or not linebreak_text:
            return space_idx

        non_ws = 0
        for i in range(min(space_idx, len(space_joined_text))):
            if not space_joined_text[i].isspace():
                non_ws += 1

        target = non_ws
        current = 0
        for i, ch in enumerate(linebreak_text):
            if not ch.isspace():
                current += 1
            if current >= target:
                return i + 1

        return len(linebreak_text)

    def _normalize_for_comparison(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.strip())