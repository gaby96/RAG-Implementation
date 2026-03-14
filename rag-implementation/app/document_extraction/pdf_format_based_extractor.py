from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Dict, Optional, Tuple, Any
from collections import Counter

import pdfplumber

from app.models.document_section import DocumentSection, PageText
from app.models.text_block import TextBlock, FontStatistics





@dataclass(slots=True)
class PdfFormatBasedExtractor:
    """
    Extraction strategy for PDFs by detecting sections using font characteristics.

    - max_heading_depth: maximum heading level to treat as section boundary
    - max_heading_font_size: ignore headings larger than this (e.g. title page)
    - skip_pages: number of leading pages to skip (0 means start at page 1)
    - heading_color_key: if set, only treat that color as heading color
    """

    max_heading_depth: int = 1
    max_heading_font_size: Optional[float] = None
    skip_pages: int = 0
    heading_color_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.max_heading_depth < 1:
            raise ValueError("max_heading_depth must be at least 1")
        if self.max_heading_font_size is not None and self.max_heading_font_size <= 0:
            raise ValueError("max_heading_font_size must be > 0")
        if self.skip_pages < 0:
            raise ValueError("skip_pages cannot be negative")

    async def extract(self, file: BinaryIO) -> list[DocumentSection]:
        """
        Equivalent to ExtractAsync(Stream file)
        """
        # pdfplumber needs the stream positioned at start
        try:
            file.seek(0)
        except Exception:
            pass

        with pdfplumber.open(file) as pdf:
            text_blocks: list[TextBlock] = []

            # First pass: collect word-level blocks with font info
            for page_index0, page in enumerate(pdf.pages):
                page_number = page_index0 + 1
                if page_number <= self.skip_pages:
                    continue

                # extract_words can include extra_attrs pulled from underlying chars
                words = page.extract_words(
                    use_text_flow=True,
                    keep_blank_chars=False,
                    extra_attrs=["size", "fontname", "non_stroking_color", "stroking_color"],
                )

                # Sort in reading order (top-to-bottom, left-to-right)
                words.sort(key=lambda w: (w.get("top", 0.0), w.get("x0", 0.0)))

                for w in words:
                    txt = (w.get("text") or "").strip()
                    if not txt:
                        continue

                    font_size = float(w.get("size") or 0.0)
                    if self.max_heading_font_size is not None and font_size > self.max_heading_font_size:
                        # ignore huge text entirely (matches your C# "continue")
                        continue

                    font_name = str(w.get("fontname") or "")
                    color_key = self._get_color_key(w.get("non_stroking_color"))

                    is_bold = self._is_bold_font(font_name)
                    is_italic = self._is_italic_font(font_name)

                    top = float(w.get("top") or 0.0)
                    bottom = float(w.get("bottom") or 0.0)

                    text_blocks.append(
                        TextBlock(
                            text=txt,
                            font_size=font_size,
                            font_name=font_name,
                            is_bold=is_bold,
                            is_italic=is_italic,
                            color_key=color_key,
                            page_number=page_number,
                            top=top,
                            bottom=bottom,
                        )
                    )

        if not text_blocks:
            return []

        font_stats = self._calculate_font_statistics(text_blocks)

        # Second pass: identify sections & group text by page
        # Keep blocks in reading order across pages
        text_blocks.sort(key=lambda b: (b.page_number, b.top, b.bottom))
        return self._identify_sections(text_blocks, font_stats)

    # ----------------------------
    # Font/color helpers
    # ----------------------------

    def _get_color_key(self, color: Any) -> str:
        """
        pdfplumber/pdfminer represent colors inconsistently:
        - None
        - int (gray)
        - tuple/list (RGB or CMYK-ish depending on PDF)
        We'll normalize to a stable string.
        """
        if color is None:
            return "default"

        # Common case: RGB tuple of floats 0..1
        if isinstance(color, (tuple, list)):
            vals = [float(v) for v in color]
            return "_".join(f"{v:.2f}" for v in vals)

        # grayscale int/float
        try:
            v = float(color)
            return f"gray_{v:.2f}"
        except Exception:
            return str(color) or "default"

    def _is_bold_font(self, font_name: str) -> bool:
        lower = (font_name or "").lower()
        return any(x in lower for x in ("bold", "heavy", "black", "demi"))

    def _is_italic_font(self, font_name: str) -> bool:
        lower = (font_name or "").lower()
        return any(x in lower for x in ("italic", "oblique", "slant"))

    # ----------------------------
    # Font statistics (body vs headings)
    # ----------------------------

    def _calculate_font_statistics(self, blocks: list[TextBlock]) -> FontStatistics:
        sizes = [b.font_size for b in blocks if b.font_size > 0]
        if not sizes:
            # fallback
            return FontStatistics(
                average_font_size=0.0,
                max_font_size=0.0,
                min_font_size=0.0,
                body_font_size=0.0,
                heading_font_sizes=[],
            )

        avg_size = sum(sizes) / len(sizes)
        max_size = max(sizes)
        min_size = min(sizes)

        rounded = [round(s, 1) for s in sizes]
        body_font_size = Counter(rounded).most_common(1)[0][0]

        # candidate heading sizes: larger than body * 1.1
        heading_candidates = [b for b in blocks if b.font_size > body_font_size * 1.1]

        if self.heading_color_key:
            heading_candidates = [b for b in heading_candidates if b.color_key == self.heading_color_key]

        if self.max_heading_font_size is not None:
            heading_candidates = [b for b in heading_candidates if b.font_size <= self.max_heading_font_size]

        heading_sizes = sorted(
            {round(b.font_size, 1) for b in heading_candidates},
            reverse=True,
        )

        return FontStatistics(
            average_font_size=avg_size,
            max_font_size=max_size,
            min_font_size=min_size,
            body_font_size=body_font_size,
            heading_font_sizes=heading_sizes,
        )

    # ----------------------------
    # Section detection
    # ----------------------------

    def _identify_sections(self, blocks: list[TextBlock], stats: FontStatistics) -> list[DocumentSection]:
        sections: list[DocumentSection] = []

        current_section = DocumentSection()
        current_page_texts: Dict[int, list[str]] = {}  # page -> parts
        current_heading_parts: list[str] = []

        is_collecting_heading = False
        heading_font_size = 0.0
        heading_is_bold = False
        heading_start_page = 0
        heading_level = 0

        last_page_number = 0
        last_body_block: Optional[TextBlock] = None

        headings_by_level: Dict[int, str] = {}

        for block in blocks:
            block_level = self._get_heading_level(block, stats)
            is_heading = (block_level > 0) and (block_level <= self.max_heading_depth)

            if is_heading:
                continuation = (
                    is_collecting_heading
                    and abs(block.font_size - heading_font_size) < 0.5
                    and block.page_number == last_page_number
                )

                if continuation:
                    current_heading_parts.append(block.text)
                else:
                    # finalize previous section if it has body content
                    if current_page_texts:
                        page_texts = self._finalize_page_texts(current_page_texts)
                        if page_texts:
                            current_section.page_texts = page_texts
                            current_section.end_page = max(p.page_number for p in page_texts)
                            sections.append(current_section)
                        current_page_texts.clear()

                    # start new heading
                    is_collecting_heading = True
                    current_heading_parts = [block.text]
                    heading_font_size = block.font_size
                    heading_is_bold = block.is_bold
                    heading_start_page = block.page_number
                    heading_level = block_level

                    current_section = DocumentSection(
                        start_page=block.page_number,
                        level=heading_level,
                    )

                last_body_block = None
            else:
                # finalize heading if needed
                if is_collecting_heading:
                    heading_text = " ".join(current_heading_parts).strip()
                    current_section.heading_text = heading_text
                    current_section.heading_font_size = heading_font_size
                    current_section.heading_is_bold = heading_is_bold
                    current_section.start_page = heading_start_page
                    current_section.level = heading_level

                    headings_by_level[heading_level] = heading_text
                    for lvl in sorted([l for l in headings_by_level.keys() if l > heading_level]):
                        headings_by_level.pop(lvl, None)

                    current_section.full_heading_path = self._build_full_heading_path(headings_by_level, heading_level)

                    is_collecting_heading = False
                    last_body_block = None

                # add body text to page bucket
                if block.page_number not in current_page_texts:
                    current_page_texts[block.page_number] = []

                parts = current_page_texts[block.page_number]

                if parts:
                    # decide newline vs space
                    if last_body_block and last_body_block.page_number == block.page_number:
                        if self._should_insert_line_break(last_body_block, block):
                            parts.append("\n")
                        else:
                            parts.append(" ")
                    else:
                        parts.append(" ")

                parts.append(block.text)
                last_body_block = block

            last_page_number = block.page_number

        # finalize last heading if still collecting
        if is_collecting_heading:
            heading_text = " ".join(current_heading_parts).strip()
            current_section.heading_text = heading_text
            current_section.heading_font_size = heading_font_size
            current_section.heading_is_bold = heading_is_bold
            current_section.start_page = heading_start_page
            current_section.level = heading_level

            headings_by_level[heading_level] = heading_text
            for lvl in sorted([l for l in headings_by_level.keys() if l > heading_level]):
                headings_by_level.pop(lvl, None)

            current_section.full_heading_path = self._build_full_heading_path(headings_by_level, heading_level)

        # finalize last section body
        if current_page_texts:
            page_texts = self._finalize_page_texts(current_page_texts)
            if page_texts:
                current_section.page_texts = page_texts
                current_section.end_page = max(p.page_number for p in page_texts)
                sections.append(current_section)

        return sections

    def _finalize_page_texts(self, page_parts: Dict[int, list[str]]) -> list[PageText]:
        pts: list[PageText] = []
        for page_num in sorted(page_parts.keys()):
            text = "".join(page_parts[page_num]).strip()
            if text:
                pts.append(PageText(page_number=page_num, text=text))
        return pts

    def _build_full_heading_path(self, headings_by_level: Dict[int, str], current_level: int) -> str:
        parts = [headings_by_level[l] for l in sorted(headings_by_level.keys()) if l <= current_level]
        return " > ".join(parts)

    def _get_heading_level(self, block: TextBlock, stats: FontStatistics) -> int:
        if self.max_heading_font_size is not None and block.font_size > self.max_heading_font_size:
            return 0

        if self.heading_color_key and block.color_key != self.heading_color_key:
            return 0

        rounded = round(block.font_size, 1)

        eligible = (
            [s for s in stats.heading_font_sizes if s <= self.max_heading_font_size]
            if self.max_heading_font_size is not None
            else stats.heading_font_sizes
        )

        if rounded in eligible:
            return eligible.index(rounded) + 1

        # bold body text treated as lowest-level heading
        if block.is_bold and block.font_size >= stats.body_font_size:
            return len(eligible) + 1

        return 0

    # ----------------------------
    # Line-break heuristic
    # ----------------------------

    def _should_insert_line_break(self, prev: TextBlock, curr: TextBlock) -> bool:
        """
        Rough equivalent of PdfTextUtilities.ShouldInsertLineBreak based on vertical movement.
        pdfplumber gives top/bottom in PDF units.

        Heuristic:
        - If the top position jumps down noticeably -> newline
        """
        prev_height = max(1.0, prev.bottom - prev.top)
        # if current word is sufficiently lower than previous line -> treat as new line
        return (curr.top - prev.top) > (prev_height * 0.8)