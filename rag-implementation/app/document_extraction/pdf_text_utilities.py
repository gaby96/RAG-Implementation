# app/document_extraction/pdf_text_utilities.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any


@dataclass(frozen=True, slots=True)
class WordPosition:
    """
    Represents a word's text and vertical position for line break detection.
    """
    text: str
    top: float
    bottom: float


def get_word_positions(words: Iterable[Mapping[str, Any]]) -> list[WordPosition]:
    """
    Extract word positions from pdfplumber/pdfminer words.

    Expected word shape (pdfplumber.extract_words):
      { "text": "...", "top": float, "bottom": float, ... }
    """
    positions: list[WordPosition] = []
    for w in words:
        txt = str(w.get("text", "")).strip()
        if not txt:
            continue

        # pdfplumber uses top/bottom where y increases downward
        top = float(w.get("top", 0.0))
        bottom = float(w.get("bottom", 0.0))

        positions.append(WordPosition(text=txt, top=top, bottom=bottom))
    return positions


def build_text_with_line_breaks(words: list[WordPosition]) -> str:
    """
    Builds text from word positions, inserting line breaks where appropriate.
    Uses heuristics based on vertical position changes to detect line breaks.
    """
    if not words:
        return ""

    out: list[str] = []
    for i, current in enumerate(words):
        if i > 0:
            prev = words[i - 1]
            out.append("\n" if should_insert_line_break(prev, current) else " ")
        out.append(current.text)

    return "".join(out)


def should_insert_line_break(previous: WordPosition, current: WordPosition) -> bool:
    """
    Determines whether a line break should be inserted between two words.

    NOTE about coordinates:
    - In your C# (PdfPig), Y typically increases upward.
    - In pdfplumber/pdfminer, Y typically increases downward.
      So "below" means current.top is GREATER than previous.top.

    We adapt the vertical gap calculations accordingly.
    """

    # line height reference (use absolute to be safe)
    prev_line_height = abs(previous.bottom - previous.top)
    cur_line_height = abs(current.bottom - current.top)
    avg_line_height = (prev_line_height + cur_line_height) / 2.0

    if avg_line_height < 1:
        avg_line_height = 10.0

    baseline_diff = abs(previous.bottom - current.bottom)
    top_diff = abs(previous.top - current.top)

    # same line if vertical positions are very similar
    if baseline_diff < avg_line_height * 0.3 and top_diff < avg_line_height * 0.3:
        return False

    # In pdfplumber coordinates, moving down increases Y.
    # vertical_gap positive means current is below previous.
    vertical_gap = current.top - previous.bottom

    # large gap => paragraph break
    if vertical_gap > avg_line_height * 1.5:
        return True

    # normal wrap gap: only break if punctuation / list boundary
    if vertical_gap > avg_line_height * 0.3:
        prev_text = previous.text.rstrip()
        if not prev_text:
            return False

        last_char = prev_text[-1]
        if last_char in (".", "?", "!", ":", ";"):
            return True

        # list items like "1." or "2)"
        if prev_text[0].isdigit() and (prev_text.endswith(".") or prev_text.endswith(")")):
            return True

    return False