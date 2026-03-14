from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextBlock:
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    color_key: str
    page_number: int
    top: float
    bottom: float


@dataclass(slots=True)
class FontStatistics:
    average_font_size: float
    max_font_size: float
    min_font_size: float
    body_font_size: float
    heading_font_sizes: list[float]  # distinct sizes (rounded), desc