from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SearchHit:
    """一个文件内部的单个命中位置，例如 PDF 第3页或 Excel 的某个单元格。"""

    # 命中的正文块编号；文件名等元数据命中时可以没有 block_id。
    block_id: int | None
    # 给用户看的位置说明和关键词周围的上下文摘要。
    location_text: str
    context: str
    # 这个位置中的命中次数，以及内容来源（正文、metadata、OCR等）。
    hit_count: int
    source_type: str
    # 不同文档类型使用不同的精确定位字段；不适用的字段保持 None。
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    cell_start: str | None = None
    cell_end: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    ocr_confidence: float | None = None
    is_fuzzy: bool = False


@dataclass(slots=True)
class SearchResult:
    """一个文件级搜索结果；可包含该文件内部的多个 SearchHit。"""

    # 文件和主要命中块的数据库标识。
    file_id: int
    block_id: int | None
    # UI 展示和打开文件所需的基本信息。
    file_path: str
    filename: str
    extension: str
    size_bytes: int
    modified_time: float
    # 最佳命中的位置、上下文、次数和来源。
    location_text: str
    context: str
    hit_count: int
    source_type: str
    parse_status: str
    # SearchEngine 根据命中次数、文件名命中和OCR来源计算的相关度分数。
    score: float
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    cell_start: str | None = None
    cell_end: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    ocr_confidence: float | None = None
    has_fuzzy_match: bool = False
    # 同一个文件内合并后的全部命中位置，供结果展开和预览使用。
    matches: list[SearchHit] = field(default_factory=list)
