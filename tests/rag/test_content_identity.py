from __future__ import annotations

import pytest

from local_full_text_search.rag.content_identity import compare_content, canonicalize_with_offsets


@pytest.mark.parametrize(("left", "right"), [
    ("5 mA", "5 MA"),
    ("2 m²", "2 m2"),
    ("1 2", "12"),
    ("A→B", "AB"),
    ("140±5 μL/s", "115±5 μL/s"),
    ("不得排放", "可以排放"),
    ("P02:废液泵", "P02废液:泵"),
    ("P02\t废液泵", "P02废液泵"),
])
def test_fact_difference_is_never_automatic_equivalence(left: str, right: str) -> None:
    assert compare_content(left, right).status not in {"exact", "layout_equivalent"}


def test_chinese_pdf_line_break_is_layout_equivalent() -> None:
    assert compare_content("清\n洗液", "清洗液").status == "layout_equivalent"


def test_high_similarity_stays_suspected_when_number_changes() -> None:
    common = "磁分离盘排放处理流程" * 20
    assert compare_content(common + "140 μL/s", common + "115 μL/s").status == "suspected_equivalent"


def test_canonical_offsets_point_back_to_original_after_pdf_line_break() -> None:
    value = canonicalize_with_offsets("清\n洗液 Ａ")
    assert value.text.startswith("清洗液A")
    assert value.offsets[1] == (2, 3)
    assert value.offsets[-1] == (5, 6)
