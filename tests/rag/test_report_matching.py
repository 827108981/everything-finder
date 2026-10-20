from __future__ import annotations

from local_full_text_search.rag.evaluation.reporting.matching import (
    MatchConfig,
    highlight_segments,
    match_quote,
    normalize_text,
)


def test_normalize_removes_whitespace_bullets_and_folds_width() -> None:
    assert normalize_text("清\n 洗液 • １２３") == "清洗液123"


def test_quote_with_pdf_line_break_matches_full_gold() -> None:
    match = match_quote(
        {"filename": "guide.pdf", "page_number": 15, "text": "请使用清洗液处理样本。"},
        {"file": "guide.pdf", "page": 15, "quote": "请使用清\n洗液处理样本。"},
        kind="positive", evidence_id="line-break",
    )
    assert match.status == "exact"
    assert match.coverage == 1.0


def test_page_tolerance_allows_one_page_but_rejects_two() -> None:
    chunk = {"filename": "guide.pdf", "page_number": 19, "text": "流量标准为 12 mL"}
    one = match_quote(
        chunk, {"file": "guide.pdf", "page": 20, "quote": "流量标准为 12 mL"},
        kind="positive", evidence_id="p1",
    )
    two = match_quote(
        chunk, {"file": "guide.pdf", "page": 21, "quote": "流量标准为 12 mL"},
        kind="positive", evidence_id="p2",
    )
    assert one.status == "exact"
    assert two.status == "unmatched"


def test_partial_threshold_boundaries_are_configurable() -> None:
    chunk = {"filename": "guide.pdf", "page_number": 1, "text": "abcdef"}
    config = MatchConfig(high_threshold=0.60, partial_threshold=0.30)
    partial = match_quote(
        chunk, {"file": "guide.pdf", "page": 1, "quote": "abcXYZ"},
        kind="positive", evidence_id="p",
        config=config,
    )
    assert partial.status == "partial"
    assert partial.coverage == 0.5


def test_same_chunk_can_highlight_gold_and_negative() -> None:
    chunk = {"filename": "guide.pdf", "page_number": 19, "text": "正确流量 12；旧机型流量 8"}
    gold = match_quote(
        chunk, {"file": "guide.pdf", "page": 19, "quote": "正确流量 12"},
        kind="positive", evidence_id="gold",
    )
    negative = match_quote(
        chunk, {"file": "guide.pdf", "page": 19, "quote": "旧机型流量 8"},
        kind="hard_negative", evidence_id="negative",
    )
    segments = highlight_segments(chunk["text"], [gold, negative])
    assert gold.status == "exact"
    assert negative.status == "exact"
    assert {kind for segment in segments for kind in segment["kinds"]} == {"positive", "hard_negative"}
