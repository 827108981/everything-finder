from __future__ import annotations

from local_full_text_search.core.semantic_progress import (
    SemanticProgress,
    is_semantic_progress,
    progress_signature,
)


def test_duplicate_heartbeat_is_not_semantic_progress() -> None:
    previous = SemanticProgress("pdf_page", 4, 20, "page=4")
    current = SemanticProgress("pdf_page", 4, 20, "page=4")

    assert not is_semantic_progress(previous, current)


def test_transport_sequence_is_not_part_of_semantic_progress() -> None:
    previous = {
        "phase": "ocr_detect",
        "completed": 2,
        "cursor": "page=1;region=2",
        "progress_sequence": 10,
    }
    current = {**previous, "progress_sequence": 11}

    assert not is_semantic_progress(previous, current)


def test_each_monotonic_unit_can_advance_progress() -> None:
    base = SemanticProgress("ocr_region", 2, 10, "page=1;region=2", 100, 2, 1)

    assert is_semantic_progress(base, SemanticProgress("ocr_region", 3, 10, "page=1;region=2", 100, 2, 1))
    assert is_semantic_progress(base, SemanticProgress("ocr_region", 2, 10, "page=1;region=3", 100, 2, 1))
    assert is_semantic_progress(base, SemanticProgress("ocr_region", 2, 10, "page=1;region=2", 101, 2, 1))
    assert is_semantic_progress(base, SemanticProgress("ocr_region", 2, 10, "page=1;region=2", 100, 3, 1))
    assert is_semantic_progress(base, SemanticProgress("ocr_region", 2, 10, "page=1;region=2", 100, 2, 2))
    assert is_semantic_progress(base, SemanticProgress("ocr_complete", 2, 10, "page=1;region=2", 100, 2, 1))


def test_cursor_must_move_forward_with_the_same_shape() -> None:
    previous = SemanticProgress("xlsx_row", 0, 0, "sheet=3;row=120")

    assert is_semantic_progress(previous, SemanticProgress("xlsx_row", 0, 0, "sheet=3;row=121"))
    assert not is_semantic_progress(previous, SemanticProgress("xlsx_row", 0, 0, "sheet=3;row=119"))
    assert not is_semantic_progress(previous, SemanticProgress("xlsx_row", 0, 0, "opaque-next"))


def test_signature_is_stable_for_the_same_stall_point() -> None:
    first = SemanticProgress("legacy_office_open", 0, 1, "file=4")
    second = SemanticProgress("legacy_office_open", 0, 1, "file=4")

    assert progress_signature(first) == progress_signature(second)
