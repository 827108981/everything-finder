from __future__ import annotations

from local_full_text_search.core.block_coalescer import BlockCoalescer
from local_full_text_search.models.content_block import ContentBlock


def make_block(index: int, text: str) -> ContentBlock:
    return ContentBlock(
        file_path="sample.docx",
        block_index=index,
        block_type="docx_paragraph",
        location_text=f"正文第 {index + 1} 段",
        raw_text=text,
        normalized_text="",
    )


def test_small_adjacent_docx_paragraphs_are_normalized_after_merge() -> None:
    blocks = [make_block(index, f"Line {index}") for index in range(10)]

    merged = BlockCoalescer(target_chars=256, max_chars=1024).coalesce(blocks)

    assert len(merged) == 1
    assert merged[0].block_index == 0
    assert merged[0].extra["coalesced_blocks"] == 10
    assert "line 0" in merged[0].normalized_text
    assert "正文第 1 段" in merged[0].location_text
    assert "正文第 10 段" in merged[0].location_text
