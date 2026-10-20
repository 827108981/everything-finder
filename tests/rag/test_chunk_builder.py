from __future__ import annotations

from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.models import SourceBlock


def _block(text: str, *, page: int = 1, block_id: int = 1) -> SourceBlock:
    return SourceBlock(
        source_generation=5,
        block_id=block_id,
        document_id=9,
        file_id=3,
        root_id=2,
        file_path=r"C:\资料\恢复手册.pdf",
        filename="恢复手册.pdf",
        extension=".pdf",
        document_key="sha256-doc",
        block_index=block_id,
        text=text,
        location_text=f"第 {page} 页",
        page_number=page,
    )


def test_chunk_ids_are_stable_for_same_input() -> None:
    builder = ChunkBuilder(target_chars=100, max_chars=120, overlap_chars=10)
    text = "索引中断后可以根据检查点恢复。" * 10
    first = list(builder.iter_chunks([_block(text)]))
    second = list(builder.iter_chunks([_block(text)]))
    assert [item.stable_chunk_id for item in first] == [
        item.stable_chunk_id for item in second
    ]
    assert all(item.source.page_number == 1 for item in first)


def test_chunking_never_crosses_page_boundary() -> None:
    builder = ChunkBuilder(target_chars=100, max_chars=120, overlap_chars=10)
    chunks = list(
        builder.iter_chunks(
            [_block("第一页内容。" * 12, page=1), _block("第二页内容。" * 12, page=2, block_id=2)]
        )
    )
    assert chunks
    assert {item.source.page_number for item in chunks} == {1, 2}
    assert all(len(item.text) <= 120 for item in chunks)
