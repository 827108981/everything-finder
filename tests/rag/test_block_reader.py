from __future__ import annotations

import sqlite3

from local_full_text_search.core.text_blob_codec import TextBlobCodec
from local_full_text_search.rag.block_reader import BlockReader


def _snapshot(path) -> None:
    codec = TextBlobCodec()
    native_codec, native_text, _ = codec.encode("原生正文包含索引恢复说明。")
    weak_codec, weak_text, _ = codec.encode("低置信度 OCR 内容。")
    good_codec, good_text, _ = codec.encode("高置信度 OCR 内容。")
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY, root_id INTEGER, path TEXT, filename TEXT,
            extension TEXT, content_hash TEXT
        );
        CREATE TABLE blocks (
            id INTEGER PRIMARY KEY, file_id INTEGER, document_id INTEGER,
            block_index INTEGER, text_blob BLOB, codec INTEGER, location_text TEXT,
            page_number INTEGER, slide_number INTEGER, sheet_name TEXT,
            cell_start TEXT, cell_end TEXT, line_start INTEGER, line_end INTEGER,
            source_type TEXT, ocr_confidence REAL
        );
        CREATE VIEW search_blocks AS SELECT
            id, file_id, document_id, block_index,
            decode_text(text_blob, codec) AS raw_text, location_text,
            page_number, slide_number, sheet_name, cell_start, cell_end,
            line_start, line_end, source_type, ocr_confidence
        FROM blocks;
        INSERT INTO files VALUES
            (1, 9, 'C:/docs/guide.pdf', 'guide.pdf', '.pdf', 'hash-1');
        """
    )
    connection.executemany(
        """INSERT INTO blocks VALUES(
            ?,1,10,?,?,?, ?,?,NULL,NULL,NULL,NULL,NULL,NULL,?,?
        )""",
        [
            (1, 0, native_text, native_codec, "第 1 页", 1, "native_text", None),
            (2, 1, weak_text, weak_codec, "第 2 页", 2, "ocr", 5500),
            (3, 2, good_text, good_codec, "第 3 页", 3, "ocr", 9200),
        ],
    )
    connection.commit()
    connection.close()


def test_reader_decodes_compressed_text_and_normalizes_scaled_ocr_confidence(tmp_path):
    snapshot = tmp_path / "generation.db"
    _snapshot(snapshot)

    blocks = list(BlockReader().iter_blocks(snapshot, 8, min_ocr_confidence=0.6))

    assert [block.block_id for block in blocks] == [1, 3]
    assert blocks[0].source_generation == 8
    assert blocks[0].text == "原生正文包含索引恢复说明。"
    assert blocks[0].file_path == "C:/docs/guide.pdf"
    assert blocks[1].source_type == "ocr"
    assert blocks[1].ocr_confidence == 0.92
    assert blocks[1].page_number == 3


def test_reader_can_exclude_all_ocr_blocks(tmp_path):
    snapshot = tmp_path / "generation.db"
    _snapshot(snapshot)

    blocks = list(BlockReader().iter_blocks(snapshot, 8, include_ocr=False))

    assert [block.block_id for block in blocks] == [1]
