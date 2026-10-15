from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_full_text_search.rag.block_reader import BlockReader
from local_full_text_search.rag.chunk_builder import ChunkBuilder, ChunkPolicy


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read evidence chunks from the active published search generation."
    )
    parser.add_argument("--index-base", required=True, type=Path)
    parser.add_argument("--file-id", required=True, type=int)
    parser.add_argument("--limit", type=int, default=5, help="maximum chunks to print")
    parser.add_argument("--target-chars", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    builder = ChunkBuilder(
        ChunkPolicy(
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
    )
    emitted = 0
    with BlockReader(args.index_base).open_active() as reader:
        for document in reader.iter_documents(
            file_ids={args.file_id},
            limit=1,
        ):
            for chunk in builder.build(document):
                print(
                    json.dumps(
                        {
                            "generation": reader.generation,
                            "stable_chunk_id": chunk.stable_chunk_id,
                            "chunk_policy_version": chunk.chunk_policy_version,
                            "text_hash": chunk.text_hash,
                            "text": chunk.text,
                            "normalized_text": chunk.normalized_text,
                            "sources": [
                                {
                                    "file_id": source.file_id,
                                    "file_path": source.file_path,
                                    "filename": source.filename,
                                    "document_id": source.document_id,
                                    "block_index_start": source.block_index_start,
                                    "block_index_end": source.block_index_end,
                                    "location_text": source.location_text,
                                    "page_number": source.page_number,
                                    "line_start": source.line_start,
                                    "line_end": source.line_end,
                                    "source_type": source.source_type,
                                    "ocr_confidence": source.ocr_confidence,
                                    "spans": [
                                        {
                                            "block_id": span.block_id,
                                            "block_index": span.block_index,
                                            "raw_start": span.raw_start,
                                            "raw_end": span.raw_end,
                                            "separator_before": span.separator_before,
                                        }
                                        for span in source.spans
                                    ],
                                }
                                for source in chunk.sources
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
                emitted += 1
                if emitted >= args.limit:
                    return 0
    return 0 if emitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
