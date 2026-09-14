"""Compact, versioned compression for published snapshot text blocks."""

from __future__ import annotations

from dataclasses import dataclass

import zstandard


CODEC_ZSTD = 1
DEFAULT_LEVEL = 6


@dataclass(frozen=True, slots=True)
class TextBlobCodec:
    """Encode text once for durable storage and decode only when needed."""

    level: int = DEFAULT_LEVEL

    def encode(self, text: str) -> tuple[int, bytes, int]:
        raw = str(text or "").encode("utf-8")
        compressor = zstandard.ZstdCompressor(level=int(self.level))
        return CODEC_ZSTD, compressor.compress(raw), len(raw)

    def decode(self, codec: int, payload: bytes | memoryview | None) -> str:
        if not payload:
            return ""
        if int(codec) != CODEC_ZSTD:
            raise ValueError(f"不支持的正文压缩 codec: {codec}")
        decompressor = zstandard.ZstdDecompressor()
        return decompressor.decompress(bytes(payload)).decode("utf-8")


DEFAULT_TEXT_CODEC = TextBlobCodec()


def decode_text_blob(payload: bytes | memoryview | None, codec: int) -> str:
    """SQLite scalar-function adapter used by the compact snapshot view."""

    return DEFAULT_TEXT_CODEC.decode(int(codec), payload)
