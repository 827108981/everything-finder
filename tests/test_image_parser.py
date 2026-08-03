from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
from PIL import Image

from local_full_text_search.core.errors import PauseRequestedError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.ocr.ocr_cache import OcrCache
from local_full_text_search.ocr.ocr_engine import OcrResult
from local_full_text_search.parsers.image_parser import ImageParser


class _CountingOcr:
    def __init__(self) -> None:
        self.calls = 0

    def recognize_adaptive(self, _path: Path, **_kwargs: object) -> OcrResult:
        self.calls += 1
        return OcrResult("EXACT_IMAGE_TEXT", 0.95, {"boxes": []})


class _ProgressOcr:
    def recognize_adaptive(
        self,
        _path: Path,
        **kwargs: object,
    ) -> OcrResult:
        progress = kwargs["progress_callback"]
        assert callable(progress)
        progress("detect", 1, 2, "detected")
        progress("recognize_original_regions", 2, 2, "recognized")
        return OcrResult("RESUMED_IMAGE_TEXT", 0.96, {"boxes": []})


def test_u0_03r_image_ocr_progress_cursor_advances_after_resume(
    tmp_path: Path,
) -> None:
    source = tmp_path / "resume.png"
    Image.new("RGB", (640, 480), "white").save(source)
    parser = ImageParser(
        min_pixels=0,
        ocr_engine=_ProgressOcr(),
    )
    parser.cache = OcrCache(tmp_path / "cache")
    progress: list[dict[str, object]] = []
    parser.configure_runtime(
        resume_cursor=3,
        content_digest=hashlib.sha256(source.read_bytes()).hexdigest(),
        progress_callback=progress.append,
    )

    blocks = list(parser.parse(source, CancelToken()))

    assert parser.supports_resume is True
    assert [item["cursor"] for item in progress] == [4, 5, 6]
    assert blocks[0].raw_text == "RESUMED_IMAGE_TEXT"


def test_u0_03r_image_ocr_does_not_convert_pause_to_ocr_failure(
    tmp_path: Path,
) -> None:
    class PausingOcr:
        def recognize_adaptive(
            self,
            _path: Path,
            **_kwargs: object,
        ) -> OcrResult:
            raise PauseRequestedError("pause")

    source = tmp_path / "pause.png"
    Image.new("RGB", (640, 480), "white").save(source)
    parser = ImageParser(min_pixels=0, ocr_engine=PausingOcr())
    parser.cache = OcrCache(tmp_path / "cache")

    with pytest.raises(PauseRequestedError):
        list(parser.parse(source, CancelToken()))

    assert parser.last_status == "success"


def test_p0_06r_identical_directory_and_zip_materialized_image_reuse_exact_cache(
    tmp_path: Path,
) -> None:
    first = tmp_path / "directory.png"
    second = tmp_path / "zip-materialized.png"
    Image.new("RGB", (640, 480), "white").save(first)
    shutil.copyfile(first, second)
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    engine = _CountingOcr()
    cache = OcrCache(tmp_path / "cache")
    parser = ImageParser(ocr_engine=engine)
    parser.cache = cache

    parser.configure_runtime(content_digest=digest)
    first_blocks = list(parser.parse(first, CancelToken()))
    parser.configure_runtime(content_digest=digest)
    second_blocks = list(parser.parse(second, CancelToken()))

    assert engine.calls == 1
    assert first_blocks[0].file_path == str(first)
    assert second_blocks[0].file_path == str(second)
    assert second_blocks[0].extra["ocr_exact_cache_hits"] == 1


def test_p0_06r_same_pixels_with_different_language_do_not_reuse(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (640, 480), "white").save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = OcrCache(tmp_path / "cache")
    first_engine = _CountingOcr()
    first_parser = ImageParser(language="ch", ocr_engine=first_engine)
    first_parser.cache = cache
    first_parser.configure_runtime(content_digest=digest)
    list(first_parser.parse(source, CancelToken()))

    second_engine = _CountingOcr()
    second_parser = ImageParser(language="en", ocr_engine=second_engine)
    second_parser.cache = cache
    second_parser.configure_runtime(content_digest=digest)
    list(second_parser.parse(source, CancelToken()))

    assert first_engine.calls == 1
    assert second_engine.calls == 1
