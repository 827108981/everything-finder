from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from local_full_text_search.config.constants import TEMP_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.docx_parser import DocxParser
from local_full_text_search.parsers.image_parser import ImageParser
from local_full_text_search.parsers.pdf_parser import PdfParser
from local_full_text_search.parsers.pptx_parser import PptxParser
from local_full_text_search.parsers.text_parser import TextParser
from local_full_text_search.parsers.xlsx_parser import XlsxParser


class ZipParser(BaseParser):
    """Safely index supported files inside ZIP archives.

    Archives are treated as containers, not trusted folders. We validate member
    count, total uncompressed size, recursion depth and paths before extracting
    anything to the app temp directory.
    """

    name = "zip"

    def __init__(self, settings: AppSettings, depth: int = 0) -> None:
        super().__init__()
        self.settings = settings
        self.depth = depth

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".zip"

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        if self.depth >= self.settings.max_zip_depth:
            self.set_status("skipped", "ZIP_DEPTH_LIMIT", "超过压缩包递归层级限制，仅索引压缩包文件名")
            return
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        extracted_root = Path(tempfile.mkdtemp(prefix="zip_index_", dir=TEMP_DIR))
        yielded = 0
        failed_members = 0
        skipped_members = 0
        try:
            with zipfile.ZipFile(file_path) as archive:
                infos = [info for info in archive.infolist() if not info.is_dir()]
                if len(infos) > self.settings.max_zip_file_count:
                    self.set_status("skipped", "ZIP_FILE_COUNT_LIMIT", "压缩包内文件数量超过安全限制")
                    return
                total_size = sum(info.file_size for info in infos)
                if total_size > self.settings.max_zip_uncompressed_bytes:
                    self.set_status("skipped", "ZIP_SIZE_LIMIT", "压缩包解压后体积超过安全限制")
                    return
                for info in infos:
                    cancel_token.wait_if_paused()
                    cancel_token.throw_if_cancelled()
                    if info.flag_bits & 0x1:
                        skipped_members += 1
                        continue
                    safe_name = safe_zip_member_name(info.filename)
                    if safe_name is None:
                        failed_members += 1
                        continue
                    parser = self._parser_for_member(safe_name)
                    if parser is None:
                        skipped_members += 1
                        continue
                    extracted = extracted_root / hashlib.sha256(info.filename.encode("utf-8")).hexdigest()
                    extracted = extracted.with_suffix(Path(safe_name).suffix)
                    with archive.open(info) as source, extracted.open("wb") as target:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            cancel_token.throw_if_cancelled()
                            target.write(chunk)
                    try:
                        for block in parser.parse(extracted, cancel_token):
                            block.file_path = str(file_path)
                            block.location_text = f"{file_path.name} > {safe_name} > {block.location_text}"
                            block.extra["zip_internal_path"] = safe_name
                            yielded += 1
                            yield block
                    except Exception:
                        failed_members += 1
                if failed_members:
                    self.set_status(
                        "partial_success" if yielded else "failed",
                        "ZIP_PARTIAL_FAILURE",
                        f"压缩包内 {failed_members} 个文件解析失败，{skipped_members} 个文件跳过",
                    )
                elif skipped_members and not yielded:
                    self.set_status("metadata_only", "ZIP_NO_SUPPORTED_MEMBER", "压缩包内没有可解析文件")
        except zipfile.BadZipFile:
            self.set_status("failed", "ZIP_CORRUPTED", "压缩包损坏或格式异常")

    def _parser_for_member(self, internal_path: str) -> BaseParser | None:
        suffix = Path(internal_path).suffix.lower()
        if suffix in {".txt", ".log", ".csv", ".md", ".json", ".xml", ".ini"}:
            return TextParser()
        if suffix == ".pdf":
            return PdfParser(
                enable_scanned_ocr=self.settings.enable_ocr and self.settings.ocr_scanned_pdf,
                ocr_language=self.settings.ocr_language,
            )
        if suffix == ".docx":
            return DocxParser()
        if suffix in {".xlsx", ".xlsm"}:
            return XlsxParser()
        if suffix == ".pptx":
            return PptxParser()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            return ImageParser(
                language=self.settings.ocr_language,
                enabled=self.settings.enable_ocr and self.settings.ocr_images,
                min_pixels=self.settings.min_ocr_image_pixels,
                max_side=self.settings.max_ocr_image_side,
            )
        if suffix == ".zip":
            return ZipParser(self.settings, depth=self.depth + 1)
        return None


def safe_zip_member_name(name: str) -> str | None:
    candidate = PurePosixPath(name.replace("\\", "/"))
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        return None
    clean = str(candidate)
    return clean if clean and clean != "." else None
