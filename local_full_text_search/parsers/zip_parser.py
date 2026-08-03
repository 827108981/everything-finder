from __future__ import annotations

import codecs
import hashlib
import io
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from local_full_text_search.config.constants import TEMP_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.errors import (
    CancelledError,
    PauseRequestedError,
    ZipMemberContentChangedError,
    ZipMemberDirectoryChangedError,
    ZipMemberSizeChangedError,
)
from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.docx_parser import DocxParser
from local_full_text_search.parsers.image_parser import ImageParser
from local_full_text_search.parsers.ooxml.docx_stream_parser import DocxStreamParser
from local_full_text_search.parsers.ooxml.pptx_stream_parser import PptxStreamParser
from local_full_text_search.parsers.ooxml.xlsx_stream_parser import XlsxStreamParser
from local_full_text_search.parsers.pdf_parser import PdfParser
from local_full_text_search.parsers.pptx_parser import PptxParser
from local_full_text_search.parsers.text_parser import TextParser
from local_full_text_search.parsers.xlsx_parser import XlsxParser
from local_full_text_search.ocr.ocr_engine import OcrEngine


ZIP_MEMBER_EXTENSIONS = {
    ".txt", ".log", ".csv", ".md", ".json", ".xml", ".ini",
    ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx",
    ".doc", ".xls", ".ppt",
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".zip",
}


@dataclass(frozen=True, slots=True)
class ZipMemberDescriptor:
    member_index: int
    member_name: str
    internal_path: str
    size_bytes: int
    compressed_bytes: int
    crc32: int
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ZipManifest:
    members: tuple[ZipMemberDescriptor, ...]
    total_members: int
    skipped_members: int
    unsafe_members: int
    encrypted_members: int


def scan_zip_manifest(
    file_path: Path,
    settings: AppSettings,
    cancel_token: CancelToken,
    *,
    progress_reporter: object | None = None,
) -> ZipManifest:
    """Validate an archive and collect stable member metadata.

    Exact hashing is intentionally deferred to the index planner so only
    deduplication candidates pay for a full SHA-256 read.
    """

    members: list[ZipMemberDescriptor] = []
    skipped = 0
    unsafe = 0
    encrypted = 0
    with zipfile.ZipFile(file_path) as archive:
        infos = archive.infolist()
        file_infos = [info for info in infos if not info.is_dir()]
        if len(file_infos) > settings.max_zip_file_count:
            raise ValueError("ZIP_FILE_COUNT_LIMIT")
        if sum(int(info.file_size) for info in file_infos) > settings.max_zip_uncompressed_bytes:
            raise ValueError("ZIP_SIZE_LIMIT")
        for member_index, info in enumerate(infos):
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            if info.is_dir():
                skipped += 1
                _report_manifest_progress(
                    progress_reporter,
                    member_index,
                    len(infos),
                    info.filename,
                )
                continue
            decoded_name = decoded_zip_member_name(info)
            safe_name = safe_zip_member_name(decoded_name)
            if safe_name is None:
                unsafe += 1
                _report_manifest_progress(
                    progress_reporter,
                    member_index,
                    len(infos),
                    decoded_name,
                )
                continue
            if info.flag_bits & 0x1:
                encrypted += 1
                _report_manifest_progress(
                    progress_reporter,
                    member_index,
                    len(infos),
                    decoded_name,
                )
                continue
            if Path(safe_name).suffix.lower() not in ZIP_MEMBER_EXTENSIONS:
                skipped += 1
                _report_manifest_progress(
                    progress_reporter,
                    member_index,
                    len(infos),
                    decoded_name,
                )
                continue
            members.append(
                ZipMemberDescriptor(
                    member_index=member_index,
                    member_name=info.filename,
                    internal_path=safe_name,
                    size_bytes=int(info.file_size),
                    compressed_bytes=int(info.compress_size),
                    crc32=int(info.CRC),
                )
            )
            _report_manifest_progress(
                progress_reporter,
                member_index,
                len(infos),
                decoded_name,
            )
    return ZipManifest(tuple(members), len(infos), skipped, unsafe, encrypted)


def _report_manifest_progress(
    reporter: object | None,
    member_index: int,
    total: int,
    detail: str,
) -> None:
    if reporter is None:
        return
    advance = getattr(reporter, "advance", None)
    if not callable(advance):
        return
    completed = member_index + 1
    advance(
        phase="zip_manifest",
        completed=completed,
        total=total,
        cursor=f"member:{completed}",
        checkpoint_version=completed,
        detail=detail,
    )


def hash_zip_member(
    file_path: Path,
    descriptor: ZipMemberDescriptor,
    cancel_token: CancelToken,
) -> str:
    """Return the exact SHA-256 for one validated ZIP member."""

    digest = hashlib.sha256()
    with zipfile.ZipFile(file_path) as archive:
        infos = archive.infolist()
        if descriptor.member_index < 0 or descriptor.member_index >= len(infos):
            raise ZipMemberDirectoryChangedError("ZIP member directory changed during indexing")
        info = infos[descriptor.member_index]
        if info.is_dir():
            raise ZipMemberDirectoryChangedError("ZIP member directory changed during indexing")
        decoded_name = decoded_zip_member_name(info)
        safe_name = safe_zip_member_name(decoded_name)
        if safe_name != descriptor.internal_path:
            raise ZipMemberDirectoryChangedError("ZIP member directory changed during indexing")
        if int(info.file_size) != int(descriptor.size_bytes):
            raise ZipMemberSizeChangedError("ZIP member size changed during indexing")
        if int(info.CRC) != int(descriptor.crc32):
            raise ZipMemberContentChangedError("ZIP member content changed during indexing")
        with archive.open(info) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                cancel_token.throw_if_cancelled()
                digest.update(chunk)
    return digest.hexdigest()


class ZipParser(BaseParser):
    """Safely index supported files inside ZIP archives.

    Archives are treated as containers, not trusted folders. We validate member
    count, total uncompressed size, recursion depth and paths before extracting
    anything to the app temp directory.
    """

    name = "zip"
    supports_resume = True

    def __init__(
        self,
        settings: AppSettings,
        depth: int = 0,
        ocr_engine: OcrEngine | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.depth = depth
        self.ocr_engine = ocr_engine

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".zip"

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        if self.depth >= self.settings.max_zip_depth:
            self.set_status("skipped", "ZIP_DEPTH_LIMIT", "超过压缩包递归层级限制，仅索引压缩包文件名")
            return
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        extracted_root = Path(tempfile.mkdtemp(prefix="zip_index_", dir=TEMP_DIR))
        yielded = 1 if self.resume_cursor > 0 else 0
        failed_members = 0
        skipped_members = 0
        try:
            with zipfile.ZipFile(file_path) as archive:
                infos = archive.infolist()
                file_infos = [info for info in infos if not info.is_dir()]
                if len(file_infos) > self.settings.max_zip_file_count:
                    self.set_status("skipped", "ZIP_FILE_COUNT_LIMIT", "压缩包内文件数量超过安全限制")
                    return
                total_size = sum(info.file_size for info in file_infos)
                if total_size > self.settings.max_zip_uncompressed_bytes:
                    self.set_status("skipped", "ZIP_SIZE_LIMIT", "压缩包解压后体积超过安全限制")
                    return
                start_member = min(self.resume_cursor, len(infos))
                for member_index, info in enumerate(infos):
                    if member_index < start_member:
                        continue
                    cancel_token.wait_if_paused()
                    cancel_token.throw_if_cancelled()
                    if info.is_dir():
                        skipped_members += 1
                        self._report_member_progress(member_index, len(infos), Path(info.filename).name or info.filename)
                        continue
                    decoded_name = decoded_zip_member_name(info)
                    if info.flag_bits & 0x1:
                        skipped_members += 1
                        self._report_member_progress(member_index, len(infos), decoded_name)
                        continue
                    safe_name = safe_zip_member_name(decoded_name)
                    if safe_name is None:
                        failed_members += 1
                        self._record_member_diagnostic(
                            decoded_name,
                            "zip",
                            "ZIP_MEMBER_UNSAFE_PATH",
                            "成员路径不安全，已拒绝解压",
                        )
                        self._report_member_progress(member_index, len(infos), decoded_name)
                        continue
                    parser = self._parser_for_member(safe_name)
                    if parser is None:
                        skipped_members += 1
                        self._report_member_progress(member_index, len(infos), safe_name)
                        continue
                    if Path(safe_name).suffix.lower() in {
                        ".txt", ".log", ".csv", ".md", ".json", ".xml", ".ini"
                    }:
                        try:
                            for block in self._parse_text_member(
                                archive,
                                info,
                                file_path,
                                safe_name,
                                cancel_token,
                            ):
                                yielded += 1
                                yield block
                        except (CancelledError, PauseRequestedError):
                            raise
                        except Exception as exc:
                            failed_members += 1
                            self._record_member_exception(
                                safe_name,
                                parser.name,
                                exc,
                            )
                        self._report_member_progress(member_index, len(infos), safe_name)
                        continue
                    extracted = extracted_root / hashlib.sha256(info.filename.encode("utf-8")).hexdigest()
                    extracted = extracted.with_suffix(Path(safe_name).suffix)
                    try:
                        with archive.open(info) as source, extracted.open("wb") as target:
                            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                                cancel_token.throw_if_cancelled()
                                target.write(chunk)
                        parser.configure_runtime(
                            progress_callback=lambda payload, current=member_index, total=len(infos), name=safe_name: self._report_inner_progress(
                                payload,
                                current,
                                total,
                                name,
                            )
                        )
                        for block in parser.parse(extracted, cancel_token):
                            block.file_path = str(file_path)
                            block.location_text = f"{file_path.name} > {safe_name} > {block.location_text}"
                            block.extra["zip_internal_path"] = safe_name
                            yielded += 1
                            yield block
                    except (CancelledError, PauseRequestedError):
                        raise
                    except Exception as exc:
                        failed_members += 1
                        self._record_member_exception(
                            safe_name,
                            parser.name,
                            exc,
                        )
                    self._report_member_progress(member_index, len(infos), safe_name)
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
        finally:
            shutil.rmtree(extracted_root, ignore_errors=True)

    def _record_member_exception(
        self,
        member_path: str,
        parser_name: str,
        exc: Exception,
    ) -> None:
        self._record_member_diagnostic(
            member_path,
            parser_name,
            exc.__class__.__name__,
            str(exc) or exc.__class__.__name__,
        )

    def _record_member_diagnostic(
        self,
        member_path: str,
        parser_name: str,
        error_code: str,
        error_message: str,
    ) -> None:
        self.last_diagnostics.append(
            {
                "member_path": str(member_path),
                "parser": str(parser_name),
                "error_code": str(error_code),
                "error_message": str(error_message)[:2000],
            }
        )

    def _report_member_progress(self, member_index: int, total: int, name: str) -> None:
        self.report_progress(
            "zip_member",
            completed=member_index + 1,
            total=total,
            unit_type="member",
            cursor=member_index + 1,
            detail=name,
        )

    def _report_inner_progress(
        self,
        payload: dict[str, object],
        member_index: int,
        total: int,
        name: str,
    ) -> None:
        phase = str(payload.get("phase") or "parse")
        completed = max(0, int(payload.get("completed") or 0))
        inner_total = max(0, int(payload.get("total") or 0))
        detail = str(payload.get("detail") or "")
        inner = f"{completed}/{inner_total}" if inner_total else str(completed)
        self.report_progress(
            f"zip_{phase}",
            completed=member_index,
            total=total,
            unit_type="member",
            cursor=member_index,
            detail=f"{name} · {inner}" + (f" · {detail}" if detail else ""),
        )

    def _parse_text_member(
        self,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        outer_path: Path,
        safe_name: str,
        cancel_token: CancelToken,
    ) -> Iterable[ContentBlock]:
        with archive.open(info) as sample_stream:
            sample = sample_stream.read(65536)
        encoding = _detect_bytes_encoding(sample)
        lines: list[str] = []
        start_line = 1
        block_index = 0
        with archive.open(info) as binary_stream:
            with io.TextIOWrapper(
                binary_stream,
                encoding=encoding,
                errors="replace",
                newline="",
            ) as text_stream:
                for line_number, line in enumerate(text_stream, start=1):
                    cancel_token.wait_if_paused()
                    cancel_token.throw_if_cancelled()
                    lines.append(line.rstrip("\r\n"))
                    if len(lines) >= 500:
                        yield self.make_block(
                            outer_path,
                            block_index,
                            "zip_text",
                            f"{outer_path.name} > {safe_name} > 第 {start_line}-{line_number} 行",
                            "\n".join(lines),
                            line_start=start_line,
                            line_end=line_number,
                            extra={"zip_internal_path": safe_name},
                        )
                        block_index += 1
                        start_line = line_number + 1
                        lines = []
        if lines:
            end_line = start_line + len(lines) - 1
            yield self.make_block(
                outer_path,
                block_index,
                "zip_text",
                f"{outer_path.name} > {safe_name} > 第 {start_line}-{end_line} 行",
                "\n".join(lines),
                line_start=start_line,
                line_end=end_line,
                extra={"zip_internal_path": safe_name},
            )

    def _parser_for_member(self, internal_path: str) -> BaseParser | None:
        suffix = Path(internal_path).suffix.lower()
        if suffix in {".txt", ".log", ".csv", ".md", ".json", ".xml", ".ini"}:
            return TextParser()
        if suffix == ".pdf":
            return PdfParser(
                enable_scanned_ocr=self.settings.enable_ocr and self.settings.ocr_scanned_pdf,
                ocr_language=self.settings.ocr_language,
                parallel_min_bytes=self.settings.pdf_parallel_min_bytes,
                parallel_min_pages=self.settings.pdf_parallel_min_pages,
                parallel_workers=max(2, self.settings.parser_workers),
                ocr_engine=self.ocr_engine,
                ocr_cpu_threads=self.settings.ocr_cpu_threads,
            )
        if suffix == ".docx":
            fallback = DocxParser()
            return DocxStreamParser(fallback, defer_normalization=True) if self.settings.fast_ooxml_enabled else fallback
        if suffix in {".xlsx", ".xlsm"}:
            fallback = XlsxParser()
            return (
                XlsxStreamParser(
                    fallback,
                    defer_normalization=True,
                    sheet_workers=self.settings.xlsx_sheet_workers,
                    shared_strings_disk_threshold_bytes=(
                        self.settings.xlsx_shared_strings_disk_threshold_bytes
                    ),
                )
                if self.settings.fast_ooxml_enabled
                else fallback
            )
        if suffix == ".pptx":
            fallback = PptxParser()
            return PptxStreamParser(fallback, defer_normalization=True) if self.settings.fast_ooxml_enabled else fallback
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            return ImageParser(
                language=self.settings.ocr_language,
                enabled=self.settings.enable_ocr and self.settings.ocr_images,
                min_pixels=self.settings.min_ocr_image_pixels,
                max_side=self.settings.max_ocr_image_side,
                ocr_engine=self.ocr_engine,
                ocr_cpu_threads=self.settings.ocr_cpu_threads,
            )
        if suffix == ".zip":
            return ZipParser(
                self.settings,
                depth=self.depth + 1,
                ocr_engine=self.ocr_engine,
            )
        return None


def safe_zip_member_name(name: str) -> str | None:
    candidate = PurePosixPath(name.replace("\\", "/"))
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        return None
    clean = str(candidate)
    return clean if clean and clean != "." else None


def decoded_zip_member_name(info: zipfile.ZipInfo) -> str:
    """Recover GB18030 names written by legacy ZIP tools without the UTF-8 flag."""

    name = info.filename
    if info.flag_bits & 0x800 or name.isascii():
        return name
    try:
        raw_name = name.encode("cp437")
        candidate = raw_name.decode("gb18030")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name
    if _filename_text_quality(candidate) > _filename_text_quality(name) + 2.0:
        return candidate
    return name


def _filename_text_quality(value: str) -> float:
    score = 0.0
    for character in value:
        codepoint = ord(character)
        if "\u3400" <= character <= "\u9fff":
            score += 2.0
        elif character.isascii():
            score += 0.1 if character.isalnum() else 0.0
        elif 0x2500 <= codepoint <= 0x259F:
            score -= 3.0
        elif character == "\ufffd" or codepoint < 32:
            score -= 5.0
        elif character.isalnum():
            score -= 0.25
        else:
            score -= 0.5
    return score


def _detect_bytes_encoding(sample: bytes) -> str:
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if sample.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if sample.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    for encoding in ("utf-8", "gb18030", "utf-16"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes

        detected = from_bytes(sample).best()
        if detected and detected.encoding:
            return detected.encoding
    except ImportError:
        pass
    return "utf-8"
