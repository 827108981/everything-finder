from __future__ import annotations

import fnmatch
import hashlib
import os
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from local_full_text_search.config.constants import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILE_PATTERNS,
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
)
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.content_fingerprint import (
    ContentFingerprint,
    OOXML_EXTENSIONS,
)
from local_full_text_search.core.errors import (
    ZipMemberContentChangedError,
    ZipMemberDirectoryChangedError,
    ZipMemberEncryptedError,
    ZipMemberSizeChangedError,
)
from local_full_text_search.core.planning_worker import PlanningProgressReporter
from local_full_text_search.core.task_manager import ProcessRunControlToken
from local_full_text_search.parsers.pdf_parser import _is_ocr_candidate
from local_full_text_search.parsers.zip_parser import (
    ZipManifest,
    decoded_zip_member_name,
    safe_zip_member_name,
    scan_zip_manifest,
)


@dataclass(frozen=True, slots=True)
class PreparedFileMetadata:
    path: str
    size_bytes: int
    modified_time: float
    created_time: float
    modified_time_ns: int
    content_hash: str | None = None
    worker_pid: int = 0


@dataclass(frozen=True, slots=True)
class FileMetadataError:
    path: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class StatBatchResult:
    metadata: tuple[PreparedFileMetadata, ...]
    errors: tuple[FileMetadataError, ...]


@dataclass(frozen=True, slots=True)
class FingerprintSourceResult:
    source_path: str
    fingerprint: ContentFingerprint
    spool_path: Path | None
    bytes_read: int
    source_size: int
    source_modified_time_ns: int
    worker_pid: int
    image_width: int = 0
    image_height: int = 0


@dataclass(frozen=True, slots=True)
class FingerprintSourceError:
    path: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class FingerprintBatchResult:
    results: tuple[FingerprintSourceResult, ...]
    errors: tuple[FingerprintSourceError, ...]


@dataclass(frozen=True, slots=True)
class PreparedZipMemberResult:
    spool_path: Path
    sha256: str
    bytes_read: int
    worker_pid: int
    image_width: int = 0
    image_height: int = 0


@dataclass(frozen=True, slots=True)
class PdfScanPage:
    page_number: int
    page_identity: str
    width_points: float
    height_points: float
    requires_ocr: bool


@dataclass(frozen=True, slots=True)
class PdfDocumentScanResult:
    pages: tuple[PdfScanPage, ...]
    source_size: int
    source_modified_time_ns: int
    worker_pid: int


class _RunControlProgressReporter:
    def __init__(
        self,
        reporter: PlanningProgressReporter,
        token: ProcessRunControlToken,
        *,
        task_id: str,
    ) -> None:
        self.reporter = reporter
        self.token = token
        self.task_id = str(task_id)

    def advance(
        self,
        *,
        phase: str,
        completed: int = 0,
        total: int = 0,
        cursor: str = "",
        bytes_read: int = 0,
        output_blocks: int = 0,
        checkpoint_version: int = 0,
        detail: str = "",
    ) -> None:
        self.token.set_pause_checkpoint(
            task_id=self.task_id,
            safe_unit_type=str(phase),
            cursor=str(cursor),
            checkpoint_version=max(
                0,
                int(checkpoint_version or completed),
            ),
        )
        self.reporter.advance(
            phase=phase,
            completed=completed,
            total=total,
            cursor=cursor,
            bytes_read=bytes_read,
            output_blocks=output_blocks,
            checkpoint_version=checkpoint_version,
            detail=detail,
        )


def discover_file_batches(
    reporter: PlanningProgressReporter,
    root_path: Path,
    include_subfolders: bool,
    settings_data: dict[str, object],
    control_dir: Path,
    batch_size: int,
) -> Iterable[list[str]]:
    """Discover supported paths in a spawned, killable planning process."""

    settings = AppSettings.from_dict(settings_data)
    token = ProcessRunControlToken(
        control_dir,
        pause_behavior="block",
    )
    excluded_dirs = set(settings.excluded_dirs or DEFAULT_EXCLUDED_DIRS)
    excluded_patterns = tuple(
        settings.excluded_file_patterns or DEFAULT_EXCLUDED_FILE_PATTERNS
    )
    root_path = Path(root_path)
    batch_limit = max(1, min(4096, int(batch_size or 1)))
    batch: list[str] = []
    discovered = 0
    completed_directories = 0
    checkpoint = 0
    reporter.advance(
        phase="root_scan",
        cursor="directory:0",
        detail=str(root_path),
    )

    if include_subfolders:
        iterator = os.walk(root_path, onerror=_raise_walk_error)
    else:
        iterator = _single_directory_walk(root_path)
    for current, dirs, files in iterator:
        _planning_safe_point(
            token,
            task_id="planning:directory_enumeration",
            phase="directory_enumeration",
            cursor=f"directory:{completed_directories}",
            checkpoint_version=checkpoint,
        )
        dirs[:] = [name for name in dirs if name not in excluded_dirs]
        completed_directories += 1
        checkpoint += 1
        reporter.advance(
            phase="directory_enumeration",
            completed=completed_directories,
            cursor=f"directory:{completed_directories}",
            checkpoint_version=checkpoint,
            detail=str(current),
        )
        for filename in files:
            _planning_safe_point(
                token,
                task_id="planning:directory_enumeration",
                phase="directory_enumeration",
                cursor=f"file:{discovered}",
                checkpoint_version=checkpoint,
            )
            path = Path(current) / filename
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in excluded_patterns):
                continue
            if not settings.include_hidden_files and _is_hidden(path):
                continue
            discovered += 1
            checkpoint += 1
            batch.append(str(path))
            reporter.advance(
                phase="directory_enumeration",
                completed=discovered,
                cursor=f"file:{discovered}",
                checkpoint_version=checkpoint,
                detail=str(path),
            )
            if len(batch) >= batch_limit:
                yield batch
                batch = []
    if batch:
        yield batch


def stat_file_batch(
    reporter: PlanningProgressReporter,
    paths: list[str],
    compute_full_hash: bool,
    validation_hang: bool = False,
    control_dir: Path | None = None,
) -> StatBatchResult:
    """Read file metadata outside the scheduler thread."""

    prepared: list[PreparedFileMetadata] = []
    errors: list[FileMetadataError] = []
    total = len(paths)
    reporter.advance(phase="file_stat", total=total, cursor="file:0")
    if validation_hang:
        _hang_for_validation()
    token = (
        ProcessRunControlToken(
            control_dir,
            pause_behavior="block",
        )
        if control_dir is not None
        else None
    )
    for index, path_text in enumerate(paths, start=1):
        _planning_safe_point(
            token,
            task_id="planning:file_stat",
            phase="file_stat",
            cursor=f"file:{index - 1}",
            checkpoint_version=index - 1,
        )
        path = Path(path_text)
        try:
            stat = path.stat()
            content_hash = (
                _sha256_with_progress(
                    reporter,
                    path,
                    start_completed=index - 1,
                    total=total,
                    token=token,
                )
                if compute_full_hash
                else None
            )
            prepared.append(
                PreparedFileMetadata(
                    path=str(path),
                    size_bytes=int(stat.st_size),
                    modified_time=float(stat.st_mtime),
                    created_time=float(stat.st_ctime),
                    modified_time_ns=int(stat.st_mtime_ns),
                    content_hash=content_hash,
                    worker_pid=os.getpid(),
                )
            )
        except Exception as exc:
            errors.append(
                FileMetadataError(
                    path=str(path),
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
            )
        reporter.advance(
            phase="file_stat",
            completed=index,
            total=total,
            cursor=f"file:{index}",
            checkpoint_version=index,
            detail=str(path),
        )
    return StatBatchResult(tuple(prepared), tuple(errors))


def fingerprint_source(
    reporter: PlanningProgressReporter,
    path: Path,
    spool_dir: Path,
    force_full_hash: bool = False,
    token: ProcessRunControlToken | None = None,
    retain_spool: bool = True,
) -> FingerprintSourceResult:
    """Fingerprint one source with byte-semantic progress and atomic spool."""

    path = Path(path)
    spool_dir = Path(spool_dir)
    stat_before = path.stat()
    suffix = path.suffix.lower()
    image_width, image_height = _planning_image_dimensions(
        path,
        suffix,
    )
    reporter.advance(
        phase="source_prepare",
        total=int(stat_before.st_size),
        cursor="offset:0",
        detail=str(path),
    )
    if not force_full_hash and (suffix in OOXML_EXTENSIONS or suffix == ".zip"):
        try:
            fingerprint, relevant_bytes, bytes_read = _fingerprint_zip_directory(
                reporter,
                path,
                suffix,
                token=token,
            )
            _assert_source_unchanged(path, stat_before.st_size, stat_before.st_mtime_ns)
            return FingerprintSourceResult(
                source_path=str(path),
                fingerprint=ContentFingerprint(
                    f"zipdir:{fingerprint}",
                    relevant_bytes,
                    "zip_directory",
                ),
                spool_path=None,
                bytes_read=bytes_read,
                source_size=int(stat_before.st_size),
                source_modified_time_ns=int(stat_before.st_mtime_ns),
                worker_pid=os.getpid(),
                image_width=image_width,
                image_height=image_height,
            )
        except zipfile.BadZipFile:
            pass
    if not force_full_hash and suffix == ".mp4":
        payload = (
            f"metadata:{stat_before.st_size}:{stat_before.st_mtime_ns}:{path.name}"
        ).encode("utf-8")
        reporter.advance(
            phase="content_hash",
            completed=1,
            total=1,
            cursor="metadata:1",
            checkpoint_version=1,
            detail=str(path),
        )
        return FingerprintSourceResult(
            source_path=str(path),
            fingerprint=ContentFingerprint(
                "metadata:" + hashlib.sha256(payload).hexdigest(),
                0,
                "metadata",
            ),
            spool_path=None,
            bytes_read=0,
            source_size=int(stat_before.st_size),
            source_modified_time_ns=int(stat_before.st_mtime_ns),
            worker_pid=os.getpid(),
            image_width=image_width,
            image_height=image_height,
        )
    if (
        force_full_hash
        or stat_before.st_size <= 64 * 1024 * 1024
        or suffix in {".doc", ".xls", ".ppt"}
    ):
        if retain_spool:
            digest, spool_path, bytes_read = _sha256_to_spool_with_progress(
                reporter,
                path,
                spool_dir,
                int(stat_before.st_size),
                token=token,
            )
        else:
            digest, bytes_read = _sha256_without_spool_with_progress(
                reporter,
                path,
                int(stat_before.st_size),
                token=token,
            )
            spool_path = None
        _assert_source_unchanged(path, stat_before.st_size, stat_before.st_mtime_ns)
        return FingerprintSourceResult(
            source_path=str(path),
            fingerprint=ContentFingerprint(
                f"sha256:{digest}",
                int(stat_before.st_size),
                "sha256",
            ),
            spool_path=spool_path,
            bytes_read=bytes_read,
            source_size=int(stat_before.st_size),
            source_modified_time_ns=int(stat_before.st_mtime_ns),
            worker_pid=os.getpid(),
            image_width=image_width,
            image_height=image_height,
        )
    digest, bytes_read = _sampled_sha256_with_progress(
        reporter,
        path,
        int(stat_before.st_size),
        token=token,
    )
    _assert_source_unchanged(path, stat_before.st_size, stat_before.st_mtime_ns)
    return FingerprintSourceResult(
        source_path=str(path),
        fingerprint=ContentFingerprint(
            f"sample:{digest}",
            int(stat_before.st_size),
            "sampled_sha256",
        ),
        spool_path=None,
        bytes_read=bytes_read,
        source_size=int(stat_before.st_size),
        source_modified_time_ns=int(stat_before.st_mtime_ns),
        worker_pid=os.getpid(),
        image_width=image_width,
        image_height=image_height,
    )


def fingerprint_source_batch(
    reporter: PlanningProgressReporter,
    requests: list[tuple[str, bool]],
    spool_dir: Path,
    validation_hang: bool = False,
    control_dir: Path | None = None,
    retain_spool: bool = True,
) -> FingerprintBatchResult:
    """Fingerprint a bounded batch while preserving monotonic byte progress."""

    results: list[FingerprintSourceResult] = []
    errors: list[FingerprintSourceError] = []
    byte_offset = 0
    total = len(requests)
    reporter.advance(phase="content_hash", total=total, cursor="file:0")
    if validation_hang:
        _hang_for_validation()
    token = (
        ProcessRunControlToken(
            control_dir,
            pause_behavior="block",
        )
        if control_dir is not None
        else None
    )
    for index, (path_text, force_full_hash) in enumerate(requests, start=1):
        _planning_safe_point(
            token,
            task_id="planning:content_hash",
            phase="content_hash",
            cursor=f"file:{index - 1}",
            checkpoint_version=byte_offset + index - 1,
        )
        adapter = _BatchProgressReporter(
            reporter,
            file_index=index,
            file_total=total,
            byte_offset=byte_offset,
        )
        try:
            result = fingerprint_source(
                adapter,
                Path(path_text),
                Path(spool_dir),
                bool(force_full_hash),
                token,
                retain_spool,
            )
            results.append(result)
            byte_offset += result.bytes_read
        except Exception as exc:
            errors.append(
                FingerprintSourceError(
                    path=str(path_text),
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
            )
        reporter.advance(
            phase="content_hash",
            completed=index,
            total=total,
            cursor=f"file:{index}",
            bytes_read=byte_offset,
            checkpoint_version=byte_offset + index,
            detail=str(path_text),
        )
    return FingerprintBatchResult(tuple(results), tuple(errors))


def scan_zip_manifest_task(
    reporter: PlanningProgressReporter,
    file_path: Path,
    settings_data: dict[str, object],
    control_dir: Path,
    validation_hang: bool = False,
) -> ZipManifest:
    settings = AppSettings.from_dict(settings_data)
    reporter.advance(phase="zip_manifest", cursor="member:0")
    if validation_hang:
        _hang_for_validation()
    token = ProcessRunControlToken(
        control_dir,
        pause_behavior="block",
    )
    token.set_pause_checkpoint(
        task_id=f"planning:zip_manifest:{file_path}",
        safe_unit_type="zip_manifest",
        cursor="member:0",
        checkpoint_version=0,
    )
    return scan_zip_manifest(
        Path(file_path),
        settings,
        token,
        progress_reporter=_RunControlProgressReporter(
            reporter,
            token,
            task_id=f"planning:zip_manifest:{file_path}",
        ),
    )


def prepare_zip_member_task(
    reporter: PlanningProgressReporter,
    archive_path: Path,
    member_index: int,
    internal_path: str,
    expected_size: int,
    expected_crc32: int,
    suffix: str,
    spool_dir: Path,
    validation_hang: bool = False,
    control_dir: Path | None = None,
) -> PreparedZipMemberResult:
    """Validate, extract and hash one ZIP member in a recoverable process."""

    archive_path = Path(archive_path)
    spool_dir = Path(spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)
    target = spool_dir / f"{uuid.uuid4().hex}{str(suffix).lower()}"
    temporary = target.with_suffix(target.suffix + ".tmp")
    digest = hashlib.sha256()
    bytes_read = 0
    token = (
        ProcessRunControlToken(
            control_dir,
            pause_behavior="block",
        )
        if control_dir is not None
        else None
    )
    reporter.advance(
        phase="zip_member_prepare",
        total=max(0, int(expected_size)),
        cursor="offset:0",
        detail=str(internal_path),
    )
    if validation_hang:
        _hang_for_validation()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if member_index < 0 or member_index >= len(infos):
                raise ZipMemberDirectoryChangedError(
                    "ZIP member directory changed during indexing"
                )
            info = infos[member_index]
            if info.is_dir():
                raise ZipMemberDirectoryChangedError(
                    "ZIP member directory changed during indexing"
                )
            if info.flag_bits & 0x1:
                raise ZipMemberEncryptedError("ZIP member is encrypted")
            decoded_name = decoded_zip_member_name(info)
            if safe_zip_member_name(decoded_name) != internal_path:
                raise ZipMemberDirectoryChangedError(
                    "ZIP member directory changed during indexing"
                )
            if int(info.file_size) != int(expected_size):
                raise ZipMemberSizeChangedError(
                    "ZIP member size changed during indexing"
                )
            if int(info.CRC) != int(expected_crc32):
                raise ZipMemberContentChangedError(
                    "ZIP member content changed during indexing"
                )
            with archive.open(info) as source, temporary.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    _planning_safe_point(
                        token,
                        task_id=(
                            "planning:zip_member_prepare:"
                            f"{archive_path}:{internal_path}"
                        ),
                        phase="zip_member_prepare",
                        cursor=f"offset:{bytes_read}",
                        checkpoint_version=bytes_read,
                        checkpoint_checksum=digest.copy().hexdigest(),
                    )
                    digest.update(chunk)
                    output.write(chunk)
                    bytes_read += len(chunk)
                    reporter.advance(
                        phase="zip_member_prepare",
                        completed=bytes_read,
                        total=max(0, int(expected_size)),
                        cursor=f"offset:{bytes_read}",
                        bytes_read=bytes_read,
                        checkpoint_version=bytes_read,
                        detail=str(internal_path),
                    )
        temporary.replace(target)
        image_width, image_height = _planning_image_dimensions(
            target,
            target.suffix.lower(),
        )
        return PreparedZipMemberResult(
            spool_path=target,
            sha256=digest.hexdigest(),
            bytes_read=bytes_read,
            worker_pid=os.getpid(),
            image_width=image_width,
            image_height=image_height,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise


def scan_pdf_document_task(
    reporter: PlanningProgressReporter,
    file_path: Path,
) -> PdfDocumentScanResult:
    """Create stable, low-cost page identities before page scheduling."""

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF task planning") from exc
    file_path = Path(file_path)
    stat = file_path.stat()
    pages: list[PdfScanPage] = []
    document = fitz.open(file_path)
    try:
        if document.needs_pass:
            raise RuntimeError("PDF_PASSWORD_REQUIRED")
        page_count = int(document.page_count)
        reporter.advance(
            phase="pdf_scan",
            total=page_count,
            cursor="page:0",
            detail=str(file_path),
        )
        for index in range(page_count):
            page = document.load_page(index)
            text = page.get_text("text") or ""
            images = page.get_images(full=True)
            rect = page.rect
            identity_payload = (
                f"{index + 1}|{rect.width:.4f}|{rect.height:.4f}|"
                f"{len(images)}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            ).encode("utf-8")
            pages.append(
                PdfScanPage(
                    page_number=index + 1,
                    page_identity=hashlib.sha256(identity_payload).hexdigest(),
                    width_points=float(rect.width),
                    height_points=float(rect.height),
                    requires_ocr=_is_ocr_candidate(text, bool(images)),
                )
            )
            reporter.advance(
                phase="pdf_scan",
                completed=index + 1,
                total=page_count,
                cursor=f"page:{index + 1}",
                checkpoint_version=index + 1,
                detail=f"第 {index + 1} 页",
            )
    finally:
        document.close()
    current = file_path.stat()
    if (
        int(current.st_size) != int(stat.st_size)
        or int(current.st_mtime_ns) != int(stat.st_mtime_ns)
    ):
        raise OSError("SOURCE_CHANGED_DURING_PDF_SCAN")
    return PdfDocumentScanResult(
        pages=tuple(pages),
        source_size=int(stat.st_size),
        source_modified_time_ns=int(stat.st_mtime_ns),
        worker_pid=os.getpid(),
    )


def _sha256_to_spool_with_progress(
    reporter: PlanningProgressReporter,
    path: Path,
    spool_dir: Path,
    expected_size: int,
    *,
    token: ProcessRunControlToken | None = None,
) -> tuple[str, Path, int]:
    spool_dir.mkdir(parents=True, exist_ok=True)
    target = spool_dir / f"{uuid.uuid4().hex}{path.suffix.lower()}"
    temporary = target.with_suffix(target.suffix + ".tmp")
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with path.open("rb") as source, temporary.open("wb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                _planning_safe_point(
                    token,
                    task_id=f"planning:content_hash:{path}",
                    phase="content_hash",
                    cursor=f"offset:{bytes_read}",
                    checkpoint_version=bytes_read,
                    checkpoint_checksum=digest.copy().hexdigest(),
                )
                digest.update(chunk)
                output.write(chunk)
                bytes_read += len(chunk)
                reporter.advance(
                    phase="content_hash",
                    completed=bytes_read,
                    total=expected_size,
                    cursor=f"offset:{bytes_read}",
                    bytes_read=bytes_read,
                    checkpoint_version=bytes_read,
                    detail=str(path),
                )
        temporary.replace(target)
        return digest.hexdigest(), target, bytes_read
    except BaseException:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise


def _sha256_without_spool_with_progress(
    reporter: PlanningProgressReporter,
    path: Path,
    expected_size: int,
    *,
    token: ProcessRunControlToken | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            _planning_safe_point(
                token,
                task_id=f"planning:content_hash:{path}",
                phase="content_hash",
                cursor=f"offset:{bytes_read}",
                checkpoint_version=bytes_read,
                checkpoint_checksum=digest.copy().hexdigest(),
            )
            digest.update(chunk)
            bytes_read += len(chunk)
            reporter.advance(
                phase="content_hash",
                completed=bytes_read,
                total=expected_size,
                cursor=f"offset:{bytes_read}",
                bytes_read=bytes_read,
                checkpoint_version=bytes_read,
                detail=str(path),
            )
    return digest.hexdigest(), bytes_read


def _sampled_sha256_with_progress(
    reporter: PlanningProgressReporter,
    path: Path,
    size: int,
    *,
    token: ProcessRunControlToken | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(size.to_bytes(8, "little", signed=False))
    sample_size = 4 * 1024 * 1024
    offsets = (0, max(0, size // 2 - sample_size // 2), max(0, size - sample_size))
    bytes_read = 0
    with path.open("rb") as stream:
        for index, offset in enumerate(offsets, start=1):
            _planning_safe_point(
                token,
                task_id=f"planning:content_hash:{path}",
                phase="content_hash",
                cursor=f"sample:{index - 1}",
                checkpoint_version=index - 1,
                checkpoint_checksum=digest.copy().hexdigest(),
            )
            stream.seek(offset, os.SEEK_SET)
            chunk = stream.read(sample_size)
            digest.update(offset.to_bytes(8, "little", signed=False))
            digest.update(chunk)
            bytes_read += len(chunk)
            reporter.advance(
                phase="content_hash",
                completed=index,
                total=len(offsets),
                cursor=f"sample:{index}",
                bytes_read=bytes_read,
                checkpoint_version=index,
                detail=str(path),
            )
    return digest.hexdigest(), bytes_read


def _fingerprint_zip_directory(
    reporter: PlanningProgressReporter,
    path: Path,
    suffix: str,
    *,
    token: ProcessRunControlToken | None = None,
) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    relevant_bytes = 0
    bytes_read = 0
    with zipfile.ZipFile(path) as archive:
        infos = sorted(
            (info for info in archive.infolist() if not info.is_dir()),
            key=lambda info: info.filename,
        )
        for index, info in enumerate(infos, start=1):
            _planning_safe_point(
                token,
                task_id=f"planning:zip_directory:{path}",
                phase="content_hash",
                cursor=f"member:{index - 1}",
                checkpoint_version=index - 1,
                checkpoint_checksum=digest.copy().hexdigest(),
            )
            encoded_name = info.filename.encode("utf-8", errors="surrogatepass")
            digest.update(len(encoded_name).to_bytes(4, "little"))
            digest.update(encoded_name)
            digest.update(int(info.CRC).to_bytes(4, "little", signed=False))
            digest.update(int(info.file_size).to_bytes(8, "little", signed=False))
            digest.update(int(info.compress_size).to_bytes(8, "little", signed=False))
            bytes_read += len(encoded_name) + 24
            if _is_relevant_entry(suffix, info.filename):
                relevant_bytes += int(info.compress_size)
            reporter.advance(
                phase="content_hash",
                completed=index,
                total=len(infos),
                cursor=f"member:{index}",
                bytes_read=bytes_read,
                checkpoint_version=index,
                detail=info.filename,
            )
    return digest.hexdigest(), relevant_bytes, bytes_read


def _sha256_with_progress(
    reporter: PlanningProgressReporter,
    path: Path,
    *,
    start_completed: int,
    total: int,
    token: ProcessRunControlToken | None = None,
) -> str:
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            _planning_safe_point(
                token,
                task_id=f"planning:content_hash:{path}",
                phase="content_hash",
                cursor=f"offset:{bytes_read}",
                checkpoint_version=bytes_read,
                checkpoint_checksum=digest.copy().hexdigest(),
            )
            digest.update(chunk)
            bytes_read += len(chunk)
            reporter.advance(
                phase="content_hash",
                completed=start_completed,
                total=total,
                cursor=f"offset:{bytes_read}",
                bytes_read=bytes_read,
                detail=str(path),
            )
    return digest.hexdigest()


def _planning_safe_point(
    token: ProcessRunControlToken | None,
    *,
    task_id: str,
    phase: str,
    cursor: str,
    checkpoint_version: int,
    checkpoint_checksum: str = "",
) -> None:
    if token is None:
        return
    token.set_pause_checkpoint(
        task_id=task_id,
        safe_unit_type=phase,
        cursor=cursor,
        checkpoint_version=checkpoint_version,
        checkpoint_checksum=checkpoint_checksum,
    )
    validation_delay = os.environ.get(
        "LFTS_VALIDATION_PLANNING_SAFE_POINT_DELAY_MS",
        "",
    )
    if validation_delay:
        try:
            time.sleep(
                max(0, min(5_000, int(validation_delay)))
                / 1000.0
            )
        except ValueError:
            pass
    token.wait_if_paused()
    token.throw_if_cancelled()


def _assert_source_unchanged(path: Path, expected_size: int, expected_mtime_ns: int) -> None:
    current = path.stat()
    if (
        int(current.st_size) != int(expected_size)
        or int(current.st_mtime_ns) != int(expected_mtime_ns)
    ):
        raise OSError("SOURCE_CHANGED_DURING_PREPARE")


def _single_directory_walk(root_path: Path):
    entries = list(root_path.iterdir())
    yield (
        str(root_path),
        [entry.name for entry in entries if entry.is_dir()],
        [entry.name for entry in entries if entry.is_file()],
    )


def _raise_walk_error(error: OSError) -> None:
    raise error


def _is_hidden(path: Path) -> bool:
    try:
        if path.name.startswith("."):
            return True
        if os.name == "nt":
            import ctypes

            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            return attrs != -1 and bool(attrs & 0x2)
    except Exception:
        return False
    return False


def _is_relevant_entry(suffix: str, name: str) -> bool:
    normalized = name.replace("\\", "/")
    if suffix == ".docx":
        return (
            normalized == "word/document.xml"
            or normalized.startswith("word/header")
            or normalized.startswith("word/footer")
        )
    if suffix == ".pptx":
        return normalized.startswith("ppt/slides/slide") or normalized.startswith(
            "ppt/notesSlides/notesSlide"
        )
    if suffix in {".xlsx", ".xlsm"}:
        return normalized in {
            "xl/workbook.xml",
            "xl/sharedStrings.xml",
            "xl/styles.xml",
        } or normalized.startswith("xl/worksheets/sheet")
    return True


def _planning_image_dimensions(
    path: Path,
    suffix: str,
) -> tuple[int, int]:
    if suffix not in IMAGE_EXTENSIONS:
        return 0, 0
    try:
        from PIL import Image

        with Image.open(path) as image:
            return max(0, int(image.width)), max(0, int(image.height))
    except Exception:
        return 0, 0


def _hang_for_validation() -> None:
    """Test-only fault point; the owning watchdog must terminate this process."""

    import time

    while True:
        time.sleep(1)


class _BatchProgressReporter:
    def __init__(
        self,
        reporter: PlanningProgressReporter,
        *,
        file_index: int,
        file_total: int,
        byte_offset: int,
    ) -> None:
        self.reporter = reporter
        self.file_index = file_index
        self.file_total = file_total
        self.byte_offset = byte_offset

    def advance(self, **values: object) -> None:
        local_bytes = max(0, int(values.get("bytes_read") or 0))
        local_cursor = str(values.get("cursor") or "")
        self.reporter.advance(
            phase=str(values.get("phase") or "source_prepare"),
            completed=max(0, self.file_index - 1),
            total=self.file_total,
            cursor=f"file:{self.file_index}:{local_cursor}",
            bytes_read=self.byte_offset + local_bytes,
            output_blocks=max(0, int(values.get("output_blocks") or 0)),
            checkpoint_version=self.byte_offset + local_bytes,
            detail=str(values.get("detail") or ""),
        )
