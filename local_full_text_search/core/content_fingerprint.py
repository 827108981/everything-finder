from __future__ import annotations

import hashlib
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path


OOXML_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".xlsm"}


@dataclass(frozen=True, slots=True)
class ContentFingerprint:
    key: str
    relevant_bytes: int
    method: str


def fingerprint_file(path: Path) -> ContentFingerprint:
    suffix = path.suffix.lower()
    if suffix in OOXML_EXTENSIONS:
        try:
            return _fingerprint_zip_directory(path, suffix)
        except (OSError, zipfile.BadZipFile):
            pass
    if suffix == ".zip":
        try:
            return _fingerprint_zip_directory(path, suffix)
        except (OSError, zipfile.BadZipFile):
            pass
    stat = path.stat()
    if suffix == ".mp4":
        payload = f"metadata:{stat.st_size}:{stat.st_mtime_ns}:{path.name}".encode("utf-8")
        return ContentFingerprint("metadata:" + hashlib.sha256(payload).hexdigest(), 0, "metadata")
    if stat.st_size <= 64 * 1024 * 1024 or suffix in {".doc", ".xls", ".ppt"}:
        return ContentFingerprint("sha256:" + _sha256_file(path), stat.st_size, "sha256")
    return ContentFingerprint("sample:" + _sampled_sha256(path, stat.st_size), stat.st_size, "sampled_sha256")


def _fingerprint_zip_directory(path: Path, suffix: str) -> ContentFingerprint:
    digest = hashlib.sha256()
    relevant_bytes = 0
    with zipfile.ZipFile(path) as archive:
        infos = sorted((info for info in archive.infolist() if not info.is_dir()), key=lambda info: info.filename)
        for info in infos:
            encoded_name = info.filename.encode("utf-8", errors="surrogatepass")
            digest.update(len(encoded_name).to_bytes(4, "little"))
            digest.update(encoded_name)
            digest.update(int(info.CRC).to_bytes(4, "little", signed=False))
            digest.update(int(info.file_size).to_bytes(8, "little", signed=False))
            digest.update(int(info.compress_size).to_bytes(8, "little", signed=False))
            if _is_relevant_entry(suffix, info.filename):
                relevant_bytes += int(info.compress_size)
    return ContentFingerprint(f"zipdir:{digest.hexdigest()}", relevant_bytes, "zip_directory")


def _is_relevant_entry(suffix: str, name: str) -> bool:
    normalized = name.replace("\\", "/")
    if suffix == ".docx":
        return normalized == "word/document.xml" or normalized.startswith("word/header") or normalized.startswith("word/footer")
    if suffix == ".pptx":
        return normalized.startswith("ppt/slides/slide") or normalized.startswith("ppt/notesSlides/notesSlide")
    if suffix in {".xlsx", ".xlsm"}:
        return normalized in {"xl/workbook.xml", "xl/sharedStrings.xml", "xl/styles.xml"} or normalized.startswith("xl/worksheets/sheet")
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return a full raw-byte digest for exact cross-source deduplication."""

    return _sha256_file(path)


def fingerprint_file_with_spool(
    path: Path,
    spool_dir: Path,
) -> tuple[ContentFingerprint, Path | None, int]:
    """Fingerprint a file and retain the same full-hash read for parsing."""

    suffix = path.suffix.lower()
    if suffix in OOXML_EXTENSIONS or suffix == ".zip":
        try:
            return _fingerprint_zip_directory(path, suffix), None, 0
        except (OSError, zipfile.BadZipFile):
            pass
    stat = path.stat()
    if suffix == ".mp4":
        payload = f"metadata:{stat.st_size}:{stat.st_mtime_ns}:{path.name}".encode("utf-8")
        return (
            ContentFingerprint(
                "metadata:" + hashlib.sha256(payload).hexdigest(),
                0,
                "metadata",
            ),
            None,
            0,
        )
    if stat.st_size <= 64 * 1024 * 1024 or suffix in {".doc", ".xls", ".ppt"}:
        digest, spool_path, bytes_read = sha256_file_to_spool(path, spool_dir)
        return (
            ContentFingerprint(f"sha256:{digest}", stat.st_size, "sha256"),
            spool_path,
            bytes_read,
        )
    return (
        ContentFingerprint(
            "sample:" + _sampled_sha256(path, stat.st_size),
            stat.st_size,
            "sampled_sha256",
        ),
        None,
        0,
    )


def sha256_file_to_spool(path: Path, spool_dir: Path) -> tuple[str, Path, int]:
    spool_dir.mkdir(parents=True, exist_ok=True)
    target = spool_dir / f"{uuid.uuid4().hex}{path.suffix.lower()}"
    temporary = target.with_suffix(target.suffix + ".tmp")
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with path.open("rb") as source, temporary.open("wb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                output.write(chunk)
                bytes_read += len(chunk)
        temporary.replace(target)
        return digest.hexdigest(), target, bytes_read
    except Exception:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise


def _sampled_sha256(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    digest.update(size.to_bytes(8, "little", signed=False))
    sample_size = 4 * 1024 * 1024
    offsets = (0, max(0, size // 2 - sample_size // 2), max(0, size - sample_size))
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset, os.SEEK_SET)
            digest.update(offset.to_bytes(8, "little", signed=False))
            digest.update(stream.read(sample_size))
    return digest.hexdigest()
