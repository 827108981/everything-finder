from __future__ import annotations

import hashlib
import json
from pathlib import Path


MODEL_MANIFEST_NAME = "manifest.json"


class OcrModelIntegrityError(RuntimeError):
    pass


def model_manifest_fingerprint(models_dir: Path) -> str:
    """Return the build-time model digest without hashing model weights at runtime."""

    if not models_dir.exists():
        return "models-missing"
    manifest_path = models_dir / MODEL_MANIFEST_NAME
    if not manifest_path.is_file():
        return _stat_fingerprint(models_dir)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("manifest_version") or 0) < 1:
            raise OcrModelIntegrityError("OCR 模型清单版本无效")
        combined_digest = str(payload.get("combined_digest") or "").strip()
        files = payload.get("files")
        if not combined_digest or not isinstance(files, list) or not files:
            raise OcrModelIntegrityError("OCR 模型清单缺少摘要或文件列表")
        for item in files:
            if not isinstance(item, dict):
                raise OcrModelIntegrityError("OCR 模型清单文件项无效")
            relative = str(item.get("path") or "")
            expected_size = int(item.get("size") or -1)
            candidate = models_dir / Path(relative)
            try:
                actual_size = candidate.stat().st_size
            except OSError as exc:
                raise OcrModelIntegrityError(f"OCR 模型文件缺失：{relative}") from exc
            if actual_size != expected_size:
                raise OcrModelIntegrityError(f"OCR 模型文件大小不匹配：{relative}")
        return combined_digest
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, OcrModelIntegrityError):
            raise
        raise OcrModelIntegrityError(f"OCR 模型清单不可读：{exc}") from exc


def _stat_fingerprint(models_dir: Path) -> str:
    """Developer-tree fallback used before a manifest has been generated."""

    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in models_dir.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        ),
        key=lambda path: path.relative_to(models_dir).as_posix(),
    )
    if not files:
        return "models-incomplete"
    for path in files:
        stat = path.stat()
        digest.update(path.relative_to(models_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
    return "models-stat:" + digest.hexdigest()
