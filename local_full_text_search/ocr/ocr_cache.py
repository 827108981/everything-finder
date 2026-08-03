from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from local_full_text_search.config.constants import OCR_CACHE_DIR, OCR_MODELS_DIR
from local_full_text_search.ocr.model_manifest import model_manifest_fingerprint
from local_full_text_search.ocr.ocr_engine import OcrResult


@dataclass(frozen=True, slots=True)
class OcrExactInput:
    """All semantic inputs that make an OCR result exactly reusable.

    Source paths deliberately do not participate in this identity.  Callers
    must build a new source location/ordering wrapper around a cached result.
    """

    content_sha256: str
    width: int
    height: int
    channels: int
    orientation: int
    crop: tuple[int, int, int, int] | None
    dpi: int
    preprocess_version: str
    strategy_version: str
    detection_model_fingerprint: str
    recognition_model_fingerprint: str
    language: str
    options: dict[str, bool | int | float | str]

    def with_changes(self, **changes: Any) -> "OcrExactInput":
        return replace(self, **changes)

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["crop"] = list(self.crop) if self.crop is not None else None
        payload["options"] = dict(sorted(self.options.items()))
        return payload


class OcrCache:
    SCHEMA_VERSION = 2

    def __init__(self, cache_dir: Path = OCR_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir = self.cache_dir / ".active_references"
        self.reference_dir.mkdir(parents=True, exist_ok=True)

    def key_for_file(self, path: Path, *, namespace: str = "") -> str:
        digest = hashlib.sha256()
        digest.update(namespace.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def key_for_digest(self, content_digest: str, *, namespace: str = "") -> str:
        digest = hashlib.sha256()
        digest.update(namespace.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(content_digest).encode("ascii", errors="backslashreplace"))
        return digest.hexdigest()

    def key_for_exact_input(
        self,
        exact_input: OcrExactInput,
        *,
        source_hint: str = "",
    ) -> str:
        del source_hint
        encoded = _canonical_json(exact_input.canonical_payload()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def load(self, key: str) -> OcrResult | None:
        result, _ = self.load_with_status(key)
        return result

    def load_with_status(self, key: str) -> tuple[OcrResult | None, str]:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None, "miss"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                int(data.get("schema_version", 0)) != self.SCHEMA_VERSION
                or str(data.get("cache_key", "")) != key
            ):
                self._discard(path)
                return None, "schema_or_key_mismatch"
            result_data = data.get("result")
            if not isinstance(result_data, dict):
                self._discard(path)
                return None, "invalid_payload"
            actual_digest = hashlib.sha256(
                _canonical_json(result_data).encode("utf-8")
            ).hexdigest()
            if actual_digest != str(data.get("result_digest", "")):
                self._discard(path)
                return None, "checksum_mismatch"
            return (
                OcrResult(
                    str(result_data.get("text", "")),
                    result_data.get("confidence"),
                    result_data.get("extra", {}),
                ),
                "hit",
            )
        except (OSError, json.JSONDecodeError):
            self._discard(path)
            return None, "corrupt"

    def save(self, key: str, result: OcrResult) -> None:
        path = self.cache_dir / f"{key}.json"
        result_data = {
            "text": result.text,
            "confidence": result.confidence,
            "extra": result.extra,
        }
        envelope = {
            "schema_version": self.SCHEMA_VERSION,
            "cache_key": key,
            "result": result_data,
            "result_digest": hashlib.sha256(
                _canonical_json(result_data).encode("utf-8")
            ).hexdigest(),
        }
        temporary = self.cache_dir / f".{key}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(envelope, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt >= 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def reference(self, key: str) -> Iterator[None]:
        """Protect an exact entry from maintenance while a task uses it."""

        # The full key lives in the payload; compact names keep references
        # usable below Windows' legacy 260-character path limit.
        token = uuid.uuid4().hex[:16]
        key_digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:24]
        reference_path = self.reference_dir / f"{key_digest}.{token}.json"
        payload = {
            "key": str(key),
            "pid": os.getpid(),
            "created_at_epoch": time.time(),
        }
        temporary = reference_path.with_suffix(".tmp")
        try:
            temporary.write_text(
                _canonical_json(payload),
                encoding="utf-8",
            )
            os.replace(temporary, reference_path)
            yield
        finally:
            temporary.unlink(missing_ok=True)
            reference_path.unlink(missing_ok=True)

    def prune(
        self,
        *,
        max_entries: int = 10_000,
        max_age_seconds: float | None = None,
        reference_ttl_seconds: float = 24 * 60 * 60,
    ) -> dict[str, Any]:
        """Remove old exact entries without touching live task references."""

        max_entries = max(0, int(max_entries))
        now = time.time()
        active_keys = self._active_reference_keys(
            now=now,
            ttl_seconds=max(1.0, float(reference_ttl_seconds)),
        )
        entries = sorted(
            (
                path
                for path in self.cache_dir.glob("*.json")
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        retained_by_count = set(entries[:max_entries])
        removed_keys: list[str] = []
        active_reference_skips = 0
        for path in entries:
            key = path.stem
            too_old = bool(
                max_age_seconds is not None
                and now - path.stat().st_mtime
                >= max(0.0, float(max_age_seconds))
            )
            over_limit = path not in retained_by_count
            if not over_limit and not too_old:
                continue
            if key in active_keys:
                active_reference_skips += 1
                continue
            try:
                path.unlink(missing_ok=True)
                removed_keys.append(key)
            except OSError:
                continue
        return {
            "removed_count": len(removed_keys),
            "removed_keys": sorted(removed_keys),
            "active_reference_skips": active_reference_skips,
            "active_reference_count": len(active_keys),
            "remaining_count": len(list(self.cache_dir.glob("*.json"))),
        }

    def _active_reference_keys(
        self,
        *,
        now: float,
        ttl_seconds: float,
    ) -> set[str]:
        active: set[str] = set()
        for path in self.reference_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                key = str(payload.get("key") or "")
                pid = int(payload.get("pid") or 0)
                created = float(payload.get("created_at_epoch") or 0.0)
                if (
                    key
                    and now - created <= ttl_seconds
                    and _pid_exists(pid)
                ):
                    active.add(key)
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return active

    @staticmethod
    def _discard(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A locked corrupt entry remains a miss and will be retried later.
            pass


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except ImportError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


@lru_cache(maxsize=1)
def ocr_models_fingerprint() -> str:
    return model_manifest_fingerprint(OCR_MODELS_DIR)
