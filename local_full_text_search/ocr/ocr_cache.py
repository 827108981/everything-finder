from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from local_full_text_search.config.constants import OCR_CACHE_DIR, OCR_MODELS_DIR
from local_full_text_search.ocr.ocr_engine import OCR_MODEL_FILES, OCR_MODEL_NAMES, OcrResult


class OcrCache:
    def __init__(self, cache_dir: Path = OCR_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for_file(self, path: Path, *, namespace: str = "") -> str:
        digest = hashlib.sha256()
        digest.update(namespace.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load(self, key: str) -> OcrResult | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return OcrResult(data.get("text", ""), data.get("confidence"), data.get("extra", {}))
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, key: str, result: OcrResult) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(
                {"text": result.text, "confidence": result.confidence, "extra": result.extra},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


@lru_cache(maxsize=1)
def ocr_models_fingerprint() -> str:
    digest = hashlib.sha256()
    if not OCR_MODELS_DIR.exists():
        return "models-missing"
    for model_name in OCR_MODEL_NAMES:
        for file_name in OCR_MODEL_FILES:
            path = OCR_MODELS_DIR / model_name / file_name
            try:
                stat = path.stat()
            except OSError:
                return "models-incomplete"
            digest.update(f"{model_name}/{file_name}".encode("ascii"))
            digest.update(f":{stat.st_size}:".encode("ascii"))
            digest.update(path.read_bytes())
    return digest.hexdigest()
