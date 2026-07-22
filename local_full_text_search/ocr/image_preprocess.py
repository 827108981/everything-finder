from __future__ import annotations

import hashlib
from pathlib import Path

from local_full_text_search.config.constants import TEMP_DIR


def image_dimensions(image_path: Path) -> tuple[int, int] | None:
    """Read image size without decoding full pixels when Pillow supports it."""

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None


def preprocess_image(image_path: Path, *, max_side: int = 2400) -> Path:
    """Downscale huge images before OCR while keeping the original file untouched."""

    dimensions = image_dimensions(image_path)
    if dimensions is None:
        return image_path
    width, height = dimensions
    if max(width, height) <= max_side:
        return image_path

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    stat = image_path.stat()
    digest = hashlib.sha256(f"{image_path}:{stat.st_size}:{stat.st_mtime_ns}:{max_side}".encode("utf-8")).hexdigest()[:16]
    target = TEMP_DIR / f"ocr_downscaled_{digest}.png"
    if target.exists():
        return target

    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        image.save(target, "PNG", optimize=True)
    return target
