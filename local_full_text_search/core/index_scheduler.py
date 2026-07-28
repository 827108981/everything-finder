from __future__ import annotations

from pathlib import Path


BASE_COST = {
    ".txt": 1.0,
    ".log": 1.0,
    ".csv": 1.2,
    ".md": 1.0,
    ".json": 1.1,
    ".xml": 1.2,
    ".ini": 1.0,
    ".docx": 5.0,
    ".xlsx": 6.0,
    ".xlsm": 6.5,
    ".pptx": 5.5,
    ".pdf": 7.0,
    ".zip": 9.0,
    ".doc": 12.0,
    ".xls": 12.0,
    ".ppt": 12.0,
    ".jpg": 18.0,
    ".jpeg": 18.0,
    ".png": 18.0,
    ".bmp": 18.0,
    ".tif": 22.0,
    ".tiff": 22.0,
}


def estimate_parse_cost(path: Path, size_bytes: int, relevant_bytes: int = 0) -> float:
    suffix = path.suffix.lower()
    base = BASE_COST.get(suffix, 0.5)
    mib = max(0, size_bytes) / (1024 * 1024)
    relevant_mib = max(0, relevant_bytes) / (1024 * 1024)
    if suffix in {".docx", ".xlsx", ".xlsm", ".pptx"}:
        return base + relevant_mib * 2.0 + mib * 0.02
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return base + mib * 4.0
    if suffix in {".doc", ".xls", ".ppt"}:
        return base + mib * 1.5
    if suffix == ".zip":
        return base + relevant_mib * 1.2
    if suffix == ".pdf":
        return base + mib * 0.8
    return base + mib * 0.25
