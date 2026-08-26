from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


EXPECTED_PAGES = 8
EXPECTED_TEXT = (
    "图文使用说明",
    "首次建立索引",
    "安全暂停与模式切换",
    "搜索、黄显与打开命中位置",
    "未成功索引与保底开放",
    "设置与数据库迁移",
    "不需要联网",
    "视频文件不解析",
)
REQUIRED_ASSETS = (
    "search_results.png",
    "index_running.png",
    "index_paused.png",
    "failed_files.png",
    "settings_database.png",
    "search_controls.png",
    "index_status.png",
    "database_settings_detail.png",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_page_images(page: object) -> int:
    resources = page.get("/Resources") or {}
    xobjects = resources.get("/XObject") or {}
    count = 0
    for reference in xobjects.values():
        candidate = reference.get_object()
        if candidate.get("/Subtype") == "/Image":
            count += 1
    return count


def validate(pdf_path: Path, assets: Path) -> dict[str, object]:
    if not pdf_path.is_file():
        raise RuntimeError(f"PDF is missing: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) != EXPECTED_PAGES:
        raise RuntimeError(
            f"Unexpected page count: {len(reader.pages)} != {EXPECTED_PAGES}"
        )
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    missing_text = [item for item in EXPECTED_TEXT if item not in extracted]
    if missing_text:
        raise RuntimeError("Required guide text is missing: " + ", ".join(missing_text))
    if "1.3.7" in extracted:
        raise RuntimeError("The guide still contains the stale 1.3.7 version")
    image_count = sum(count_page_images(page) for page in reader.pages)
    if image_count < EXPECTED_PAGES:
        raise RuntimeError(f"The guide is not image-rich enough: images={image_count}")
    title = str((reader.metadata or {}).get("/Title") or "")
    if "图文使用说明" not in title:
        raise RuntimeError(f"Unexpected PDF title metadata: {title}")

    manifest_path = assets / "assets-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = {
        str(row["name"]): row
        for row in manifest.get("files", [])
        if isinstance(row, dict) and row.get("name")
    }
    for name in REQUIRED_ASSETS:
        path = assets / name
        if not path.is_file() or name not in manifest_files:
            raise RuntimeError(f"Required screenshot asset is missing: {name}")
        expected_hash = str(manifest_files[name].get("sha256") or "")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Screenshot manifest hash mismatch: {name}")
        with Image.open(path) as image:
            if image.width < 700 or image.height < 90:
                raise RuntimeError(
                    f"Screenshot asset is undersized: {name}={image.width}x{image.height}"
                )

    return {
        "passed": True,
        "pages": len(reader.pages),
        "embedded_images": image_count,
        "pdf_bytes": pdf_path.stat().st_size,
        "pdf_sha256": sha256_file(pdf_path),
        "assets": len(REQUIRED_ASSETS),
        "title": title,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the illustrated PDF user guide")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.pdf.resolve(), args.assets.resolve())
    except Exception as exc:
        print(f"USER_GUIDE_VALIDATION_FAILED: {type(exc).__name__}: {exc}")
        return 1
    print("USER_GUIDE_VALIDATION_OK " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
