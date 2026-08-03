from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_NAMES = ("PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec")
MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")


def build_manifest(models_dir: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    combined = hashlib.sha256()
    for model_name in MODEL_NAMES:
        for file_name in MODEL_FILES:
            path = models_dir / model_name / file_name
            digest = _sha256_file(path)
            relative = path.relative_to(models_dir).as_posix()
            size = path.stat().st_size
            files.append({"path": relative, "size": size, "sha256": digest})
            combined.update(relative.encode("utf-8"))
            combined.update(b"\0")
            combined.update(str(size).encode("ascii"))
            combined.update(b"\0")
            combined.update(digest.encode("ascii"))
            combined.update(b"\0")
    return {
        "manifest_version": 1,
        "engine": "PaddleOCR",
        "model_family": "PP-OCRv4-mobile",
        "models": list(MODEL_NAMES),
        "files": files,
        "combined_digest": combined.hexdigest(),
    }


def verify_manifest(models_dir: Path, manifest: dict[str, object]) -> None:
    for item in manifest["files"]:
        path = models_dir / str(item["path"])
        if path.stat().st_size != int(item["size"]):
            raise RuntimeError(f"Model size mismatch: {path}")
        if _sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(f"Model SHA-256 mismatch: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest_path = args.models_dir / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_manifest(args.models_dir, manifest)
        print(f"verified {manifest_path}")
        return 0
    manifest = build_manifest(args.models_dir)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
