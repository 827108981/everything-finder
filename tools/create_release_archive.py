from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

MANIFEST_PATH = Path("发行资料") / "SHA256SUMS.txt"


def create_archive(package_dir: Path, archive_path: Path) -> tuple[int, int]:
    package_dir = package_dir.resolve()
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.unlink(missing_ok=True)
    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for source in sorted(package_dir.rglob("*")):
            if not source.is_file():
                continue
            relative = Path(package_dir.name) / source.relative_to(package_dir)
            archive.write(source, relative.as_posix())
            file_count += 1
            total_bytes += source.stat().st_size
    return file_count, total_bytes


def verify_archive(archive_path: Path, expected_root: str) -> int:
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if not members:
            raise RuntimeError("release archive contains no files")
        prefix = expected_root.rstrip("/") + "/"
        if any(not item.filename.startswith(prefix) for item in members):
            raise RuntimeError("release archive contains a file outside the package root")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"release archive CRC verification failed: {bad_member}")
        manifest = prefix + MANIFEST_PATH.as_posix()
        if manifest not in archive.namelist():
            raise RuntimeError("release archive is missing SHA256SUMS.txt")
        return len(members)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("archive_path", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.package_dir.is_dir():
        parser.error(f"package directory does not exist: {args.package_dir}")
    try:
        files, total_bytes = create_archive(args.package_dir, args.archive_path)
        if args.verify:
            verified_files = verify_archive(args.archive_path, args.package_dir.name)
            if verified_files != files:
                raise RuntimeError(
                    f"archive file count changed: written={files}; verified={verified_files}"
                )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"RELEASE_ARCHIVE_FAILED: {exc}")
        return 1
    print(
        "RELEASE_ARCHIVE_OK "
        f"files={files} source_bytes={total_bytes} archive={args.archive_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
