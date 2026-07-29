from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


MANIFEST_NAME = "SHA256SUMS.txt"
ROOT_DOCUMENTS = ("启动与分发说明.md",)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    files: int
    bytes: int


class DistributionIntegrityError(RuntimeError):
    pass


def stage_release_files(package_dir: Path, project_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    for name in ROOT_DOCUMENTS:
        source = project_dir / "docs" / name
        if not source.is_file():
            raise DistributionIntegrityError(f"Release document is missing: {source}")
        shutil.copy2(source, package_dir / name)


def write_manifest(package_dir: Path) -> VerificationReport:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / MANIFEST_NAME
    files = sorted(
        (
            path
            for path in package_dir.rglob("*")
            if path.is_file() and path.resolve() != manifest_path.resolve()
        ),
        key=lambda path: path.relative_to(package_dir).as_posix().lower(),
    )
    lines: list[str] = []
    total_bytes = 0
    for path in files:
        relative_path = path.relative_to(package_dir).as_posix()
        lines.append(f"{sha256_file(path)} *{relative_path}")
        total_bytes += path.stat().st_size
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return VerificationReport(len(files), total_bytes)


def verify_manifest(package_dir: Path) -> VerificationReport:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise DistributionIntegrityError(f"Manifest is missing: {manifest_path}")
    files = 0
    total_bytes = 0
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line:
            continue
        try:
            expected_hash, relative_marker = raw_line.split(" *", 1)
        except ValueError as exc:
            raise DistributionIntegrityError(f"Malformed manifest line {line_number}") from exc
        if len(expected_hash) != 64 or any(char not in "0123456789ABCDEF" for char in expected_hash):
            raise DistributionIntegrityError(f"Invalid SHA-256 on manifest line {line_number}")
        target = (package_dir / relative_marker).resolve()
        if package_dir not in target.parents or not target.is_file():
            raise DistributionIntegrityError(f"Missing packaged file: {relative_marker}")
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            raise DistributionIntegrityError(f"SHA-256 mismatch: {relative_marker}")
        files += 1
        total_bytes += target.stat().st_size
    if files == 0:
        raise DistributionIntegrityError("Manifest has no files")
    return VerificationReport(files, total_bytes)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and verify a frozen release manifest.")
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--stage-release-files", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.write_manifest and not args.verify:
        parser.error("choose --write-manifest, --verify, or both")
    try:
        if args.stage_release_files:
            stage_release_files(args.package_dir, args.project_dir)
        report: VerificationReport | None = None
        if args.write_manifest:
            report = write_manifest(args.package_dir)
        if args.verify:
            report = verify_manifest(args.package_dir)
    except DistributionIntegrityError as exc:
        print(f"DISTRIBUTION_VERIFICATION_FAILED: {exc}")
        return 1
    assert report is not None
    print(f"DISTRIBUTION_VERIFICATION_OK files={report.files} bytes={report.bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
