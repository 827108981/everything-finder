from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

RELEASE_INFO_DIR = "发行资料"
MANIFEST_NAME = Path(RELEASE_INFO_DIR) / "SHA256SUMS.txt"
USER_GUIDE_NAME = "本地多格式全文搜索工具-使用说明.pdf"
ROOT_DOCUMENTS = (
    "启动与分发说明.md",
    "首次全量索引性能优化验证报告-20260730.md",
)
BUILD_INFO_NAME = "BUILD-INFO.txt"
FAILURE_DEMO_LAUNCHER = "异常文件保底功能演示.bat"


@dataclass(frozen=True, slots=True)
class VerificationReport:
    files: int
    bytes: int


class DistributionIntegrityError(RuntimeError):
    pass


def stage_release_files(package_dir: Path, project_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    info_dir = package_dir / RELEASE_INFO_DIR
    info_dir.mkdir(parents=True, exist_ok=True)
    for name in ROOT_DOCUMENTS:
        source = project_dir / "docs" / name
        if not source.is_file():
            raise DistributionIntegrityError(f"Release document is missing: {source}")
        shutil.copy2(source, info_dir / name)
    guide = project_dir / "output" / "pdf" / USER_GUIDE_NAME
    if not guide.is_file():
        raise DistributionIntegrityError(f"PDF user guide is missing: {guide}")
    shutil.copy2(guide, package_dir / USER_GUIDE_NAME)
    demo_launcher = project_dir / "tools" / "launch_failure_fallback_demo.bat"
    if not demo_launcher.is_file():
        raise DistributionIntegrityError(
            f"Failure fallback demo launcher is missing: {demo_launcher}"
        )
    shutil.copy2(demo_launcher, package_dir / FAILURE_DEMO_LAUNCHER)
    write_build_info(package_dir, project_dir)


def write_build_info(package_dir: Path, project_dir: Path) -> Path:
    commit = _git_text(project_dir, "rev-parse", "HEAD") or "unavailable"
    status = _git_text(project_dir, "status", "--porcelain")
    source_state = "clean" if status == "" else "working-tree-with-changes"
    app_version = _read_app_version(project_dir)
    info_path = package_dir / RELEASE_INFO_DIR / BUILD_INFO_NAME
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        "\n".join(
            (
                f"application_version={app_version}",
                f"source_commit={commit}",
                f"source_state={source_state}",
                f"built_at_utc={dt.datetime.now(dt.UTC).isoformat()}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return info_path


def _read_app_version(project_dir: Path) -> str:
    version_path = project_dir / "local_full_text_search" / "version.py"
    try:
        source = version_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DistributionIntegrityError(
            f"Application version file is missing: {version_path}"
        ) from exc
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', source, re.MULTILINE)
    if match is None:
        raise DistributionIntegrityError(
            f"Application version is invalid: {version_path}"
        )
    return match.group(1)


def _git_text(project_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_dir,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def write_manifest(package_dir: Path) -> VerificationReport:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
