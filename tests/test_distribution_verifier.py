from __future__ import annotations

import pytest

from tools.verify_distribution import (
    BUILD_INFO_NAME,
    MANIFEST_NAME,
    RELEASE_INFO_DIR,
    DistributionIntegrityError,
    stage_release_files,
    verify_manifest,
    write_manifest,
)


def test_manifest_round_trip_and_detects_tampering(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    payload = package / "_internal" / "runtime.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"original")

    report = write_manifest(package)

    assert report.files == 1
    assert verify_manifest(package).files == 1
    payload.write_bytes(b"modified")
    with pytest.raises(DistributionIntegrityError, match="SHA-256 mismatch"):
        verify_manifest(package)


def test_release_documents_do_not_clutter_the_package_root(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "_internal").mkdir()
    (package / "本地多格式全文搜索工具.exe").write_bytes(b"exe")
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("启动与分发说明.md", "首次全量索引性能优化验证报告-20260730.md"):
        (docs / name).write_text(name, encoding="utf-8")
    pdf_dir = tmp_path / "output" / "pdf"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "本地多格式全文搜索工具-使用说明.pdf").write_bytes(b"%PDF-test")
    package_source = tmp_path / "local_full_text_search"
    package_source.mkdir()
    (package_source / "version.py").write_text(
        '__version__ = "test"\n',
        encoding="utf-8",
    )
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "launch_failure_fallback_demo.bat").write_text(
        "@echo off\n",
        encoding="ascii",
    )

    stage_release_files(package, tmp_path)

    assert (package / "本地多格式全文搜索工具-使用说明.pdf").is_file()
    assert (package / RELEASE_INFO_DIR / BUILD_INFO_NAME).is_file()
    assert not (package / BUILD_INFO_NAME).exists()
    assert set(path.name for path in package.iterdir()) == {
        "_internal",
        "本地多格式全文搜索工具.exe",
        "本地多格式全文搜索工具-使用说明.pdf",
        "异常文件保底功能演示.bat",
        RELEASE_INFO_DIR,
    }
    write_manifest(package)
    assert (package / MANIFEST_NAME).is_file()
