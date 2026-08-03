from __future__ import annotations

from tools.create_release_archive import MANIFEST_PATH, create_archive, verify_archive


def test_release_archive_round_trip(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manifest = package / MANIFEST_PATH
    manifest.parent.mkdir()
    manifest.write_text("manifest\n", encoding="utf-8")
    internal = package / "_internal"
    internal.mkdir()
    (internal / "runtime.bin").write_bytes(b"runtime")
    archive = tmp_path / "release.zip"

    files, _ = create_archive(package, archive)

    assert files == 2
    assert verify_archive(archive, "package") == 2
