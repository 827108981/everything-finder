from __future__ import annotations

import pytest

from tools.verify_distribution import DistributionIntegrityError, verify_manifest, write_manifest


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
