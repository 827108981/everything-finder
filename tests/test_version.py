from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search import version


def test_source_checkout_does_not_use_stale_release_timestamp(monkeypatch) -> None:
    monkeypatch.delenv("LFTS_BUILD_TIMESTAMP", raising=False)
    assert version._load_build_timestamp() == "dev"


def test_configured_build_timestamp_is_used(monkeypatch) -> None:
    monkeypatch.setenv("LFTS_BUILD_TIMESTAMP", "20260922123456")
    assert version._load_build_timestamp() == "20260922123456"


def test_frozen_metadata_timestamp_is_read_from_package(tmp_path: Path, monkeypatch) -> None:
    metadata = tmp_path / "build_info.json"
    metadata.write_text(
        json.dumps({"build_timestamp": "20260922112233"}),
        encoding="utf-8",
    )
    monkeypatch.delenv("LFTS_BUILD_TIMESTAMP", raising=False)
    monkeypatch.setattr(version.sys, "frozen", True, raising=False)
    monkeypatch.setattr(version.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert version._load_build_timestamp() == "20260922112233"
