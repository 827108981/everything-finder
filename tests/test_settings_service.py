from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search.services.settings_service import SettingsService


def test_legacy_settings_preserve_explicit_ocr_choices(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "enable_ocr": True,
                "ocr_images": True,
                "ocr_scanned_pdf": False,
                "parser_workers": 3,
            }
        ),
        encoding="utf-8",
    )

    settings = SettingsService(path).load()

    assert settings.enable_ocr is True
    assert settings.ocr_images is True
    assert settings.ocr_scanned_pdf is False
    assert settings.parser_workers == 3


def test_legacy_settings_are_persisted_at_current_version(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"ocr_scanned_pdf": false}', encoding="utf-8")

    settings = SettingsService(path).load()
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert settings.settings_version == 9
    assert persisted["settings_version"] == 9
    assert persisted["ocr_scanned_pdf"] is False
    assert "process_parser_workers" in persisted
    assert "pdf_parser_workers" in persisted


def test_planning_watchdog_settings_are_migrated_and_persisted(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"settings_version": 7}', encoding="utf-8")

    settings = SettingsService(path).load()
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert settings.planning_no_progress_timeout_seconds == 300
    assert settings.planning_startup_timeout_seconds == 30
    assert settings.planning_discovery_batch_size == 128
    assert persisted["planning_no_progress_timeout_seconds"] == 300
    assert persisted["planning_startup_timeout_seconds"] == 30
    assert persisted["planning_discovery_batch_size"] == 128


def test_ocr_microbatch_limits_are_migrated_and_persisted(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"settings_version": 8}', encoding="utf-8")

    settings = SettingsService(path).load()
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert settings.settings_version == 9
    assert settings.ocr_microbatch_parent_jobs == 4
    assert settings.ocr_microbatch_max_requests == 64
    assert settings.ocr_microbatch_max_pixels == 8_000_000
    assert settings.ocr_microbatch_memory_mb == 256
    assert settings.ocr_microbatch_wait_ms == 30
    assert persisted["ocr_microbatch_parent_jobs"] == 4


def test_v3_default_ocr_size_is_migrated_to_mobile_model_limit(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"settings_version": 3, "max_ocr_image_side": 2400}),
        encoding="utf-8",
    )

    settings = SettingsService(path).load()

    assert settings.max_ocr_image_side == 960


def test_v3_custom_ocr_size_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"settings_version": 3, "max_ocr_image_side": 1600}),
        encoding="utf-8",
    )

    settings = SettingsService(path).load()

    assert settings.max_ocr_image_side == 1600
