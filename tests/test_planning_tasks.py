from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.planning_tasks import (
    PreparedFileMetadata,
    discover_file_batches,
    fingerprint_source,
    prepare_zip_member_task,
    scan_pdf_document_task,
    scan_zip_manifest_task,
    stat_file_batch,
)
from local_full_text_search.core.planning_worker import (
    PlanningProgress,
    PlanningProgressReporter,
    RecoverablePlanningRunner,
)


def _collect_progress(target: list[PlanningProgress]):
    def collect(value: PlanningProgress) -> None:
        target.append(value)

    return collect


def test_s0_01r_discovery_and_stat_run_in_spawned_planning_processes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "first.txt").write_text("first", encoding="utf-8")
    (root / "second.pdf").write_bytes(b"%PDF")
    (root / "ignored.bin").write_bytes(b"ignored")
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=2,
        startup_timeout_seconds=2,
    )
    progress: list[PlanningProgress] = []

    batches = list(
        runner.stream(
            "directory_enumeration",
            discover_file_batches,
            root,
            True,
            AppSettings(enable_ocr=False).to_dict(),
            tmp_path / "run_control",
            1,
            progress_callback=_collect_progress(progress),
        )
    )
    discovered = [Path(item) for batch in batches for item in batch]
    metadata = runner.run(
        "file_stat",
        stat_file_batch,
        [str(path) for path in discovered],
        False,
        progress_callback=_collect_progress(progress),
    )

    assert {path.name for path in discovered} == {"first.txt", "second.pdf"}
    assert metadata.errors == ()
    assert all(isinstance(item, PreparedFileMetadata) for item in metadata.metadata)
    assert {item.size_bytes for item in metadata.metadata} == {4, 5}
    assert {item.worker_pid for item in metadata.metadata}.isdisjoint({0})
    assert all(item.worker_pid != 0 for item in metadata.metadata)
    assert {item.phase for item in progress} >= {
        "directory_enumeration",
        "file_stat",
    }


def test_s0_01r_hash_reports_real_bytes_and_publishes_atomic_spool(
    tmp_path: Path,
) -> None:
    source = tmp_path / "payload.txt"
    payload = b"A" * (3 * 1024 * 1024 + 17)
    source.write_bytes(payload)
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=2,
        startup_timeout_seconds=2,
    )
    progress: list[PlanningProgress] = []

    result = runner.run(
        "content_hash",
        fingerprint_source,
        source,
        tmp_path / "spool",
        progress_callback=_collect_progress(progress),
    )

    assert result.fingerprint.key.startswith("sha256:")
    assert result.bytes_read == len(payload)
    assert result.spool_path is not None
    assert result.spool_path.read_bytes() == payload
    byte_marks = [
        item.bytes_read for item in progress if item.phase == "content_hash"
    ]
    assert byte_marks == sorted(set(byte_marks))
    assert byte_marks[-1] == len(payload)
    assert not list((tmp_path / "spool").glob("*.tmp"))


def test_physical_source_hash_can_avoid_redundant_spool_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "payload.pdf"
    payload = b"PHYSICAL_SOURCE" * 4096
    source.write_bytes(payload)
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=2,
        startup_timeout_seconds=2,
    )

    result = runner.run(
        "content_hash",
        fingerprint_source,
        source,
        tmp_path / "spool",
        False,
        None,
        False,
    )

    assert result.fingerprint.key == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )
    assert result.bytes_read == len(payload)
    assert result.spool_path is None
    assert not list((tmp_path / "spool").glob("*"))


def test_pdf_scan_uses_meaningful_text_for_ocr_classification(
    tmp_path: Path,
) -> None:
    import fitz

    source = tmp_path / "image-heavy.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "." * 40 + "ABC")
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 2, 2), False)
    pixmap.clear_with(255)
    page.insert_image(fitz.Rect(72, 90, 90, 108), pixmap=pixmap)
    document.save(source)
    document.close()
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=2,
        startup_timeout_seconds=2,
    )

    result = runner.run("pdf_scan", scan_pdf_document_task, source)

    assert len(result.pages) == 1
    assert result.pages[0].requires_ocr is True


def test_s0_01r_zip_manifest_reports_member_semantic_progress(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("one.txt", "one")
        output.writestr("two.txt", "two")
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=2,
        startup_timeout_seconds=2,
    )
    progress: list[PlanningProgress] = []

    manifest = runner.run(
        "zip_manifest",
        scan_zip_manifest_task,
        archive,
        AppSettings(enable_ocr=False).to_dict(),
        tmp_path / "run_control",
        progress_callback=_collect_progress(progress),
    )

    assert len(manifest.members) == 2
    member_progress = [
        item for item in progress if item.phase == "zip_manifest"
    ]
    assert member_progress
    assert member_progress[-1].completed == 2
    assert member_progress[-1].cursor == "member:2"


def test_s0_01r_zip_member_prepare_streams_to_atomic_spool(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "sample.zip"
    payload = b"member payload" * 100_000
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("nested/member.txt", payload)
    with zipfile.ZipFile(archive) as source:
        info = source.infolist()[0]
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=2,
        startup_timeout_seconds=2,
    )
    progress: list[PlanningProgress] = []

    result = runner.run(
        "zip_member_prepare",
        prepare_zip_member_task,
        archive,
        0,
        "nested/member.txt",
        len(payload),
        int(info.CRC),
        ".txt",
        tmp_path / "spool",
        progress_callback=_collect_progress(progress),
    )

    assert result.spool_path.read_bytes() == payload
    assert result.bytes_read == len(payload)
    assert result.sha256
    assert progress[-1].phase == "zip_member_prepare"
    assert progress[-1].bytes_read == len(payload)
