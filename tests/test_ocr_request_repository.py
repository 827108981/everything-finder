from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_manager import (
    IndexManager,
    ParseJob,
    ParseOutcome,
)
from local_full_text_search.core.ocr_scheduler import OcrRequestRepository


def _file_and_task(
    db: DatabaseManager,
    root: Path,
    name: str,
    run_id: str,
) -> tuple[int, int, Path]:
    source = root / name
    source.write_bytes((name * 20).encode("utf-8"))
    root_id = int(db.list_roots()[0]["id"])
    file_id, _changed = db.upsert_file_metadata(root_id, source)
    task_id = db.create_parse_tasks(
        [(file_id, run_id, "ocr", 100)]
    )[0]
    return file_id, task_id, source


def test_p0_02r_durable_ocr_requests_are_idempotent_and_source_fair(
    tmp_path: Path,
) -> None:
    db = DatabaseManager(tmp_path / "index.db")
    db.initialize()
    root = tmp_path / "files"
    root.mkdir()
    db.add_root(root)
    first_file, first_task, first_path = _file_and_task(
        db,
        root,
        "manual.pdf",
        "run",
    )
    second_file, second_task, second_path = _file_and_task(
        db,
        root,
        "image.jpg",
        "run",
    )
    repository = OcrRequestRepository(db)
    first = repository.enqueue(
        file_id=first_file,
        parent_task_id=first_task,
        source_kind="pdf_page",
        source_unit="page:1",
        image_spool_path=first_path,
        content_sha256="a" * 64,
        width=1200,
        height=1800,
        config_fingerprint="cfg",
        priority=100,
        pixel_cost=2_160_000,
    )
    duplicate = repository.enqueue(
        file_id=first_file,
        parent_task_id=first_task,
        source_kind="pdf_page",
        source_unit="page:1",
        image_spool_path=first_path,
        content_sha256="a" * 64,
        width=1200,
        height=1800,
        config_fingerprint="cfg",
        priority=100,
        pixel_cost=2_160_000,
    )
    repository.enqueue(
        file_id=first_file,
        parent_task_id=first_task,
        source_kind="pdf_page",
        source_unit="page:2",
        image_spool_path=first_path,
        content_sha256="b" * 64,
        width=1200,
        height=1800,
        config_fingerprint="cfg-2",
        priority=90,
        pixel_cost=2_160_000,
    )
    image_request = repository.enqueue(
        file_id=second_file,
        parent_task_id=second_task,
        source_kind="image",
        source_unit="image",
        image_spool_path=second_path,
        content_sha256="c" * 64,
        width=800,
        height=600,
        config_fingerprint="cfg",
        priority=80,
        pixel_cost=480_000,
    )

    claims = repository.claim(
        "worker-1",
        limit=2,
        max_pixels=10_000_000,
        lease_seconds=60,
    )

    assert duplicate == first
    assert {claim.request_id for claim in claims} == {
        first,
        image_request,
    }
    assert {claim.file_id for claim in claims} == {
        first_file,
        second_file,
    }


def test_p0_02r_expired_lease_recovers_and_confirmation_validates_spool(
    tmp_path: Path,
) -> None:
    db = DatabaseManager(tmp_path / "index.db")
    db.initialize()
    root = tmp_path / "files"
    root.mkdir()
    db.add_root(root)
    file_id, task_id, source = _file_and_task(
        db,
        root,
        "scan.jpg",
        "run",
    )
    repository = OcrRequestRepository(db)
    request_id = repository.enqueue(
        file_id=file_id,
        parent_task_id=task_id,
        source_kind="image",
        source_unit="image",
        image_spool_path=source,
        content_sha256="d" * 64,
        width=100,
        height=100,
        config_fingerprint="cfg",
        priority=100,
        pixel_cost=10_000,
    )
    repository.claim(
        "old-worker",
        limit=1,
        max_pixels=20_000,
        lease_seconds=60,
    )
    repository.expire_all_leases_for_validation()
    recovered = repository.claim(
        "new-worker",
        limit=1,
        max_pixels=20_000,
        lease_seconds=60,
    )
    result = tmp_path / "ocr-result.json"
    result.write_text('{"text":"confirmed"}', encoding="utf-8")
    digest = hashlib.sha256(result.read_bytes()).hexdigest()

    with pytest.raises(ValueError):
        repository.confirm(
            request_id,
            worker_id="old-worker",
            result_spool_path=result,
            result_digest=digest,
        )
    repository.confirm(
        request_id,
        worker_id="new-worker",
        result_spool_path=result,
        result_digest=digest,
    )

    assert recovered[0].request_id == request_id
    row = repository.get(request_id)
    assert row["status"] == "confirmed"
    assert row["lease_owner"] == ""


def test_p0_02r_database_restart_requeues_unconfirmed_ocr_request(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "index.db"
    db = DatabaseManager(database_path)
    db.initialize()
    root = tmp_path / "files"
    root.mkdir()
    db.add_root(root)
    file_id, task_id, source = _file_and_task(
        db,
        root,
        "restart.jpg",
        "run",
    )
    repository = OcrRequestRepository(db)
    request_id = repository.enqueue(
        file_id=file_id,
        parent_task_id=task_id,
        source_kind="image",
        source_unit="image",
        image_spool_path=source,
        content_sha256="e" * 64,
        width=100,
        height=100,
        config_fingerprint="cfg",
        priority=100,
        pixel_cost=10_000,
    )
    repository.claim_specific(
        [request_id],
        worker_id="crashed-run",
        lease_seconds=600,
    )

    restarted = DatabaseManager(database_path)
    restarted.initialize()
    row = OcrRequestRepository(restarted).get(request_id)

    assert row["status"] == "queued"
    assert row["lease_owner"] == ""


def test_p0_02r_manager_confirms_only_owned_validated_ocr_spool(
    tmp_path: Path,
) -> None:
    db = DatabaseManager(tmp_path / "index.db")
    db.initialize()
    root = tmp_path / "files"
    root.mkdir()
    db.add_root(root)
    file_id, task_id, source = _file_and_task(
        db,
        root,
        "managed.jpg",
        "run",
    )
    repository = OcrRequestRepository(db)
    request_id = repository.enqueue(
        file_id=file_id,
        parent_task_id=task_id,
        source_kind="image",
        source_unit="image",
        image_spool_path=source,
        content_sha256="f" * 64,
        width=100,
        height=100,
        config_fingerprint="cfg",
        priority=100,
        pixel_cost=10_000,
    )
    repository.claim_specific(
        [request_id],
        worker_id="run",
        lease_seconds=60,
    )
    spool = tmp_path / "result.pickle"
    spool.write_bytes(b"validated-result")
    digest = hashlib.sha256(spool.read_bytes()).hexdigest()
    job = ParseJob(
        file_id=file_id,
        file_path=source,
        task_id=task_id,
        lane="ocr",
        ocr_request_id=request_id,
        ocr_request_owner="run",
    )
    outcome = ParseOutcome(
        file_id=file_id,
        file_path=source,
        blocks=[],
        parser_name="image_ocr",
        status="success",
        spool_path=spool,
        spool_checksum=digest,
    )

    IndexManager(db, AppSettings())._update_ocr_request_for_outcome(
        job,
        outcome,
    )

    assert repository.get(request_id)["status"] == "confirmed"
