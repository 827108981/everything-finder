from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.pdf_task_graph import (
    PdfPagePlan,
    PdfTaskGraphRepository,
)


def _add_pdf(database: DatabaseManager, root_id: int, path: Path) -> int:
    path.write_bytes(b"%PDF-placeholder")
    file_id, _changed = database.upsert_file_metadata(root_id, path)
    return file_id


def test_p0_01r_pdf_graph_persists_scan_pages_ocr_and_merge_tasks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id = _add_pdf(database, root_id, root / "mixed.pdf")
    graph = PdfTaskGraphRepository(database)

    document_task_id = graph.plan_document(
        file_id=file_id,
        run_id="run-1",
        source_digest="sha256:one",
        parser_version="pdf-v1",
        pages=[
            PdfPagePlan(1, "page-one", 595.0, 842.0, False),
            PdfPagePlan(2, "page-two", 595.0, 842.0, True),
        ],
        ocr_config_fingerprint="ocr-v1",
    )

    with database.connect() as con:
        rows = con.execute(
            """
            SELECT task_type, unit_key, status, parent_task_id
            FROM parse_tasks
            WHERE id = ? OR parent_task_id = ?
            ORDER BY id
            """,
            (document_task_id, document_task_id),
        ).fetchall()
    assert [(row["task_type"], row["unit_key"]) for row in rows] == [
        ("pdf_document", "document"),
        ("pdf_scan", "scan"),
        ("pdf_native_page", "page:1"),
        ("pdf_native_page", "page:2"),
        ("pdf_ocr_page", "page:2"),
        ("document_merge", "merge"),
    ]
    assert rows[1]["status"] == "complete"
    assert all(row["parent_task_id"] == document_task_id for row in rows[1:])


def test_p0_01r_page_claims_are_fair_leased_and_recoverable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    first_id = _add_pdf(database, root_id, root / "first.pdf")
    second_id = _add_pdf(database, root_id, root / "second.pdf")
    graph = PdfTaskGraphRepository(database)
    pages = [
        PdfPagePlan(index, f"page-{index}", 595.0, 842.0, False)
        for index in range(1, 4)
    ]
    graph.plan_document(
        file_id=first_id,
        run_id="run-1",
        source_digest="sha256:first",
        parser_version="pdf-v1",
        pages=pages,
        ocr_config_fingerprint="ocr-v1",
    )
    graph.plan_document(
        file_id=second_id,
        run_id="run-1",
        source_digest="sha256:second",
        parser_version="pdf-v1",
        pages=pages,
        ocr_config_fingerprint="ocr-v1",
    )

    claims = graph.claim_page_tasks("worker-a", limit=4, lease_seconds=30)

    assert [claim.file_id for claim in claims] == [
        first_id,
        second_id,
        first_id,
        second_id,
    ]
    assert all(claim.lease_owner == "worker-a" for claim in claims)
    assert graph.claim_page_tasks("worker-b", limit=2, lease_seconds=30)
    graph.expire_all_leases_for_validation()
    recovered = graph.claim_page_tasks("worker-c", limit=6, lease_seconds=30)
    assert recovered
    assert all(claim.lease_owner == "worker-c" for claim in recovered)


def test_p0_01r_merge_is_blocked_by_failed_or_unconfirmed_page_and_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id = _add_pdf(database, root_id, root / "manual.pdf")
    graph = PdfTaskGraphRepository(database)
    document_task_id = graph.plan_document(
        file_id=file_id,
        run_id="run-1",
        source_digest="sha256:manual",
        parser_version="pdf-v1",
        pages=[
            PdfPagePlan(1, "one", 595.0, 842.0, False),
            PdfPagePlan(2, "two", 595.0, 842.0, False),
        ],
        ocr_config_fingerprint="ocr-v1",
    )
    claims = graph.claim_page_tasks("worker", limit=2, lease_seconds=30)
    graph.confirm_page_task(
        claims[0].task_id,
        result_spool_path=tmp_path / "page-1.pickle",
        result_digest="page-one-result",
    )

    assert graph.merge_readiness(document_task_id).ready is False

    graph.fail_page_task(claims[1].task_id, "PAGE_FAILED", "injected")
    readiness = graph.merge_readiness(document_task_id)
    assert readiness.ready is False
    assert readiness.failed_pages == 1

    graph.requeue_page_task(claims[1].task_id)
    retried = graph.claim_page_tasks("worker-2", limit=1, lease_seconds=30)[0]
    graph.confirm_page_task(
        retried.task_id,
        result_spool_path=tmp_path / "page-2.pickle",
        result_digest="page-two-result",
    )
    assert graph.merge_readiness(document_task_id).ready is True
    assert graph.confirm_merge(document_task_id, "merged-digest") is True
    assert graph.confirm_merge(document_task_id, "merged-digest") is False


def test_p0_01r_page_confirmation_closes_the_active_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id = _add_pdf(database, root_id, root / "attempt.pdf")
    graph = PdfTaskGraphRepository(database)
    document_task_id = graph.plan_document(
        file_id=file_id,
        run_id="run-attempt",
        source_digest="sha256:attempt",
        parser_version="pdf-v1",
        pages=[PdfPagePlan(1, "one", 595.0, 842.0, False)],
        ocr_config_fingerprint="ocr-v1",
    )
    page_task_id = graph.scheduled_page_tasks(document_task_id)[0].task_id
    spool = tmp_path / "page.pickle"
    spool.write_bytes(b"page-result")
    database.mark_task_running(page_task_id)
    database.mark_task_spooled(page_task_id, spool, "spool-digest")

    graph.confirm_page_task(
        page_task_id,
        result_spool_path=spool,
        result_digest="result-digest",
    )

    with database.connect() as connection:
        attempt = connection.execute(
            """
            SELECT status, finished_at
            FROM parse_task_attempts
            WHERE task_id = ?
            ORDER BY attempt_no DESC
            LIMIT 1
            """,
            (page_task_id,),
        ).fetchone()
    assert attempt["status"] == "complete"
    assert attempt["finished_at"]


def test_pdf_page_batch_confirmation_closes_all_tasks_in_one_operation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id = _add_pdf(database, root_id, root / "batch.pdf")
    graph = PdfTaskGraphRepository(database)
    document_task_id = graph.plan_document(
        file_id=file_id,
        run_id="run-batch",
        source_digest="sha256:batch",
        parser_version="pdf-v1",
        pages=[
            PdfPagePlan(page, f"page-{page}", 595.0, 842.0, False)
            for page in range(1, 5)
        ],
        ocr_config_fingerprint="ocr-v1",
    )
    claims = graph.scheduled_page_tasks(document_task_id)
    database.mark_tasks_running([claim.task_id for claim in claims])
    confirmations = []
    for claim in claims:
        spool = tmp_path / f"page-{claim.page_number}.pickle"
        spool.write_bytes(f"page-{claim.page_number}".encode("ascii"))
        confirmations.append(
            (claim.task_id, spool, hashlib.sha256(spool.read_bytes()).hexdigest())
        )

    graph.confirm_page_tasks(confirmations)

    assert graph.merge_readiness(document_task_id).ready
    with database.connect() as connection:
        task_rows = connection.execute(
            """
            SELECT status, confirmed_at FROM parse_tasks
            WHERE parent_task_id = ? AND task_type = 'pdf_native_page'
            """,
            (document_task_id,),
        ).fetchall()
        attempt_rows = connection.execute(
            """
            SELECT pta.status FROM parse_task_attempts AS pta
            JOIN parse_tasks AS pt ON pt.id = pta.task_id
            WHERE pt.parent_task_id = ?
              AND pt.task_type = 'pdf_native_page'
            """,
            (document_task_id,),
        ).fetchall()
    assert len(task_rows) == 4
    assert all(row["status"] == "complete" and row["confirmed_at"] for row in task_rows)
    assert len(attempt_rows) == 4
    assert all(row["status"] == "complete" for row in attempt_rows)


def test_p0_01r_changed_pdf_invalidates_even_completed_old_graph(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id = _add_pdf(database, root_id, root / "changed.pdf")
    graph = PdfTaskGraphRepository(database)
    first = graph.plan_document(
        file_id=file_id,
        run_id="run-1",
        source_digest="sha256:old",
        parser_version="pdf-v1",
        pages=[PdfPagePlan(1, "old-page", 595.0, 842.0, False)],
        ocr_config_fingerprint="ocr-v1",
    )
    claim = graph.claim_page_tasks("worker", limit=1, lease_seconds=30)[0]
    spool = tmp_path / "old-page.pickle"
    spool.write_bytes(b"old-page-result")
    graph.confirm_page_task(
        claim.task_id,
        result_spool_path=spool,
        result_digest="old-digest",
    )
    assert graph.confirm_merge(first, "old-merge") is True

    second = graph.plan_document(
        file_id=file_id,
        run_id="run-2",
        source_digest="sha256:new",
        parser_version="pdf-v1",
        pages=[PdfPagePlan(1, "new-page", 595.0, 842.0, False)],
        ocr_config_fingerprint="ocr-v1",
    )

    with database.connect() as con:
        old_statuses = {
            str(row["status"])
            for row in con.execute(
                """
                SELECT status FROM parse_tasks
                WHERE (id = ? OR parent_task_id = ?)
                """,
                (first, first),
            )
        }
        active_old_identities = int(
            con.execute(
                """
                SELECT COUNT(*) FROM pdf_page_identities
                WHERE file_id = ? AND source_digest = 'sha256:old'
                  AND invalidated_at IS NULL
                """,
                (file_id,),
            ).fetchone()[0]
        )
    assert second != first
    assert old_statuses == {"invalidated"}
    assert active_old_identities == 0


def test_p0_01r_restart_requeues_prior_run_lease_and_reports_ready_merge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id = _add_pdf(database, root_id, root / "recover.pdf")
    first_repository = PdfTaskGraphRepository(database)
    document_task_id = first_repository.plan_document(
        file_id=file_id,
        run_id="run-before-crash",
        source_digest="sha256:recover",
        parser_version="pdf-v1",
        pages=[
            PdfPagePlan(1, "one", 595.0, 842.0, False),
            PdfPagePlan(2, "two", 595.0, 842.0, False),
        ],
        ocr_config_fingerprint="ocr-v1",
    )
    claims = first_repository.claim_page_tasks(
        "crashed-worker",
        limit=2,
        lease_seconds=3600,
    )
    first_spool = tmp_path / "page-1.pickle"
    first_spool.write_bytes(b"page-one")
    first_repository.confirm_page_task(
        claims[0].task_id,
        result_spool_path=first_spool,
        result_digest=__import__("hashlib").sha256(b"page-one").hexdigest(),
    )

    restarted = PdfTaskGraphRepository(database)
    recovered_id = restarted.plan_document(
        file_id=file_id,
        run_id="run-after-crash",
        source_digest="sha256:recover",
        parser_version="pdf-v1",
        pages=[
            PdfPagePlan(1, "one", 595.0, 842.0, False),
            PdfPagePlan(2, "two", 595.0, 842.0, False),
        ],
        ocr_config_fingerprint="ocr-v1",
    )
    recovered_claim = restarted.claim_page_tasks(
        "new-worker",
        limit=1,
        lease_seconds=30,
    )[0]

    assert recovered_id == document_task_id
    assert recovered_claim.task_id == claims[1].task_id
    assert recovered_claim.lease_owner == "new-worker"
    second_spool = tmp_path / "page-2.pickle"
    second_spool.write_bytes(b"page-two")
    restarted.confirm_page_task(
        recovered_claim.task_id,
        result_spool_path=second_spool,
        result_digest=__import__("hashlib").sha256(b"page-two").hexdigest(),
    )
    assert restarted.ready_document_task_ids() == [document_task_id]


def test_p0_01r_corrupt_confirmed_page_spool_is_requeued_on_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id = _add_pdf(database, root_id, root / "corrupt.pdf")
    graph = PdfTaskGraphRepository(database)
    document_task_id = graph.plan_document(
        file_id=file_id,
        run_id="run-1",
        source_digest="sha256:corrupt",
        parser_version="pdf-v1",
        pages=[PdfPagePlan(1, "one", 595.0, 842.0, False)],
        ocr_config_fingerprint="ocr-v1",
    )
    claim = graph.claim_page_tasks("worker", limit=1, lease_seconds=30)[0]
    spool = tmp_path / "corrupt-page.pickle"
    spool.write_bytes(b"before-corruption")
    graph.confirm_page_task(
        claim.task_id,
        result_spool_path=spool,
        result_digest=__import__("hashlib").sha256(b"before-corruption").hexdigest(),
    )
    spool.write_bytes(b"tampered")

    recovered = graph.recover_document(
        document_task_id,
        run_id="run-2",
    )

    assert recovered["requeued_corrupt"] == 1
    assert graph.merge_readiness(document_task_id).ready is False
    assert graph.claim_page_tasks("worker-2", limit=1, lease_seconds=30)


def test_failed_pdf_pages_are_requeued_on_explicit_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id = _add_pdf(database, root_id, root / "retry-failed.pdf")
    graph = PdfTaskGraphRepository(database)
    document_task_id = graph.plan_document(
        file_id=file_id,
        run_id="run-before-failure",
        source_digest="sha256:retry-failed",
        parser_version="pdf-v1",
        pages=[
            PdfPagePlan(1, "one", 595.0, 842.0, False),
            PdfPagePlan(2, "two", 595.0, 842.0, False),
        ],
        ocr_config_fingerprint="ocr-v1",
    )
    failed_task_ids = [
        claim.task_id for claim in graph.scheduled_page_tasks(document_task_id)
    ]
    for task_id in failed_task_ids:
        database.mark_task_failed(
            task_id,
            "FILE_IN_USE",
            "temporary result was locked",
        )

    recovered_id = graph.plan_document(
        file_id=file_id,
        run_id="run-after-failure",
        source_digest="sha256:retry-failed",
        parser_version="pdf-v1",
        pages=[
            PdfPagePlan(1, "one", 595.0, 842.0, False),
            PdfPagePlan(2, "two", 595.0, 842.0, False),
        ],
        ocr_config_fingerprint="ocr-v1",
    )
    recovered = graph.scheduled_page_tasks(recovered_id)

    assert recovered_id == document_task_id
    assert [claim.task_id for claim in recovered] == failed_task_ids
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT run_id, status, error_code, error_message
            FROM parse_tasks
            WHERE id IN (?, ?)
            ORDER BY id
            """,
            tuple(failed_task_ids),
        ).fetchall()
    assert all(row["run_id"] == "run-after-failure" for row in rows)
    assert all(row["status"] == "queued" for row in rows)
    assert all(row["error_code"] is None for row in rows)
    assert all(row["error_message"] is None for row in rows)


def test_p0_01r_real_index_executes_persistent_pdf_page_graph(
    tmp_path: Path,
) -> None:
    import fitz

    root = tmp_path / "root"
    root.mkdir()
    pdf_path = root / "native-pages.pdf"
    document = fitz.open()
    for page_number in range(1, 4):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"PERSISTENT_PDF_PAGE_TASK_{page_number}",
        )
    document.save(pdf_path)
    document.close()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)

    summary = IndexManager(
        database,
        AppSettings(
            enable_ocr=False,
            ocr_scanned_pdf=False,
            pdf_parser_workers=2,
        ),
    ).index_root(root_id)

    with database.connect() as con:
        tasks = con.execute(
            """
            SELECT task_type, unit_key, status, confirmed_at
            FROM parse_tasks
            ORDER BY id
            """
        ).fetchall()
        blocks = con.execute(
            """
            SELECT block_type, page_number, raw_text
            FROM content_blocks
            ORDER BY page_number, block_index
            """
        ).fetchall()
    task_types = [str(row["task_type"]) for row in tasks]
    assert summary.failed == 0
    assert task_types.count("pdf_native_page") == 3
    assert task_types.count("document_merge") == 1
    assert all(
        row["status"] == "complete" and row["confirmed_at"]
        for row in tasks
        if row["task_type"] in {"pdf_native_page", "document_merge"}
    )
    assert [int(row["page_number"]) for row in blocks] == [1, 2, 3]
    assert [
        str(row["raw_text"]).strip() for row in blocks
    ] == [
        "PERSISTENT_PDF_PAGE_TASK_1",
        "PERSISTENT_PDF_PAGE_TASK_2",
        "PERSISTENT_PDF_PAGE_TASK_3",
    ]


def test_p0_01r_real_planner_round_robins_pages_across_pdf_documents(
    tmp_path: Path,
) -> None:
    import fitz

    root = tmp_path / "root"
    root.mkdir()
    paths: list[Path] = []
    for name in ("first.pdf", "second.pdf"):
        path = root / name
        document = fitz.open()
        for page_number in range(1, 4):
            page = document.new_page()
            page.insert_text((72, 72), f"{name}-PAGE-{page_number}")
        document.save(path)
        document.close()
        paths.append(path)
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    manager = IndexManager(
        database,
        AppSettings(enable_ocr=False, ocr_scanned_pdf=False),
    )
    run_id = "fair-run"
    from local_full_text_search.core.index_manager import IndexSummary
    from local_full_text_search.core.task_manager import CancelToken
    from local_full_text_search.models.index_metrics import IndexRunMetrics

    metrics = IndexRunMetrics(run_id=run_id)
    database.start_index_run(metrics)
    jobs = manager._prepare_jobs(
        root_id,
        paths,
        run_id,
        IndexSummary(scanned=2),
        metrics,
        CancelToken(),
    )

    assert [job.file_path.name for job in jobs[:6]] == [
        "first.pdf",
        "second.pdf",
        "first.pdf",
        "second.pdf",
        "first.pdf",
        "second.pdf",
    ]


def test_p0_01r_real_planner_schedules_merge_of_confirmed_spools_after_restart(
    tmp_path: Path,
) -> None:
    import fitz

    from local_full_text_search.core.index_manager import (
        IndexSummary,
        ParseOutcome,
    )
    from local_full_text_search.core.task_manager import CancelToken
    from local_full_text_search.models.content_block import ContentBlock
    from local_full_text_search.models.index_metrics import IndexRunMetrics

    root = tmp_path / "root"
    root.mkdir()
    pdf_path = root / "recover-after-pages.pdf"
    document = fitz.open()
    for page_number in range(1, 3):
        page = document.new_page()
        page.insert_text((72, 72), f"RECOVER_PAGE_{page_number}")
    document.save(pdf_path)
    document.close()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    settings = AppSettings(enable_ocr=False, ocr_scanned_pdf=False)

    first_manager = IndexManager(database, settings)
    first_jobs = first_manager._prepare_jobs(
        root_id,
        [pdf_path],
        "run-before-crash",
        IndexSummary(scanned=1),
        IndexRunMetrics(run_id="run-before-crash"),
        CancelToken(),
    )
    document_task_id = int(first_jobs[0].pdf_document_task_id or 0)
    graph = PdfTaskGraphRepository(database)
    database.mark_tasks_running(
        [int(job.task_id or 0) for job in first_jobs]
    )
    for job in first_jobs:
        block = ContentBlock(
            file_path=str(pdf_path),
            block_index=int(job.pdf_page_number or 0) - 1,
            block_type="pdf_page",
            location_text=f"第 {job.pdf_page_number} 页",
            raw_text=f"RECOVER_PAGE_{job.pdf_page_number}",
            normalized_text=f"recover_page_{job.pdf_page_number}",
            page_number=job.pdf_page_number,
        )
        outcome = ParseOutcome(
            file_id=job.file_id,
            file_path=pdf_path,
            blocks=[block],
            parser_name="pdf",
            status="success",
            task_id=job.task_id,
        )
        spool = tmp_path / f"confirmed-{job.task_id}.pickle"
        with spool.open("wb") as handle:
            pickle.dump(outcome, handle, protocol=pickle.HIGHEST_PROTOCOL)
        graph.confirm_page_task(
            int(job.task_id or 0),
            result_spool_path=spool,
            result_digest=hashlib.sha256(spool.read_bytes()).hexdigest(),
        )
    assert graph.merge_readiness(document_task_id).ready
    first_manager.force_terminate_processes()

    restarted = IndexManager(database, settings)
    recovered_jobs = restarted._prepare_jobs(
        root_id,
        [pdf_path],
        "run-after-crash",
        IndexSummary(scanned=1),
        IndexRunMetrics(run_id="run-after-crash"),
        CancelToken(),
    )
    restarted.force_terminate_processes()

    assert len(recovered_jobs) == 1
    assert recovered_jobs[0].pdf_task_type == "document_merge"
    merged = restarted._record_pdf_page_outcome(
        recovered_jobs[0],
        ParseOutcome(
            file_id=recovered_jobs[0].file_id,
            file_path=pdf_path,
            blocks=[],
            parser_name="pdf_task_graph",
            status="success",
        ),
    )
    assert merged is not None
    merged_outcome, _parent = merged
    assert [block.raw_text for block in merged_outcome.blocks] == [
        "RECOVER_PAGE_1",
        "RECOVER_PAGE_2",
    ]
