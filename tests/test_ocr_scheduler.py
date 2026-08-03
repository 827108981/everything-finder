from __future__ import annotations

from local_full_text_search.core.ocr_scheduler import (
    FairOcrScheduler,
    OcrRequest,
)


def test_p0_02r_ocr_scheduler_round_robins_sources_under_pixel_budget() -> None:
    scheduler = FairOcrScheduler()
    for index in range(3):
        scheduler.submit(
            OcrRequest(
                request_id=f"pdf-{index}",
                source_id="large-pdf",
                source_kind="pdf_page",
                pixel_cost=100,
                payload={"page": index + 1},
            )
        )
    scheduler.submit(
        OcrRequest(
            request_id="image-1",
            source_id="standalone-image",
            source_kind="image",
            pixel_cost=50,
            payload={"path": "image.png"},
        )
    )

    first = scheduler.claim_batch(max_requests=2, max_pixels=200)
    second = scheduler.claim_batch(max_requests=2, max_pixels=200)

    assert [request.source_id for request in first] == [
        "large-pdf",
        "standalone-image",
    ]
    assert [request.request_id for request in second] == ["pdf-1", "pdf-2"]
    assert scheduler.pending_count == 0


def test_p0_02r_one_parent_cannot_fill_all_inflight_slots() -> None:
    scheduler = FairOcrScheduler(max_inflight_per_source=1)
    for index in range(4):
        scheduler.submit(
            OcrRequest(
                request_id=f"huge-{index}",
                source_id="huge-pdf",
                source_kind="pdf_page",
                pixel_cost=100,
                payload={},
            )
        )
    scheduler.submit(
        OcrRequest(
            request_id="small-image",
            source_id="small-image",
            source_kind="image",
            pixel_cost=100,
            payload={},
        )
    )

    claimed = scheduler.claim_batch(max_requests=4, max_pixels=1000)

    assert {request.request_id for request in claimed} == {
        "huge-0",
        "small-image",
    }
    scheduler.confirm("huge-0")
    assert scheduler.claim_batch(max_requests=1, max_pixels=1000)[
        0
    ].request_id == "huge-1"
